"""Cold-prepared physical H2D views for immutable expert parameters.

Quantization buffers loaded in final device strides and dtype are reused.
Other supported layouts are packed once as a compatibility fallback.
Decode submits storage copies, without Tensor.copy_ layout conversion. Owners
stay alive until DMA is drained and the prepared layer is rebuilt.
"""
import torch
from .h2d_transfer import H2DCopyTask


class PreparedH2DUnsupportedLayout(ValueError):
    """The legacy tensor path is required for an unsupported physical layout."""


def _dense_strides(shape, strides):
    expected = 1
    for stride, size in sorted((st, n) for n, st in zip(shape, strides) if n > 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _storage_view(tensor, offset, nbytes):
    storage = tensor.untyped_storage()
    if offset < 0 or nbytes <= 0 or offset + nbytes > storage.nbytes():
        raise ValueError("Prepared H2D storage range is out of bounds")
    return storage[offset:offset + nbytes]


def prepare_mapping_copy(source, destination):
    """Cache a live contiguous mapping copy; the CPU planner edits its bytes."""
    if (source.dtype != destination.dtype or source.shape != destination.shape
            or not source.is_contiguous() or not destination.is_contiguous()):
        raise ValueError("Native mapping DMA requires equal contiguous tensor layouts")
    nbytes = source.numel() * source.element_size()
    return H2DCopyTask(
        _storage_view(source, source.storage_offset() * source.element_size(), nbytes),
        _storage_view(destination, destination.storage_offset() * destination.element_size(), nbytes),
        nbytes, name="log2phy")


class _PreparedQuantField:
    def __init__(self, allocate, sources, target, owners):
        self.originals = tuple(sources)
        sample = target[0]
        shape, strides = tuple(sample.shape), tuple(sample.stride())
        count = sample.numel()
        if not count or not _dense_strides(shape, strides):
            raise PreparedH2DUnsupportedLayout("Quant expert view is not non-overlapping and dense")
        if target.shape[0] > 1 and target.stride(0) < count:
            raise PreparedH2DUnsupportedLayout("Quant expert slots overlap")
        nbytes = count * target.element_size()
        self.destinations = tuple(
            _storage_view(target, (target.storage_offset() + s * target.stride(0)) * target.element_size(), nbytes)
            for s in range(target.shape[0]))
        if any(t.device.type != "cpu" or t.numel() != count for t in sources):
            raise ValueError("Prepared quant sources must be equal-sized CPU expert tensors")
        self.versions = None
        same_layout = all(
            t.dtype == target.dtype and tuple(t.shape) == shape
            and all(n <= 1 or a == b for n, a, b in zip(shape, t.stride(), strides))
            for t in sources)
        direct = same_layout or (
            all(t.dtype == target.dtype and t.is_contiguous() for t in sources)
            and sample.is_contiguous())
        if direct:
            self.logical_sources = tuple(t.reshape(shape) for t in sources)
            self.sources = tuple(_storage_view(t, t.storage_offset() * t.element_size(), nbytes) for t in sources)
        else:
            slab = allocate((len(sources) * count,), target.dtype)
            self.logical_sources = tuple(slab.as_strided(shape, strides, e * count) for e in range(len(sources)))
            for packed, source in zip(self.logical_sources, sources):
                try:
                    packed.copy_(source.reshape(shape))
                except (NotImplementedError, RuntimeError) as error:
                    if "not implemented" not in str(error).lower() and "not supported" not in str(error).lower():
                        raise
                    raise PreparedH2DUnsupportedLayout("Quant CPU dtype conversion is unsupported") from error
            self.sources = tuple(_storage_view(slab, e * nbytes, nbytes) for e in range(len(sources)))
            owners.append(slab)
            versions = []
            for source in sources:
                try:
                    versions.append(source._version)
                except RuntimeError as error:
                    if "version counter" not in str(error):
                        raise
                    versions.append(None)
            self.versions = tuple(versions)
        owners.extend((target, *sources, *self.logical_sources))
        self.nbytes = nbytes

    def validate(self, eid):
        if self.versions is not None and self.versions[eid] is not None:
            if self.originals[eid]._version != self.versions[eid]:
                raise RuntimeError("Packed quant source changed; drain DMA and rebuild the prepared layer")


class PreparedLayerH2D:
    def __init__(self, manager, layer, layer_idx, weight):
        self.layer = layer
        self.fields = []
        self.quant_fields = []
        self.owners = []
        self.num_experts = manager.num_total_experts
        self.num_slots = weight(layer, "w13_weight").shape[0]
        self.names = ()
        self.provides_fp32_scale = False
        if self.num_slots == 0:
            return
        for which, name, size in (("w13", "w13_weight", manager.w13_expert_size_bytes),
                                 ("w2", "w2_weight", manager.w2_expert_size_bytes)):
            target = weight(layer, name)
            sources = getattr(manager, which + "_weights_cpu")[layer_idx]
            if target.shape[0] != self.num_slots or len(sources) != self.num_experts:
                raise ValueError("Prepared H2D expert/slot count mismatch")
            self.owners.extend((target, *sources))
            base = target.storage_offset() * target.element_size()
            destinations = tuple(_storage_view(target, base + s * size, size) for s in range(self.num_slots))
            self.fields.append((tuple(_storage_view(t, t.storage_offset() * t.element_size(), size) for t in sources),
                                destinations, (size,) * self.num_experts, which))
        scale_values = None
        for buffers in (manager.scale_cpu_buffers, manager.offset_cpu_buffers, manager.scale_bias_cpu_buffers):
            for name, layers in buffers.items():
                target = getattr(layer, name, None)
                if target is None or layer_idx >= len(layers):
                    continue
                field = self._add_quant(manager, target, layers[layer_idx], name)
                if name == "w13_weight_scale":
                    scale_values = field.logical_sources
        derived = getattr(layer, "w13_weight_scale_fp32", None)
        if derived is not None:
            if scale_values is None:
                raise PreparedH2DUnsupportedLayout("Derived FP32 scale has no prepared source")
            # Match source->device-scale->FP32 conversion, including rounding.
            self._add_quant(manager, derived, scale_values, "w13_weight_scale_fp32")
            self.provides_fp32_scale = True
        self.names = tuple(tuple(f"{field}[L{layer_idx},E{e}->S" for e in range(self.num_experts))
                           for _, _, _, field in self.fields)

    def _add_quant(self, manager, target, sources, name):
        if target.shape[0] != self.num_slots or len(sources) != self.num_experts:
            raise ValueError("Prepared H2D quant expert/slot count mismatch")
        field = _PreparedQuantField(manager._allocate_expert_host_tensor, sources, target, self.owners)
        self.quant_fields.append(field)
        self.fields.append((field.sources, field.destinations, (field.nbytes,) * self.num_experts, name))
        return field

    def tasks(self, eid, slot):
        if not 0 <= eid < self.num_experts or not 0 <= slot < self.num_slots:
            raise ValueError("Prepared H2D expert/slot out of range")
        for field in self.quant_fields:
            field.validate(eid)
        return [H2DCopyTask(sources[eid], destinations[slot], sizes[eid], name=names[eid] + str(slot) + "]")
                for (sources, destinations, sizes, _), names in zip(self.fields, self.names)]
