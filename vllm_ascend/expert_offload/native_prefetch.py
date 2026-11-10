"""Pinned prefetch channels sharing the native CPU dispatcher's monitor.

Each capture site owns its input, mapping, sequence and completion event.
Only actual demand routes update LRC; speculative work reads target history.
"""

import logging

import torch
import torch_npu

_LOG = logging.getLogger(__name__)
_MAX_SITES = 64


_NATIVE_WAIT_OPS = None


def _load_native_wait():
    global _NATIVE_WAIT_OPS
    if _NATIVE_WAIT_OPS is None:
        import importlib
        import importlib.util
        import sys
        import sysconfig
        from pathlib import Path
        try:
            _NATIVE_WAIT_OPS = importlib.import_module("_vllm_ascend_prefetch_wait")
        except ModuleNotFoundError as error:
            if error.name != "_vllm_ascend_prefetch_wait":
                raise
            import _vllm_ascend_lrc
            path = Path(_vllm_ascend_lrc.__file__).with_name(
                "_vllm_ascend_prefetch_wait" + sysconfig.get_config_var("EXT_SUFFIX"))
            if not path.is_file():
                raise ImportError("Build native prefetch wait with: cd csrc/lrc_policy && "
                                  "python setup_npu_wait.py build_ext --inplace") from error
            spec = importlib.util.spec_from_file_location("_vllm_ascend_prefetch_wait", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[spec.name] = module
            _NATIVE_WAIT_OPS = module
    return _NATIVE_WAIT_OPS


class NativePrefetchReadyEvent:
    """Owned device flag using one native protocol for reset, signal and wait."""

    def __init__(self):
        self._ops = _load_native_wait()
        self._flag = torch.zeros(32, dtype=torch.uint8, device="npu")

    @property
    def npu_event(self):
        return self._flag.data_ptr()

    def _write(self, value, stream):
        if stream is None:
            stream = torch_npu.npu.current_stream()
        self._ops.write_value(self.npu_event, value, stream.stream_id, stream.device_index, stream.device_type)

    def reset(self, stream=None):
        self._write(0, stream)

    def record(self, stream=None):
        self._write(1, stream)

    def wait(self, stream=None):
        if stream is None:
            stream = torch_npu.npu.current_stream()
        self._ops.wait_value(int(self.npu_event), stream.stream_id, stream.device_index, stream.device_type)


class NativePrefetchState:
    def __init__(self, transfer):
        self.transfer = transfer
        self.device_flags = torch.zeros((_MAX_SITES, 16), dtype=torch.int32, device=transfer.layer.log2phy.device)
        self.host_flags = torch.zeros((_MAX_SITES, 16), dtype=torch.int32, pin_memory=True)
        self.flags = [(self.device_flags[i, :1], self.host_flags[i, :1]) for i in range(_MAX_SITES)]
        self.sites = []
        self.eager = {}
        self.by_mapping = {}

    def stage(self, predicted, capturing, dependency=None):
        from .cpu_backend.kt_kernel_backend import _graph_capture_active

        # npugraph_ex can capture outside the Python context wrapper. Each
        # real capture site must own its event even when that flag is false.
        capturing = capturing or _graph_capture_active()
        weights, ids = predicted
        key = (tuple(ids.shape), dependency)
        site = None if capturing else self.eager.get(key)
        if site is None:
            if len(self.sites) >= _MAX_SITES:
                raise RuntimeError("Native prefetch capture-site limit reached")
            site = NativePrefetchSite(self.transfer, predicted, self.flags[len(self.sites)], capturing=capturing)
            site.dependency = dependency
            self.sites.append(site)
            self.by_mapping[site.mapping.data_ptr()] = site
            if not capturing:
                self.eager[key] = site
        stream = torch_npu.npu.current_stream()
        site.ready.reset(stream)
        site.ids.copy_(ids.to(torch.int32), non_blocking=True)
        if weights is None:
            site.weights.fill_(1.0)
        else:
            site.weights.copy_(weights.to(torch.float32), non_blocking=True)
        site.mapping.copy_(self.transfer.layer.log2phy, non_blocking=True)
        return site


class NativePrefetchSite:
    def __init__(self, transfer, predicted, flags, capturing=False):
        self.capturing = capturing
        weights, ids = predicted
        self.transfer = transfer
        self.ids = torch.empty(tuple(ids.shape), dtype=torch.int32, pin_memory=True)
        self.weights = torch.empty(tuple(ids.shape), dtype=torch.float32, pin_memory=True)
        self.mapping = torch.empty(tuple(transfer.layer.log2phy.shape), dtype=transfer.layer.log2phy.dtype,
                                   pin_memory=True)
        from .prepared_h2d import prepare_mapping_copy
        self.mapping_task = prepare_mapping_copy(self.mapping, transfer.layer.log2phy)
        self.ready = NativePrefetchReadyEvent()
        if not self.ready.npu_event:
            raise RuntimeError("Native prefetch requires value-based ExternalEvent support")
        self.device_flag, self.host_flag = flags
        self.registered = False
        self.dependency = None

    def dispatch(self, manager):
        from .cpu_backend import poll_submission

        worker = poll_submission._WORKER
        if worker is None or not hasattr(worker, "register_prefetch"):
            raise RuntimeError("Native prefetch requires the initialized native CPU monitor")
        if not self.registered:
            worker.register_prefetch(self, manager.prefetch_topk)
            self.registered = True
        # On the staging stream: reset + routes/mapping D2H precede notification.
        # No ACL Python report callback and no host synchronization.
        self.device_flag.add_(1)
        self.host_flag.copy_(self.device_flag, non_blocking=True)
        with manager._prefetch_state_lock:
            manager._prefetch_layer_npu_event[self.transfer.layer_idx] = self.ready


def stage_native_prefetch(manager, next_idx, predicted, capturing):
    layer = manager.moe_layers[next_idx]
    executor = getattr(layer, "_hybrid_expert_executor", None)
    transfer = getattr(getattr(executor, "backend", None), "_host_transfer", None)
    state = getattr(transfer, "native_prefetch", None)
    if state is None:
        raise RuntimeError("Missing native prefetch target transfer")
    context = getattr(manager, "_prefetch_demand_context", None)
    site = state.stage(predicted, capturing, getattr(context, "native_channel", None))
    return site.ids, site.weights, site.mapping, site.mapping.numpy(), layer, next_idx, None


def dispatch_native_prefetch(manager, staged):
    ids, weights, mapping, _, layer, next_idx, _ = staged
    state = layer._hybrid_expert_executor.backend._host_transfer.native_prefetch
    state.by_mapping[mapping.data_ptr()].dispatch(manager)
