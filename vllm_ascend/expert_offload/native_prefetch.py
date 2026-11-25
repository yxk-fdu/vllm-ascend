"""Pinned demand and prefetch channels sharing the native dispatcher's monitor.

Each capture site owns its input, mapping, sequence and completion event.
Only actual demand routes update LRC; speculative work reads target history.
"""

import os

import torch
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

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


class NativeRouteState:
    def __init__(self, transfer):
        self.transfer = transfer
        self.device_flags = torch.zeros((_MAX_SITES, 16), dtype=torch.int32, device=transfer.layer.log2phy.device)
        self.host_flags = torch.zeros((_MAX_SITES, 16), dtype=torch.int32, pin_memory=True)
        self.flags = [(self.device_flags[i, :1], self.host_flags[i, :1]) for i in range(_MAX_SITES)]
        self.sites = []
        self.eager = {}

    def stage(self, predicted, capturing, dependency=None):
        # npugraph_ex can capture outside the Python context wrapper. Each
        # real capture site must own its event even when that flag is false.
        capturing = capturing or torch.npu.is_current_stream_capturing()
        weights, ids = predicted
        key = (tuple(ids.shape), dependency)
        site = None if capturing else self.eager.get(key)
        if site is None:
            if len(self.sites) >= _MAX_SITES:
                raise RuntimeError("Native MoE capture-site limit reached")
            site = NativeRouteSite(self.transfer, ids.shape, self.flags[len(self.sites)])
            site.dependency = dependency
            self.sites.append(site)
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


class NativeRouteSite:
    def __init__(self, transfer, shape, flags):
        self.transfer = transfer
        self.ids = torch.empty(shape, dtype=torch.int32, pin_memory=True)
        self.weights = torch.empty(shape, dtype=torch.float32, pin_memory=True)
        self.mapping = torch.empty(tuple(transfer.layer.log2phy.shape), dtype=transfer.layer.log2phy.dtype,
                                   pin_memory=True)
        self.ready = NativePrefetchReadyEvent()
        if not self.ready.npu_event:
            raise RuntimeError("Native MoE requires value-based readiness support")
        self.device_flag, self.host_flag = flags
        self.native_channel = None
        self.dependency = None

    def publish(self):
        # Reset and route/mapping D2H are already queued on this stream.
        self.device_flag.add_(1)
        self.host_flag.copy_(self.device_flag, non_blocking=True)

    def dispatch_prefetch(self, manager):
        worker = getattr(manager, "_native_dma_worker", None)
        if worker is None and manager._host_hybrid_enabled:
            from .cpu_backend import poll_submission

            worker = poll_submission._WORKER
        if worker is None:
            raise RuntimeError("Native prefetch requires an initialized DMA dispatcher")
        if self.native_channel is None:
            self.native_channel = worker.register_prefetch(self, manager.prefetch_topk)
        self.publish()
        with manager._prefetch_state_lock:
            manager._prefetch_layer_npu_event[self.transfer.layer_idx] = self.ready


class NativeNpuTransfer:
    """A layer's planner, used for both demand and predictions without CPU FFN."""

    def __init__(self, manager, layer, layer_idx, worker):
        from _vllm_ascend_lrc import NativeHostPlanner

        manager._prepare_expert_h2d_layer(layer, layer_idx)
        plan = manager._prepared_expert_h2d.get(layer_idx)
        if plan is None or not plan.fields:
            raise ValueError(f"native_expert_dma requires prepared weights and scales for layer {layer_idx}")
        self.layer = layer
        self.layer_idx = layer_idx
        self.worker = worker
        self.native_dma_cpu = worker.dma_cpu
        self.native_planner = NativeHostPlanner(
            plan.num_experts, plan.num_slots, layer_idx, 0, 1, 0,
            getattr(manager.cache_policy, "_native", None), [0.0] * plan.num_experts)
        self.demand = NativeRouteState(self)
        if manager.offload_config.expert_prefetch_enabled:
            self.native_prefetch = NativeRouteState(self)

    def wait_for_prefetch(self):
        manager = self.worker.manager
        with manager._prefetch_state_lock:
            pending = manager._prefetch_layer_npu_event.pop(self.layer_idx, None)
        if pending is not None:
            pending.wait()

    def submit(self, topk_weights, topk_ids):
        manager = self.worker.manager
        site = self.demand.stage((topk_weights, topk_ids), bool(_EXTRA_CTX.capturing))
        dispatcher = manager._native_dma_worker
        if site.native_channel is None:
            site.native_channel = dispatcher.register_npu_demand(site)
        site.publish()
        return site


def start_native_npu_dma(manager):
    from types import SimpleNamespace

    from .cpu_backend.native_submission import NativePollWorker

    if manager._native_dma_worker is not None:
        return manager._native_npu_dma
    device = torch.npu.current_device()
    allowed = sorted(os.sched_getaffinity(0))
    dma_cpu = manager.offload_config.cpu_moe.native_dispatch_dma_cpu
    if dma_cpu is None:
        dma_cpu = allowed[-1]
    poll_cpu = next((cpu for cpu in allowed if cpu != dma_cpu), allowed[0])
    if "KT_MOE_POLL_CPU" in os.environ:
        poll_cpu = int(os.environ["KT_MOE_POLL_CPU"])
    if poll_cpu not in allowed:
        raise ValueError("Native expert poll CPU is outside the process CPU affinity")
    if poll_cpu == dma_cpu:
        raise ValueError("Native expert DMA requires distinct poll and DMA CPUs")
    stream = torch_npu.npu.SyncLaunchStream(device=device)
    manager._native_dma_worker = NativePollWorker(device, poll_cpu, completion_cpu=-1, cpu_compute=False)
    manager._native_npu_dma = SimpleNamespace(manager=manager, stream=stream, dma_cpu=dma_cpu)
    return manager._native_npu_dma


def get_native_npu_transfer(manager, layer):
    layer_idx = layer.offload_layer_idx
    transfer = manager._native_npu_transfers.get(layer_idx)
    if transfer is None:
        from vllm_ascend.quantization.quant_type import QuantType

        if layer.quant_type != QuantType.W4A8MXFP:
            raise ValueError("native_expert_dma currently supports W4A8MXFP MoE layers")
        transfer = NativeNpuTransfer(manager, layer, layer_idx, start_native_npu_dma(manager))
        manager._native_npu_transfers[layer_idx] = transfer
    return transfer


def native_prefetch_state(manager, layer):
    if manager._native_npu_dma_enabled:
        return getattr(get_native_npu_transfer(manager, layer), "native_prefetch", None)
    executor = getattr(layer, "_hybrid_expert_executor", None)
    transfer = getattr(getattr(executor, "backend", None), "_host_transfer", None)
    return getattr(transfer, "native_prefetch", None)


def stage_native_prefetch(manager, next_idx, predicted, capturing):
    layer = manager.moe_layers[next_idx]
    state = native_prefetch_state(manager, layer)
    if state is None:
        return None
    context = getattr(manager, "_prefetch_demand_context", None)
    site = state.stage(predicted, capturing, getattr(context, "native_channel", None))
    return site


def prefetch_after_demand(manager, layer, hidden_states, demand):
    previous = getattr(manager, "_prefetch_demand_context", None)
    manager._prefetch_demand_context = demand
    try:
        manager.trigger_next_layer_prefetch(layer, hidden_states)
    finally:
        manager._prefetch_demand_context = previous
