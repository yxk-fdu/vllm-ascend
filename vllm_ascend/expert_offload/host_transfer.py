"""Host planning/DMA with a weight event independent of CPU output readiness."""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
from dataclasses import dataclass

import torch
import torch_npu

from .host_transfer_policy import HostTransferPolicy
from .dma_submit_timing import stamp
from .prepared_h2d import prepare_mapping_copy

_LOG = logging.getLogger(__name__)


@dataclass
class WeightRequest:
    transfer: object
    mapping_host: object
    mapping_device: object
    loads: tuple
    ready: object
    timing: object = None
    mapping_task: object = None
    recorded: threading.Event | None = None


class HostWeightWorker:
    """One uncaptured SyncLaunchStream; never waits for CPU expert compute."""

    def __init__(self, manager):
        self.manager = manager
        self.device = torch.npu.current_device()
        self.stream = torch_npu.npu.SyncLaunchStream(device=self.device)
        self.jobs = queue.SimpleQueue()
        self._close_lock = threading.Lock()
        self._closed = False
        self._error = None
        self._started = threading.Event()
        self.thread = threading.Thread(target=self._run, name="cpu_expert_dma", daemon=True)
        self.thread.start()
        self._started.wait()
        if self._error is not None:
            raise RuntimeError(f"Failed to initialize CPU MoE DMA worker on device {self.device}") from self._error
        atexit.register(self.close)

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("Expert DMA failed; weight readiness was not published") from self._error
        if self._closed:
            raise RuntimeError("Expert DMA worker is closed")

    def submit(self, request):
        self.raise_if_failed()
        self.jobs.put(request)

    def _process(self, request):
        transfer = request.transfer
        timing = getattr(request, 'timing', None)
        stamp(timing, 'process_enter')
        with torch.npu.stream(self.stream):
            stamp(timing, 'stream_enter')
            if timing is None:
                self.manager._load_expert_weights_into_slots(transfer.layer, transfer.layer_idx, request.loads)
            else:
                self.manager._load_expert_weights_into_slots(transfer.layer, transfer.layer_idx, request.loads, timing=timing)
            mapping_task = request.mapping_task
            if mapping_task is None:
                mapping_task = prepare_mapping_copy(request.mapping_host, request.mapping_device)
            self.manager._get_h2d_transport().copy_batch((mapping_task,))
            stamp(timing, 'mapping_return')
            # Record on DMA itself after weights, quant attributes and mapping.
            # Never use the independent CPU-output completion stream here.
            request.ready.record(self.stream)
            if request.recorded is not None:
                request.recorded.set()
            stamp(timing, 'ready_record_return')

    @torch.inference_mode()
    def _run(self):
        try:
            torch.npu.set_device(self.device)
            cpu = getattr(getattr(self.manager, "offload_config", None), "cpu_moe", None)
            if getattr(cpu, "enabled", False):
                affinity = getattr(cpu, "startup_affinity", None)
                core = affinity.dma_cpu if affinity else cpu.native_dispatch_dma_cpu
                os.sched_setaffinity(0, {core if core is not None else int(os.environ.get("KT_MOE_COMPLETION_CPU", "7"))})
        except BaseException as exc:
            self._error = exc
        finally:
            self._started.set()
        if self._error is not None:
            return

        while True:
            request = self.jobs.get()
            if request is None:
                return
            try:
                self._process(request)
            except BaseException as exc:
                self._error = exc
                # The NPU may already be waiting on this request. Releasing
                # the event would let it consume incomplete expert weights.
                _LOG.exception(
                    "Expert DMA failed on device %s, layer %s; terminating the worker with weight readiness unpublished",
                    self.device, request.transfer.layer_idx,
                )
                os._exit(1)
            del request

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self.raise_if_failed()
            # Poll/CPU/DMA consumers remain alive until live replays retire.
            torch.npu.synchronize(self.device)
            self.jobs.put(None)
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("Expert DMA worker still owns live requests")
            self.manager._synchronize_h2d(self.stream)
            self._closed = True
            atexit.unregister(self.close)


class HostTransferContext:
    """Capture-site owners, kept alive for all replays of their graph."""

    def __init__(self, transfer, hidden, *, eager=False):
        self.transfer, self.hidden = transfer, hidden
        self.mapping = torch.empty(
            transfer.layer.log2phy.shape, dtype=transfer.layer.log2phy.dtype, device="cpu", pin_memory=True
        )
        self.mapping_task = prepare_mapping_copy(self.mapping, transfer.layer.log2phy)
        self.recorded = threading.Event() if eager else None
        if eager:
            self.ready = torch.npu.Event()
        else:
            self.ready = torch_npu.npu.ExternalEvent()
            if not self.ready.npu_event:
                raise RuntimeError("Host DMA requires value-based ExternalEvent support")

    def stage(self, stream):
        if self.recorded is None:
            # Captured reset precedes sequence publication on every replay.
            self.ready.reset(stream)
        else:
            self.recorded.clear()
        self.mapping.copy_(self.transfer.layer.log2phy, non_blocking=True)

    def wait_for_eager_weights(self, stream):
        # A normal event must be recorded before the consumer queues its wait.
        # The worker signals after enqueueing the record, while DMA can still be in flight.
        assert self.recorded is not None
        self.recorded.wait()
        self.transfer.worker.raise_if_failed()
        stream.wait_event(self.ready)

    def prepare(self):
        transfer = self.transfer
        transfer.worker.raise_if_failed()
        ids, weights = transfer.backend._pinned_route_buffers(self.hidden)
        if getattr(transfer, "native_planner", None) is not None:
            loads = transfer.native_planner.prepare(
                ids.data_ptr(), weights.data_ptr(), self.mapping.data_ptr(),
                ids.shape[0], ids.shape[1], ids.element_size(), self.mapping.element_size())
        else:
            plan = transfer.policy.plan(ids.tolist(), weights.float().tolist(), self.mapping.tolist())
            # Keep the original device routes. Mask only the pinned inputs used by CPU compute.
            ids.copy_(torch.tensor(plan.cpu_ids, dtype=ids.dtype).reshape(ids.shape))
            weights.copy_(torch.tensor(plan.cpu_weights, dtype=weights.dtype).reshape(weights.shape))
            self.mapping.copy_(torch.tensor(plan.mapping, dtype=self.mapping.dtype))
            loads = plan.loads
        transfer.worker.submit(WeightRequest(
            transfer, self.mapping, transfer.layer.log2phy, loads, self.ready,
            mapping_task=self.mapping_task, recorded=self.recorded,
        ))


class HostExpertTransfer:
    def __init__(self, manager, layer, backend, worker, *, calibration_cache=None):
        self.layer, self.backend, self.worker = layer, backend, worker
        self.layer_idx = manager.moe_layers.index(layer)
        # Also covers draft layers attached after the manager's finalization.
        manager._prepare_expert_h2d_layer(layer, self.layer_idx)
        self._eager_contexts = {}
        costs = None
        if calibration_cache is not None:
            from .cpu_moe_calibration import calibrate_cpu_moe

            plan = manager._prepared_expert_h2d.get(self.layer_idx)
            if plan is None or not plan.fields:
                raise RuntimeError("CPU MoE calibration requires prepared expert weight and scale sources")
            config = backend.config
            key = (config.hidden_size, config.intermediate_size, config.top_k,
                   tuple(sizes[0] for _, _, sizes, _ in plan.fields))
            costs = calibration_cache.get(key)
            if costs is None:
                costs = calibrate_cpu_moe(
                    backend, plan, manager._get_h2d_transport(), worker.stream, layer.log2phy.device,
                )
                calibration_cache[key] = costs
                _LOG.info(
                    "CPU MoE cost calibration for layer %s: one expert=%.3f us, "
                    "one expert/two tokens=%.3f us, additional route=%.3f us, one H2D expert=%.3f us",
                    config.layer_idx, costs.first_route_ns / 1000, costs.two_routes_ns / 1000,
                    costs.extra_route_ns / 1000, costs.copy_expert_ns / 1000,
                )
        self.policy = HostTransferPolicy(
            experts=manager.num_total_experts,
            slots=manager.num_device_experts_for_layer(self.layer_idx),
            config=manager.offload_config.cpu_moe,
            lrc=manager.cache_policy,
            layer_idx=self.layer_idx,
            costs=costs,
        )

        cpu = manager.offload_config.cpu_moe
        affinity = getattr(cpu, "startup_affinity", None)
        self.native_dma_cpu = affinity.dma_cpu if affinity else getattr(cpu, "native_dispatch_dma_cpu", None)
        if self.native_dma_cpu is None:
            self.native_dma_cpu = int(os.environ.get("KT_MOE_COMPLETION_CPU", "7"))
        from _vllm_ascend_lrc import NativeHostPlanner
        policy = self.policy
        if policy.lrc is not None and not hasattr(policy.lrc, "_native"):
            raise ValueError("Native CPU dispatch requires C++ LRC when cache policy is enabled")
        self.native_planner = NativeHostPlanner(
            policy.experts, policy.slots, policy.layer_idx, policy.cpu_unit,
            policy.cpu_unit + policy.copy_unit, policy.config.balance_max_transfers,
            getattr(policy.lrc, "_native", None), policy.scores,
            policy.npu_fixed_unit, policy.cpu_extra_unit)

    def context(self, hidden, *, eager=False):
        if not eager:
            return HostTransferContext(self, hidden)
        # Retain pinned mapping/event owners beyond asynchronous eager H2D.
        # The next use first synchronizes its input stream after prior work.
        key = (tuple(hidden.shape), hidden.dtype)
        context = self._eager_contexts.get(key)
        if context is None:
            context = HostTransferContext(self, hidden, eager=True)
            self._eager_contexts[key] = context
        context.hidden = hidden
        return context


def attach_host_transfers(manager):
    """Called after weights/hot preload are final, before warmup/capture."""
    cpu = manager.offload_config.cpu_moe
    if not cpu.enabled:
        return
    if manager.ep_size != 1 or manager.enable_multi_card:
        raise ValueError("Host hybrid DMA currently requires single-card execution")
    from .h2d_transfer import TorchCopyH2DTransport

    transport = manager._get_h2d_transport()
    base = transport
    if not isinstance(base, TorchCopyH2DTransport):
        raise ValueError("Host hybrid DMA requires h2d_backend=torch")
    manager._host_weight_worker = HostWeightWorker(manager)
    calibration_cache = {}
    for layer in manager.moe_layers:
        executor = getattr(layer, "_hybrid_expert_executor", None)
        if executor is not None:
            executor.backend._host_transfer = HostExpertTransfer(
                manager, layer, executor.backend, manager._host_weight_worker,
                calibration_cache=calibration_cache,
            )
            if manager.offload_config.expert_prefetch_enabled:
                from .native_prefetch import NativeRouteState
                executor.backend._host_transfer.native_prefetch = NativeRouteState(executor.backend._host_transfer)
