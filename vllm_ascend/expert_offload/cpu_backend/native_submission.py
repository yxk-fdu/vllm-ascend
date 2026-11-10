"""Register pinned tasks once; native poll/compute/completion never enter Python."""

import atexit
import logging
import os
import threading
import time

import torch
import torch_npu

from vllm_ascend.expert_offload.host_transfer import WeightRequest

_LOG = logging.getLogger(__name__)


class NativePollWorker:
    def __init__(self, device):
        from kt_kernel import kt_kernel_ext
        from _vllm_ascend_lrc import NativeCpuDispatcher

        self.device = device
        self.stream = torch_npu.npu.SyncLaunchStream(device=device)
        self.weight_stream = torch_npu.npu.SyncLaunchStream(device=device)
        self.engine = NativeCpuDispatcher(
            kt_kernel_ext.native_dispatch_api(), int(self.stream.npu_stream), int(self.weight_stream.npu_stream),
            int(os.environ.get("KT_MOE_POLL_CPU", "6")),
            int(os.environ.get("KT_MOE_COMPLETION_CPU", "7")))
        from ..dma_submit_timing import DmaSubmitTrace
        self.dma_trace = DmaSubmitTrace()
        self.channels = []
        self.prefetch_sites = {}
        self._closed = False
        self._close_lock = threading.Lock()
        self.dma_thread = threading.Thread(target=self._dma, name="kt_native_dma", daemon=True)
        self.dma_thread.start()
        atexit.register(self.close)
        _LOG.info("CPU MoE native dispatch: input flag -> C++ planner -> CPUInfer; "
                  "native completion records the external event; Python handles weight DMA only")

    def register(self, backend, hidden, stream, ready, hostflag, before_submit=None):
        from kt_kernel import kt_kernel_ext
        from kt_kernel.experts_base import KExpertsCPUBuffer

        context = getattr(before_submit, "__self__", None)
        if context is None or context.transfer.native_planner is None:
            raise RuntimeError("Native dispatch requires a registered native host admission context")
        wrapper = backend._wrapper
        if wrapper.max_deferred_experts_per_token:
            raise ValueError("Native dispatch does not support deferred CPU experts")
        flat = hidden.view(-1, hidden.shape[-1])
        inputs, ids, _, weights, outputs, batches, _ = KExpertsCPUBuffer.get_buffer(flat, backend.config.top_k)
        slot = wrapper.layer_idx % KExpertsCPUBuffer.buffer_depth
        if weights[slot].dtype != torch.float32 or not ids[slot].is_contiguous() or not context.mapping.is_contiguous():
            raise ValueError("Native dispatch requires contiguous IDs/mapping and float32 route weights")
        task = wrapper.moe.forward_task(
            batches[slot].data_ptr(), ids[slot].size(-1), ids[slot].data_ptr(),
            weights[slot].data_ptr(), inputs[slot].data_ptr(), outputs[slot].data_ptr(), False)
        # Owners remain alive until graphs have retired and native threads joined.
        index = len(self.channels)
        self.channels.append((backend, hidden, ready, hostflag, context, task))
        native = self.engine.add(
            hostflag.data_ptr(), kt_kernel_ext.native_cpu_handle(wrapper.cpu_infer), task,
            ids[slot].data_ptr(), weights[slot].data_ptr(), context.mapping.data_ptr(),
            int(ready.npu_event), int(context.ready.npu_event), flat.shape[0], ids[slot].shape[-1], backend.config.layer_idx,
            ids[slot].element_size(), context.mapping.element_size(), context.transfer.native_planner)
        if native != index:
            raise RuntimeError("Native task registration mismatch")
        context.native_channel = index

    def register_prefetch(self, site, cap):
        index = len(self.channels)
        self.channels.append((None, None, site.ready, site.host_flag, site, None))
        self.prefetch_sites[index] = site
        register = self.engine.add_prefetch_after if site.dependency is not None else self.engine.add_prefetch
        dependency = (site.dependency,) if site.dependency is not None else ()
        native = register(
            site.host_flag.data_ptr(), site.ids.data_ptr(), site.weights.data_ptr(), site.mapping.data_ptr(),
            int(site.ready.npu_event), site.ids.shape[0], site.ids.shape[1], site.transfer.layer_idx,
            site.ids.element_size(), site.mapping.element_size(), cap, site.transfer.native_planner, *dependency)
        if native != index:
            raise RuntimeError("Native prefetch registration mismatch")

    @torch.inference_mode()
    def _dma(self):
        try:
            torch.npu.set_device(self.device)
            bound = False
            while True:
                index, loads, task, queued, dequeued, dequeued_cpu = self.engine.next_weight_details()
                resumed, resumed_cpu = time.monotonic_ns(), time.thread_time_ns()
                timing = None
                if index < 4096 and self.dma_trace.enabled:
                    timing = self.dma_trace.begin({
                        'event': 'cpu_moe.dma_submit', 'channel': index, 'task_id': task,
                        'experts': len(loads), 'prefetch': index in self.prefetch_sites,
                        'cpp_enqueue_ns': queued, 'cpp_dequeue_ns': dequeued,
                        'cpp_dequeue_cpu_ns': dequeued_cpu, 'python_resume_ns': resumed,
                        'python_resume_cpu_ns': resumed_cpu, 'native_tid': threading.get_native_id(),
                        'wall_ns': time.time_ns(), 'reference_mono_ns': time.monotonic_ns()})
                if index >= 4096:
                    return
                context = self.channels[index][4]
                if not bound:
                    cpu = context.transfer.native_dma_cpu
                    os.sched_setaffinity(0, {cpu})
                    bound = True
                # This stage is independent of native CPU submit and completion.
                if index in self.prefetch_sites:
                    manager = context.transfer.worker.manager
                    manager._host_prefetch_calls += 1
                    manager._host_prefetch_loads += len(loads)
                    if manager._host_prefetch_calls == 1 or manager._host_prefetch_calls % 1024 == 0:
                        _LOG.info("[NATIVE-PREFETCH] transfers=%d experts=%d native_notifications=%d planned_experts=%d",
                                  manager._host_prefetch_calls, manager._host_prefetch_loads,
                                  *self.engine.prefetch_stats())
                if timing is not None:
                    timing['layer'] = context.transfer.layer_idx
                try:
                    context.transfer.worker._process(WeightRequest(
                        context.transfer, context.mapping, context.transfer.layer.log2phy,
                        loads, context.ready, timing, context.mapping_task))
                finally:
                    self.dma_trace.end(timing)
        except BaseException:
            _LOG.exception("Native CPU MoE weight DMA failed; refusing stale output")
            os._exit(1)

    def set_dma_capture(self, enabled, directory):
        self.dma_trace.set_capture(enabled, directory)

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            torch.npu.synchronize(self.device)
            self.engine.close()
            self.dma_thread.join(timeout=5)
            if self.dma_thread.is_alive():
                raise RuntimeError("Native DMA worker did not retire")
            _LOG.info("CPU MoE native dispatcher retired: submitted=%d completed=%d", *self.engine.stats())
            self._closed = True
