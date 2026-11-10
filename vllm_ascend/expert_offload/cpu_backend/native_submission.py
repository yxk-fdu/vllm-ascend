"""Register pinned tasks once; native poll/compute/completion never enter Python."""

import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path

import torch
import torch_npu


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
        self._dma_layers = {}
        self._dma_stream = None
        self._dma_capture_directory = None
        self._dma_clock = None
        self.channels = []
        self.prefetch_sites = {}
        self._closed = False
        self._close_lock = threading.Lock()
        atexit.register(self.close)
        _LOG.info("CPU MoE native dispatch: input flag -> C++ planner -> CPUInfer; "
                  "native completion and DMA never return to Python during replay")

    def _register_dma_layer(self, transfer):
        manager = transfer.worker.manager
        plan = getattr(manager, "_prepared_expert_h2d", {}).get(transfer.layer_idx)
        if plan is None:
            raise RuntimeError("Native DMA requires prepared physical expert buffers")
        if self._dma_stream is None:
            self.engine.configure_dma(transfer.native_dma_cpu, int(transfer.worker.stream.npu_stream))
            self._dma_stream = transfer.worker.stream
        elif int(self._dma_stream.npu_stream) != int(transfer.worker.stream.npu_stream):
            raise RuntimeError("A native dispatcher must use one shared weight DMA stream")
        key = (id(manager), transfer.layer_idx)
        cached = self._dma_layers.get(key)
        if cached is None:
            index = self.engine.add_dma_layer(*plan.native_dma_table())
            # Keep Python owners until the C++ queue and DMA stream are drained.
            # Registration is cold; the native consumer stores only raw bytes.
            plan.native_dma_live = True
            self._dma_layers[key] = (index, plan)
            manager._native_dma_worker = self
        else:
            index, previous = cached
            if previous is not plan:
                raise RuntimeError("Close native DMA before replacing expert buffers")
        return index

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
        dma_layer = self._register_dma_layer(context.transfer)
        index = len(self.channels)
        self.channels.append((backend, hidden, ready, hostflag, context, task))
        native = self.engine.add(
            hostflag.data_ptr(), kt_kernel_ext.native_cpu_handle(wrapper.cpu_infer), task,
            ids[slot].data_ptr(), weights[slot].data_ptr(), context.mapping.data_ptr(),
            int(ready.npu_event), int(context.ready.npu_event), flat.shape[0], ids[slot].shape[-1], backend.config.layer_idx,
            ids[slot].element_size(), context.mapping.element_size(), context.transfer.native_planner)
        if native != index:
            raise RuntimeError("Native task registration mismatch")
        self.engine.bind_dma_channel(index, dma_layer, context.transfer.layer.log2phy.data_ptr(),
                                     context.mapping.numel() * context.mapping.element_size())
        context.native_channel = index

    def register_prefetch(self, site, cap):
        dma_layer = self._register_dma_layer(site.transfer)
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
        self.engine.bind_dma_channel(index, dma_layer, site.transfer.layer.log2phy.data_ptr(),
                                     site.mapping.numel() * site.mapping.element_size())

    def set_dma_capture(self, enabled, directory):
        if enabled:
            if not directory:
                raise ValueError("Native DMA capture requires an output directory")
            self._dma_capture_directory = Path(directory)
            self._dma_clock = {"reference_mono_ns": time.monotonic_ns(), "wall_ns": time.time_ns()}
        self.engine.set_dma_capture(enabled)
        if not enabled and self._dma_capture_directory is not None:
            records = self.engine.dma_timings()
            self._dma_capture_directory.mkdir(parents=True, exist_ok=True)
            path = self._dma_capture_directory / f"cpu_native_dma_submit_{os.getpid()}.jsonl"
            with path.open("a") as output:
                for record in records:
                    record["event"] = "cpu_moe.native_dma_submit" if "channel" in record else "cpu_moe.native_dma_flush"
                    record["pid"] = os.getpid()
                    record.update(self._dma_clock)
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._dma_capture_directory = None
            self._dma_clock = None

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            torch.npu.synchronize(self.device)
            self.engine.close()
            for _, plan in self._dma_layers.values():
                plan.native_dma_live = False
            _LOG.info("CPU MoE native dispatcher retired: submitted=%d completed=%d", *self.engine.stats())
            self._closed = True
