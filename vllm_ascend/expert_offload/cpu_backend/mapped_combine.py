"""Combine CPU BF16 results directly from mapped host memory on NPU."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import math
import mmap
from pathlib import Path
import threading

import torch


_ARM_RESULT_ALIGNMENT = 64 * 1024


class _MappedCombineRuntime:
    def __init__(self, device_index: int):
        library = Path(__file__).parent / "lib/libcpu_moe_mapped_combine_dispatch.so"
        if not library.is_file():
            raise RuntimeError(f"Mapped CPU result combine is not built: {library}")
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.CpuMoeMappedCombine
        self.launch.argtypes = ([ctypes.c_void_p] * 3
                                + [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint8])
        self.launch.restype = ctypes.c_int
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtHostRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.acl.aclrtHostRegister.restype = ctypes.c_int
        self.device_index = device_index
        self.lock = threading.Lock()
        # Captured graphs keep raw addresses. Retain the exact storage owner
        # and its registration for the process lifetime, shared across layers.
        self.registrations = {}
        self.graph_results = {}

    def _register_host_storage(self, storage):
        key = (storage.data_ptr(), storage.nbytes())
        mapped = ctypes.c_void_p()
        ret = self.acl.aclrtHostRegister(key[0], key[1], 0,
                                        ctypes.byref(mapped))
        if ret or not mapped.value:
            raise RuntimeError(f"CPU result host registration failed: ACL {ret}")
        entry = (mapped.value, storage)
        self.registrations[key] = entry
        return entry

    def arm_graph_result(self, shape, slot):
        # CANN can map ordinary aligned host pages for AIV reads on ARM.
        # Its managed pinned allocator can return a host-only address here.
        key = (tuple(shape), slot)
        with self.lock:
            result = self.graph_results.get(key)
            if result is None:
                elements = math.prod(shape)
                allocation = mmap.mmap(-1, elements * 2 + _ARM_RESULT_ALIGNMENT - 1,
                                       flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
                address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
                offset = -address % _ARM_RESULT_ALIGNMENT
                result = torch.frombuffer(allocation, dtype=torch.bfloat16,
                                          count=elements, offset=offset).view(shape)
                self._register_host_storage(result.untyped_storage())
                self.graph_results[key] = result
            return result

    def map_host(self, host: torch.Tensor) -> int:
        storage = host.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        with self.lock:
            entry = self.registrations.get(key)
            if entry is None:
                if not host.is_pinned():
                    raise ValueError("Mapped CPU result must use pinned or registered storage")
                entry = self._register_host_storage(storage)
            return entry[0] + host.data_ptr() - key[0]


@lru_cache(maxsize=None)
def _runtime(device_index: int) -> _MappedCombineRuntime:
    return _MappedCombineRuntime(device_index)


def arm_graph_result(device_index: int, shape, slot: int) -> torch.Tensor:
    return _runtime(device_index).arm_graph_result(shape, slot)


@torch.library.custom_op("vllm_ascend::cpu_moe_mapped_combine", mutates_args=())
def mapped_combine(npu: torch.Tensor, host: torch.Tensor) -> torch.Tensor:
    if npu.device.type != "npu" or host.device.type != "cpu":
        raise ValueError("Mapped combine requires NPU output and CPU result")
    if host.dtype != torch.bfloat16 or not host.is_contiguous():
        raise ValueError("Mapped CPU result must be contiguous BF16")
    if npu.shape != host.shape or npu.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Mapped combine requires equal shapes and BF16/FP32 NPU output")
    npu_input = npu.contiguous()
    result = torch.empty_like(npu_input, memory_format=torch.contiguous_format)
    if not npu.numel():
        return result
    device_index = npu.device.index
    runtime = _runtime(device_index)
    host_pointer = runtime.map_host(host)
    status = runtime.launch(host_pointer, npu_input.data_ptr(), result.data_ptr(),
                            npu.numel(), 0 if npu.dtype == torch.bfloat16 else 1,
                            device_index)
    if status:
        raise RuntimeError(f"Mapped CPU result combine launch failed: {status}")
    return result


@mapped_combine.register_fake
def _mapped_combine_fake(npu: torch.Tensor, host: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(npu, memory_format=torch.contiguous_format)
