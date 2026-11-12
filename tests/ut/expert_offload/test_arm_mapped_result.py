import ctypes
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.cpu_backend.mapped_combine import _MappedCombineRuntime


def test_aligned_graph_result_reuses_registered_storage():
    registrations = []

    def register(address, size, flags, out):
        registrations.append((address, size, flags))
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = address + 0x10000
        return 0

    runtime = object.__new__(_MappedCombineRuntime)
    runtime.lock = threading.Lock()
    runtime.acl = SimpleNamespace(aclrtHostRegister=register)
    runtime.graph_results = {}
    runtime.registrations = {}

    result = runtime.arm_graph_result((8, 4096), 0)
    assert result.dtype == torch.bfloat16
    assert result.is_contiguous()
    assert result.data_ptr() % 65536 == 0
    assert runtime.arm_graph_result((8, 4096), 0) is result
    assert runtime.map_host(result[1:]) == result[1:].data_ptr() + 0x10000
    assert registrations == [(result.data_ptr(), 65536, 0)]
    assert runtime.arm_graph_result((8, 4096), 1).data_ptr() != result.data_ptr()

    with pytest.raises(ValueError, match="pinned or registered"):
        runtime.map_host(torch.empty_like(result))
