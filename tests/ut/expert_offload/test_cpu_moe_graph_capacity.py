from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.expert_offload.graph_capacity import (
    needs_eager_expert_loading,
    validate_graph_token_capacity,
    validate_requested_decode_capacity,
)


def make_config(sizes, *, max_seqs=2, eager=False, mode=CUDAGraphMode.FULL_DECODE_ONLY):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=SimpleNamespace(cudagraph_mode=mode, cudagraph_capture_sizes=sizes),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
    )


def test_full_decode_ignores_sizes_above_scheduler_request_limit():
    config = make_config([1, 2, 4, 8], max_seqs=2)
    validate_requested_decode_capacity(2, config, 1)
    with pytest.raises(ValueError, match=r"requested decode graph token sizes \[2\].*max_num_tokens=1"):
        validate_requested_decode_capacity(1, config, 1)


def test_spec_decode_checks_normalized_token_sizes_once():
    config = make_config([1, 9, 16, 24])

    class Compilation:
        cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        cudagraph_capture_sizes = config.compilation_config.cudagraph_capture_sizes

        def adjust_cudagraph_sizes_for_spec_decode(self, query_len, tensor_parallel_size):
            assert (query_len, tensor_parallel_size) == (8, 1)
            self.cudagraph_capture_sizes = [8, 16, 24]

    config.compilation_config = Compilation()
    with pytest.raises(ValueError, match=r"graph token sizes \[16\].*decode tokens per request=8"):
        validate_requested_decode_capacity(8, config, 8)
    assert config.compilation_config.cudagraph_capture_sizes == [1, 9, 16, 24]
    validate_requested_decode_capacity(16, config, 8)


@pytest.mark.parametrize("eager, mode", [(True, CUDAGraphMode.FULL_DECODE_ONLY), (False, CUDAGraphMode.PIECEWISE)])
def test_early_decode_check_does_not_reject_eager_or_unresolved_mixed_sizes(eager, mode):
    validate_requested_decode_capacity(1, make_config([8, 16], max_seqs=16, eager=eager, mode=mode), 1)


def test_resolved_sizes_are_checked_without_reapplying_decode_filter():
    with pytest.raises(ValueError, match=r"resolved graph token sizes \[16, 32\].*max_num_tokens=8"):
        validate_graph_token_capacity(8, [32, 1, 16, 32], stage="resolved", query_len=8, max_num_seqs=2)


def test_npu_only_decode_falls_back_on_resident_slot_overflow():
    manager = SimpleNamespace(_host_hybrid_enabled=False, enable_multi_card=False, offload_threshold=14)
    assert not needs_eager_expert_loading(None, 32)
    assert not needs_eager_expert_loading(manager, 8)
    assert not needs_eager_expert_loading(manager, 14)
    assert needs_eager_expert_loading(manager, 32)
    manager._host_hybrid_enabled = True
    assert not needs_eager_expert_loading(manager, 32)
