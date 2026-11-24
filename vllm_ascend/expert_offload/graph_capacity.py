# SPDX-License-Identifier: Apache-2.0

"""Startup validation for the fixed-size CPU MoE graph buffers."""

from collections.abc import Iterable
from copy import copy

from vllm.config import CUDAGraphMode, VllmConfig


def needs_eager_expert_loading(manager, moe_rows: int) -> bool:
    """Oversized single-card NPU-only batches require the eager weight pool."""
    return (
        manager is not None
        and not manager._host_hybrid_enabled
        and not manager.enable_multi_card
        and moe_rows > manager.offload_threshold
    )


def validate_graph_token_capacity(
    max_num_tokens: int, sizes: Iterable[int], *, stage: str, query_len: int, max_num_seqs: int
) -> None:
    oversized = sorted({size for size in sizes if size > max_num_tokens})
    if oversized:
        raise ValueError(
            f"CPU MoE {stage} graph token sizes {oversized} exceed "
            f"cpu_moe.max_num_tokens={max_num_tokens} "
            f"(max_num_seqs={max_num_seqs}, decode tokens per request={query_len}). "
            f"Set cpu_moe.max_num_tokens >= {oversized[-1]} or reduce cudagraph_capture_sizes."
        )


def validate_requested_decode_capacity(max_num_tokens: int, vllm_config: VllmConfig, query_len: int) -> None:
    """Check the single-card MRV1 full-decode request before weight loading."""
    compilation = vllm_config.compilation_config
    if vllm_config.model_config.enforce_eager or compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return
    if not compilation.cudagraph_capture_sizes:
        return
    if query_len > 1:
        # MRV1 normalizes speculative-decode sizes before building graph keys.
        # Preview that calculation without mutating the live compilation config.
        compilation = copy(compilation)
        compilation.adjust_cudagraph_sizes_for_spec_decode(query_len, tensor_parallel_size=1)
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    sizes = (size for size in compilation.cudagraph_capture_sizes if query_len <= size <= query_len * max_num_seqs)
    validate_graph_token_capacity(
        max_num_tokens, sizes, stage="requested decode", query_len=query_len, max_num_seqs=max_num_seqs
    )
