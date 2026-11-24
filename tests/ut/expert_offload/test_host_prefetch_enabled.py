"""Host hybrid honors prefetch and joins it before preparing demand routes."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("enabled", [False, True])
def test_host_initialization_preserves_requested_prefetch(enabled):
    source = Path(__file__).parents[3] / "vllm_ascend/expert_offload/expert_offload_manager.py"
    tree = ast.parse(source.read_text())
    manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExpertOffloadManager")
    init = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    # Stop before unrelated model/distributed initialization, using the actual
    # constructor statements rather than a copy of the setting-handling code.
    body = []
    for node in init.body:
        if isinstance(node, ast.ImportFrom):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("get_ascend_config().benchmark_route_stats_enabled"):
            break
        body.append(node)

    class Config:
        def __init__(self):
            self.config = {"expert_prefetch_enabled": enabled}
            self.cpu_moe = SimpleNamespace(enabled=True)
            self.native_expert_dma = False

        @property
        def expert_prefetch_enabled(self):
            return self.config["expert_prefetch_enabled"]

    cfg = Config()
    instance = SimpleNamespace()
    namespace = {"self": instance, "get_ascend_config": lambda: SimpleNamespace(expert_offload_config=cfg), "logger": Mock()}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    assert instance._host_hybrid_enabled
    assert cfg.expert_prefetch_enabled is enabled


@pytest.mark.parametrize("pending", [False, True])
def test_prefetch_join_precedes_demand_preparation(monkeypatch, pending):
    import torch_npu
    import vllm.forward_context as forward
    from vllm_ascend.expert_offload import hybrid_executor as hybrid

    operations = []
    event = object()
    other_event = object()
    layer = SimpleNamespace(offload_layer_idx=1)
    executor = SimpleNamespace(layer_idx=3, backend=SimpleNamespace(max_num_tokens=48))
    events = {3: other_event}
    if pending:
        events[1] = event
    manager = SimpleNamespace(
        _finish_pending_predict=lambda i: operations.append(("predict_join", i)),
        _prefetch_state_lock=threading.Lock(),
        _prefetch_layer_npu_event=events,
        offload_config=SimpleNamespace(expert_substitution_enabled=False),
    )
    stream = SimpleNamespace(wait_event=lambda e: operations.append(("prefetch_join", e)))
    monkeypatch.setattr(hybrid, "get_hybrid_expert_executor", lambda _: executor)
    monkeypatch.setattr(forward, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(torch_npu.npu, "current_stream", lambda: stream)
    monkeypatch.setattr(hybrid, "PreparedHybridRoutes", lambda *a, **k: operations.append("demand_prepare"))
    hybrid.begin_hybrid_routes(layer, hidden_states=torch.zeros(1, 4),
                               topk_ids=torch.tensor([[0]]), topk_weights=torch.ones(1, 1),
                               log2phy=torch.tensor([0]), transfer_manager=manager)
    expected = [("predict_join", 1)]
    if pending:
        expected.append(("prefetch_join", event))
    assert operations == [*expected, "demand_prepare"]
    assert manager._prefetch_layer_npu_event == {3: other_event}
