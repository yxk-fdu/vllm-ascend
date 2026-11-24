"""Report-thread callback must fail visibly without leaking the GIL."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_manager_method(name, namespace):
    source = Path(__file__).parents[3] / 'vllm_ascend/expert_offload/expert_offload_manager.py'
    tree = ast.parse(source.read_text())
    manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ExpertOffloadManager')
    method = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def callback(exits):
    return load_manager_method('_run_prefetch_callback', {'torch': torch, 'os': SimpleNamespace(_exit=exits.append)})


@pytest.mark.parametrize("capturing", [False, True])
def test_standard_prefetch_keeps_callback_and_weight_ready_order(capturing):
    steps = []
    stream = SimpleNamespace(
        wait_event=lambda _: steps.append("wait"),
        record_event=lambda _: steps.append("ready"),
    )

    def launch(_, fn, args):
        steps.append("launch")
        fn(args)

    namespace = {
        "_EXTRA_CTX": SimpleNamespace(capturing=capturing),
        "torch_npu": SimpleNamespace(npu=SimpleNamespace(
            stream=lambda _: nullcontext(), current_stream=lambda: stream, Event=object,
            _subscribe_report=lambda _: None, _launch_host_func=launch,
        )),
        "get_subscribed_compute_streams": lambda: [stream],
    }
    manager = SimpleNamespace(
        _host_hybrid_enabled=False, _native_npu_dma_enabled=False, _prefetch_stream=stream,
        _prefetch_state_lock=nullcontext(), _prefetch_layer_npu_event={}, enable_multi_card=False,
        _build_prefetch_call=lambda *args: (lambda _: steps.append("load"), None),
        _run_prefetch_callback=lambda fn, args: fn(args),
    )
    layer = SimpleNamespace(log2phy=SimpleNamespace(copy_=lambda *args, **kwargs: steps.append("mapping")))
    dispatch = load_manager_method("_dispatch_prefetch", namespace)
    dispatch(manager, (None, None, None, None, layer, 3, None), object())
    assert steps == ["wait"] + (["launch"] if capturing else []) + ["load", "mapping", "ready"]
    assert 3 in manager._prefetch_layer_npu_event


def test_report_thread_mutates_inference_tensor():
    with torch.inference_mode():
        mapping = torch.tensor([0, -1, 1])
    exits, failures = [], []
    manager = SimpleNamespace(_note_cb_failure=failures.append)
    def planner(args):
        assert torch.is_inference_mode_enabled()
        args.copy_(torch.tensor([-1, 0, 1]))
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(callback(exits), manager, planner, mapping).result(timeout=5)
    assert mapping.tolist() == [-1, 0, 1]
    assert exits == failures == []


def test_native_prefetch_dispatches_the_staged_site():
    steps = []
    manager = SimpleNamespace(_host_hybrid_enabled=False, _native_npu_dma_enabled=True)
    site = SimpleNamespace(dispatch_prefetch=steps.append)
    dispatch = load_manager_method("_dispatch_prefetch", {})
    dispatch(manager, site, None)
    assert steps == [manager]


@pytest.mark.parametrize('error', [RuntimeError('partial copy'), SystemExit('callback failed')])
def test_planner_failure_terminates_instead_of_reusing_partial_weights(error):
    exits, failures = [], []
    manager = SimpleNamespace(_note_cb_failure=failures.append)
    def planner(args):
        raise error
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(callback(exits), manager, planner, None).result(timeout=5)
    assert failures == ['prefetch']
    assert exits == [1]


def test_logger_failure_cannot_escape_callback_bridge():
    exits = []
    def fail(*args):
        raise RuntimeError('failed')
    manager = SimpleNamespace(_note_cb_failure=fail)
    callback(exits)(manager, fail, None)
    assert exits == [1]
