# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload import expert_offload_manager
from vllm_ascend.ops.fused_moe import routed_experts as routed_experts_module
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, EplbExpertTensorList
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("positional", [False, True])
def test_offload_capacity_is_applied_before_parent_init(monkeypatch, positional):
    manager = SimpleNamespace(moe_layers=[object()])
    monkeypatch.setattr(expert_offload_manager, "_EXPERT_OFFLOAD_MANAGER", manager)
    offload_config = SimpleNamespace(
        expert_offload=True,
        num_device_experts_for_layer=[32, 84].__getitem__,
    )
    monkeypatch.setattr(
        routed_experts_module, "get_ascend_config", lambda: SimpleNamespace(expert_offload_config=offload_config)
    )

    class ParentInitReached(Exception):
        pass

    def parent_init(self, layer_name, params_dtype, moe_config, quant_config, expert_map_manager):
        assert self._offload_preallocated_capacity == 84
        assert moe_config.num_local_experts == expert_map_manager._local_num_experts == 84
        assert torch.equal(expert_map_manager._expert_map[:84], torch.arange(84, dtype=torch.int32))
        assert torch.all(expert_map_manager._expert_map[84:] == -1)
        raise ParentInitReached

    monkeypatch.setattr(AscendRoutedExperts.__mro__[1], "__init__", parent_init)
    args = {
        "layer_name": "model.layers.3.mlp.experts",
        "params_dtype": torch.bfloat16,
        "moe_config": SimpleNamespace(ep_size=1, num_logical_experts=256, num_local_experts=256),
        "quant_config": object(),
        "expert_map_manager": SimpleNamespace(_local_num_experts=256, _expert_map=None),
    }
    with pytest.raises(ParentInitReached):
        if positional:
            AscendRoutedExperts(*args.values())
        else:
            AscendRoutedExperts(**args)


@pytest.mark.parametrize(
    ("enabled", "prefix", "quant_type", "reject"),
    [
        (True, "model.layers.3", QuantType.W4A4MXFP, False),
        (True, "model.layers.3", QuantType.W4A8MXFP, False),
        (True, "model.layers.3", QuantType.W4A16MXFP, False),
        (True, "model.layers.3", QuantType.W8A8, True),
        (True, "model.layers.3", QuantType.W8A8MXFP, True),
        (True, "model.layers.3", QuantType.NONE, True),
        (False, "model.layers.3", QuantType.W8A8, False),
        (True, "model.mtp.0", QuantType.W8A8, False),
    ],
)
def test_cpu_moe_validates_resolved_expert_quantization(monkeypatch, enabled, prefix, quant_type, reject):
    offload_config = SimpleNamespace(cpu_moe=SimpleNamespace(enabled=enabled))
    monkeypatch.setattr(
        routed_experts_module, "get_ascend_config", lambda: SimpleNamespace(expert_offload_config=offload_config)
    )
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    object.__setattr__(routed_experts, "tid2eid", None)
    if quant_type is QuantType.NONE:
        method = object()
        quant_config = None
        monkeypatch.setattr(AscendRoutedExperts.__mro__[1], "_get_quant_method", lambda *_: method)
    else:
        method = SimpleNamespace(quant_method=SimpleNamespace(quant_type=quant_type))
        quant_config = SimpleNamespace(get_quant_method=lambda *args, **kwargs: method)

    if reject:
        with pytest.raises(ValueError, match=rf"CPU MoE requires MXFP4.*{prefix!r}.*{quant_type.name}"):
            routed_experts._get_quant_method(prefix, quant_config, None)
    else:
        assert routed_experts._get_quant_method(prefix, quant_config, None) is method


def _routed_experts(weight_views):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace(
        get_eplb_weight_views=lambda layer: weight_views,
    )
    return routed_experts


def test_get_expert_weights_flattens_layout_aware_views():
    weights = [torch.randn(2, 3, 4), torch.randn(2, 5)]

    views = list(_routed_experts(weights).get_expert_weights())

    assert [view.shape for view in views] == [torch.Size([2, 12]), torch.Size([2, 5])]
    assert views[0].untyped_storage().data_ptr() == weights[0].untyped_storage().data_ptr()


def test_get_expert_weights_preserves_independent_expert_tensors():
    expert_tensors = [torch.randn(3, 4), torch.randn(3, 4)]

    views = list(_routed_experts([expert_tensors]).get_expert_weights())

    assert len(views) == 1
    assert isinstance(views[0], EplbExpertTensorList)
    assert views[0].shape == torch.Size([2, 3, 4])
    assert all(actual is expected for actual, expected in zip(views[0], expert_tensors))

    buffer = torch.empty_like(views[0])
    assert isinstance(buffer, EplbExpertTensorList)
    assert buffer.shape == views[0].shape
    assert all(tensor.storage_offset() == 0 for tensor in buffer)


def test_get_expert_weights_rejects_unsupported_quantization():
    with pytest.raises(NotImplementedError, match="weight views are not defined"):
        list(_routed_experts([]).get_expert_weights())


def test_get_expert_weights_rejects_missing_weight_view_contract():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="must implement get_eplb_weight_views"):
        list(routed_experts.get_expert_weights())


def test_get_expert_weights_rejects_non_expert_first_dimension():
    with pytest.raises(ValueError, match="first dimension"):
        list(_routed_experts([torch.randn(3, 4)]).get_expert_weights())


def test_get_expert_weights_rejects_wrong_expert_tensor_list_length():
    with pytest.raises(ValueError, match="must contain local_num_experts"):
        list(_routed_experts([[torch.randn(3, 4)]]).get_expert_weights())


def test_get_expert_weights_rejects_non_contiguous_view():
    with pytest.raises(ValueError, match="flattenable without a copy"):
        list(_routed_experts([torch.randn(2, 3, 4).transpose(1, 2)]).get_expert_weights())


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_ascend_expert_map_follows_model_runner(use_v2_model_runner):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    upstream_map = torch.tensor([0, 1], dtype=torch.int32)
    object.__setattr__(routed_experts, "_use_v2_model_runner", use_v2_model_runner)
    # Both v0.28.0 and main read quant_method in RoutedExperts.expert_map.
    routed_experts.quant_method = SimpleNamespace(moe_kernel=None)
    routed_experts.ascend_expert_map = legacy_map
    object.__setattr__(routed_experts, "_expert_map", upstream_map)
    object.__setattr__(routed_experts, "rocm_aiter_fmoe_enabled", False)

    expected = upstream_map if use_v2_model_runner else legacy_map
    assert routed_experts.ascend_expert_map is expected


def test_update_expert_map_preserves_upstream_and_legacy_contracts(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    parent_update_calls = []

    def parent_update(instance):
        parent_update_calls.append(instance)

    monkeypatch.setattr(type(routed_experts).__mro__[1], "update_expert_map", parent_update)
    expert_map_manager = SimpleNamespace(_expert_map=None)
    object.__setattr__(routed_experts, "expert_map_manager", expert_map_manager)

    routed_experts.update_expert_map()

    assert parent_update_calls == [routed_experts]

    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    routed_experts.update_expert_map(legacy_map)

    assert routed_experts.ascend_expert_map is legacy_map
    assert expert_map_manager._expert_map is legacy_map


@pytest.fixture
def eplb_capacity_layer(monkeypatch):
    eplb_config = SimpleNamespace(
        dynamic_eplb=False, expert_map_path=None, num_redundant_experts=0,
        eplb_policy_type=0, expert_heat_collection_interval=1,
    )
    monkeypatch.setattr(
        routed_experts_module, "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=eplb_config),
    )
    monkeypatch.setattr(
        routed_experts_module, "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1)),
    )
    monkeypatch.setattr(torch.Tensor, "npu", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.setattr(AscendRoutedExperts, "_promote_attr_to_buffer", lambda *args: None)
    monkeypatch.setattr(routed_experts_module.VllmEplbAdaptor, "register_layer", lambda *args: None)
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    layer.moe_config = SimpleNamespace(
        num_experts=256, num_logical_experts=256, num_local_experts=256, ep_size=1,
    )
    layer.mix_placement = False
    return layer, eplb_config


def test_eplb_accepts_preallocated_offload_slots(eplb_capacity_layer):
    layer, _ = eplb_capacity_layer
    layer._offload_preallocated_capacity = 84
    layer.moe_config.num_local_experts = 84
    layer.init_eplb(0)
    assert layer.moe_config.num_logical_experts == layer.moe_config.num_experts == 256
    assert layer.moe_config.num_local_experts == 84
    assert layer.moe_load.shape == (84,)
    assert layer.log2phy is None  # _init_expert_offload builds the resident map next.
    assert not layer.dynamic_eplb


@pytest.mark.parametrize(("allocated", "capacity"), [(84, None), (83, 84)])
def test_eplb_still_rejects_unexpected_local_capacity(eplb_capacity_layer, allocated, capacity):
    layer, _ = eplb_capacity_layer
    layer._offload_preallocated_capacity = capacity
    layer.moe_config.num_local_experts = allocated
    with pytest.raises(ValueError, match="local expert capacity mismatch"):
        layer.init_eplb(0)


def test_eplb_preserves_full_resident_layer_validation(eplb_capacity_layer):
    layer, _ = eplb_capacity_layer
    layer.init_eplb(0)
    assert layer.moe_load.shape == (256,)


@pytest.mark.parametrize("unsupported", ["dynamic_eplb", "mix_placement", "redundancy", "expert_map"])
def test_preallocated_offload_rejects_incompatible_eplb(eplb_capacity_layer, unsupported):
    layer, config = eplb_capacity_layer
    layer._offload_preallocated_capacity = 84
    layer.moe_config.num_local_experts = 84
    if unsupported == "dynamic_eplb":
        config.dynamic_eplb = True
    elif unsupported == "mix_placement":
        layer.mix_placement = True
    elif unsupported == "redundancy":
        layer.moe_config.num_experts += 1
    else:
        config.expert_map_path = "placement.json"
    with pytest.raises(ValueError, match="Preallocated expert offload requires"):
        layer.init_eplb(0)
