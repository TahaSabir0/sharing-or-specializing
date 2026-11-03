"""CPU contract tests. These do not replace the Blackwell smoke gate."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from methods.base import (count_trainable, load_method_state, load_targets, routing,
                          save_method_state, validate_routing_stats)
from methods.hydra import HydraLinear, apply_method
from methods import backbones


@pytest.fixture(autouse=True)
def isolate_routing():
    torch.manual_seed(1337)
    routing.clear()
    yield
    routing.clear()


def make_layer(mode, **kwargs):
    return HydraLinear(nn.Linear(5, 7), "test", {"mode": mode, "r": 3, "dropout": 0,
                                                **kwargs})


def randomize_b(layer):
    bs = layer.lora_B if layer.mode == "hydraT" else [layer.lora_B]
    for b in bs:
        nn.init.normal_(b.weight, std=0.2)


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
@pytest.mark.parametrize("shape", [(6, 5), (6, 4, 5)])
def test_initial_output_equals_frozen_base(mode, shape):
    layer = make_layer(mode)
    x = torch.randn(shape)
    with routing.use(task_id=torch.arange(6)):
        torch.testing.assert_close(layer(x), layer.base(x), rtol=0, atol=0)
    assert all(not p.requires_grad for p in layer.base.parameters())


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
@pytest.mark.parametrize("tasks", [[4], [4, 4, 4], [5, 0, 5, 2]])
def test_mixed_batch_matches_individual_lora_formulas_and_gradients(mode, tasks):
    layer = make_layer(mode)
    randomize_b(layer)
    ref = copy.deepcopy(layer)
    x = torch.randn(len(tasks), 3, 5, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    ids = torch.tensor(tasks)
    with routing.use(task_id=ids):
        got = layer(x)
    expected = []
    for i, task in enumerate(ids.tolist()):
        a = ref.lora_A if mode == "hydraT" else ref.lora_A[task]
        b = ref.lora_B[task] if mode == "hydraT" else ref.lora_B
        expected.append(ref.base(xr[i]) + ref.scaling * b(a(xr[i])))
    want = torch.stack(expected)
    torch.testing.assert_close(got, want)
    got.square().sum().backward()
    want.square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad)
    for (name, p), (_, q) in zip(layer.named_parameters(), ref.named_parameters()):
        if q.grad is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
def test_only_selected_heads_and_shared_factor_update(mode):
    layer = make_layer(mode)
    randomize_b(layer)  # after the first step, both factors must receive gradients
    heads = layer.lora_B if mode == "hydraT" else layer.lora_A
    shared = layer.lora_A if mode == "hydraT" else layer.lora_B
    before = [head.weight.detach().clone() for head in heads]
    shared_before = shared.weight.detach().clone()
    base_before = copy.deepcopy(layer.base.state_dict())
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],
                            lr=0.01, weight_decay=0)
    with routing.use(task_id=torch.tensor([1, 4])):
        layer(torch.randn(2, 3, 5)).square().mean().backward()
    opt.step()
    assert not torch.equal(shared.weight, shared_before)
    for task, head in enumerate(heads):
        if task in (1, 4):
            assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
            assert not torch.equal(head.weight, before[task])
        else:
            assert head.weight.grad is None
            torch.testing.assert_close(head.weight, before[task], rtol=0, atol=0)
    for key, val in layer.base.state_dict().items():
        torch.testing.assert_close(val, base_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
@pytest.mark.parametrize("reentrant", [True, False])
@pytest.mark.parametrize("tasks", [[3], [2, 4]])
def test_checkpoint_backward_keeps_route(mode, reentrant, tasks):
    layer = make_layer(mode, dropout=0.25)
    randomize_b(layer)
    ref = copy.deepcopy(layer)
    x = torch.randn(len(tasks), 3, 5, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    routing.set(task_id=torch.tensor(tasks))
    rng = torch.get_rng_state()
    loss = checkpoint(layer, x, use_reentrant=reentrant).square().mean()
    loss.backward()  # routing must still exist during the replay
    torch.set_rng_state(rng)
    ref(xr).square().mean().backward()
    torch.testing.assert_close(x.grad, xr.grad)
    for p, q in zip(layer.parameters(), ref.parameters()):
        if q.grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
def test_batch_one_forward_and_checkpoint_replay_avoid_row_ops_and_scalar_sync(mode):
    from torch.profiler import ProfilerActivity, profile
    layer = make_layer(mode)
    randomize_b(layer)
    x = torch.randn(1, 3, 5, requires_grad=True)
    routing.set(task_id=torch.tensor([3]))
    layer(x)  # resolve host labels once for this batch
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        checkpoint(layer, x, use_reentrant=True).square().mean().backward()
    names = {event.key for event in prof.key_averages()}
    assert not names.intersection({"aten::nonzero", "aten::_unique2", "aten::index_select",
                                   "aten::index_copy", "aten::_local_scalar_dense"})


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
def test_cached_routes_refresh_on_new_batch_mutation_and_restore_nested_context(mode):
    layer = make_layer(mode)
    randomize_b(layer)
    x = torch.randn(1, 3, 5)
    ids = torch.tensor([1])
    with routing.use(task_id=ids):
        first = layer(x)
        with routing.use(task_id=torch.tensor([5])):
            other = layer(x)
        torch.testing.assert_close(layer(x), first, rtol=0, atol=0)
        ids.fill_(5)
        torch.testing.assert_close(layer(x), other, rtol=0, atol=0)
        ids.fill_(6)
        with pytest.raises(ValueError, match="task_id"):
            layer(x)
    with routing.use(task_id=torch.tensor([1])):
        torch.testing.assert_close(layer(x), first, rtol=0, atol=0)
    routing.clear()
    with pytest.raises(ValueError, match="task_id"):
        layer(x)


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
def test_routes_created_inside_inference_mode_support_prefill_and_decode(mode):
    layer = make_layer(mode).eval()
    randomize_b(layer)
    x = torch.randn(1, 3, 5)
    with routing.use(task_id=torch.tensor([2])):
        expected = layer(x)
    with torch.inference_mode(), routing.use(task_id=torch.tensor([2])):
        torch.testing.assert_close(layer(x), expected)
        torch.testing.assert_close(layer(x[:, -1:]), expected[:, -1:])


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
def test_bfloat16_base_fp32_adapters(mode):
    base = nn.Linear(5, 7, dtype=torch.bfloat16)
    layer = HydraLinear(base, "test", {"mode": mode, "dropout": 0})
    randomize_b(layer)
    with routing.use(task_id=torch.tensor([0, 5])):
        out = layer(torch.randn(2, 3, 5, dtype=torch.bfloat16))
        out.float().square().mean().backward()
    assert out.dtype == torch.bfloat16
    assert layer.base is base and base.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in layer.parameters() if p.requires_grad)


@pytest.mark.parametrize("ids", [None, torch.tensor([-1, 0]), torch.tensor([0, 6]),
                                 torch.tensor([0]), torch.tensor([[0, 1]]),
                                 torch.tensor([0., 1.])])
def test_invalid_routes_fail_instead_of_silently_selecting_a_head(ids):
    layer = make_layer("hydraT")
    with routing.use(task_id=ids), pytest.raises(ValueError, match="task_id"):
        layer(torch.randn(2, 3, 5))


@pytest.mark.parametrize("cfg", [{"mode": "hydraL"}, {"r": 0}, {"r": True},
                                  {"n_tasks": 5}, {"alpha": float('nan')},
                                  {"dropout": 1}, {"log_routing": "false"},
                                  {"typo": 1}])
def test_invalid_configs_fail(cfg):
    with pytest.raises(ValueError):
        HydraLinear(nn.Linear(5, 7), "test", cfg)


def toy_model(scope="language_model"):
    model = nn.Module()
    model.model = nn.Module()
    lm = nn.Module()
    setattr(model.model, scope, lm)
    lm.layers = nn.ModuleList([nn.Module()])
    lm.layers[0].q_proj = nn.Linear(5, 7)
    model.vision = nn.Linear(5, 7)
    spec = {"scope_prefix": f"model.{scope}.layers", "target_modules": ["q_proj"], "n_modules": 1}
    return model, spec, f"model.{scope}.layers.0.q_proj"


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
@pytest.mark.parametrize("scope", ["language_model", "gemma_language_model"])
def test_target_spec_roundtrip_and_prefill_decode_routing(tmp_path, mode, scope):
    model, spec, name = toy_model(scope)
    cfg = {"mode": mode, "r": 3, "dropout": 0, "target_spec": spec}
    original_base = model.get_submodule(name)
    apply_method(model, cfg)
    layer = model.get_submodule(name)
    assert layer.base is original_base
    assert not any(p.requires_grad for p in model.vision.parameters())
    randomize_b(layer)
    expected_stored = 3 * (5 + 6 * 7 if mode == "hydraT" else 6 * 5 + 7)
    assert count_trainable(model) == expected_stored
    assert model.journal_parameter_counts == {
        "stored_params": expected_stored, "active_params_per_example": 3 * (5 + 7)}
    base_state = copy.deepcopy(original_base.state_dict())
    save_method_state(model, str(tmp_path), "hydra", cfg)
    from safetensors.torch import load_file
    saved = load_file(str(tmp_path / "method_state.safetensors"))
    assert saved and all("lora_" in key for key in saved)
    assert json.loads((tmp_path / "method_config.json").read_text())["n_params"] == expected_stored
    loaded, _, _ = toy_model(scope)
    loaded.get_submodule(name).load_state_dict(base_state)
    apply_method(loaded, cfg)
    load_method_state(loaded, str(tmp_path))
    model.eval()
    loaded.eval()
    ids = torch.tensor([5, 1])
    with routing.use(task_id=ids, modality_id=torch.tensor([0, 1]), dataset_id=torch.tensor([2, 3])):
        for length in (4, 1, 1):  # prefill followed by incremental decoding
            x = torch.randn(2, length, 5)
            torch.testing.assert_close(layer(x), loaded.get_submodule(name)(x), rtol=0, atol=0)
        model.journal_routing_stats.end_batch(["sample-a", "sample-b"])
    stats_path = tmp_path / "routing_stats.npz"
    model.journal_routing_stats.save(stats_path)
    assert validate_routing_stats(stats_path) == {"sequences": 2, "layers": 1, "experts": 6}
    with np.load(stats_path) as z:
        np.testing.assert_array_equal(z["weights"][:, 0], np.eye(6)[ids.numpy()])
        np.testing.assert_array_equal(z["task_id"], ids.numpy())
        np.testing.assert_array_equal(z["sample_id"], ["sample-a", "sample-b"])


@pytest.mark.parametrize("backbone, mode, expected", [
    ("q35", "hydraT", 78716928), ("q35", "hydraR", 72757248),
    ("mg15", "hydraT", 55287808), ("mg15", "hydraR", 49020928)])
def test_full_frozen_target_set_parameter_counts_without_allocating_weights(backbone, mode, expected):
    from types import SimpleNamespace
    spec = backbones.load_spec(backbone)
    model = nn.Module()
    model.config = SimpleNamespace(model_type=backbones.get_backbone(backbone).model_type)
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    # The frozen spec supplies the full 248 target shapes; meta avoids 9B-model allocation.
    layers = nn.ModuleList()
    for leaf, (count, shape) in spec["shapes_in_x_out"].items():
        n_in, n_out = map(int, shape.split("x"))
        for _ in range(count):
            layer = nn.Module()
            setattr(layer, leaf, nn.Linear(n_in, n_out, bias=False, device="meta"))
            layers.append(layer)
    model.model.language_model.layers = layers
    apply_method(model, {"mode": mode})
    assert len(layers) == spec["n_modules"]
    assert count_trainable(model) == expected
    assert model.journal_parameter_counts["active_params_per_example"] == spec["P8"]


def test_wrong_target_set_fails_before_freezing_model():
    model, _, _ = toy_model()
    with pytest.raises(RuntimeError, match="expected 248"):
        apply_method(model, {})
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("backbone, folder, study, res", [
    ("q35", "v2", "E1", 768), ("mg15", "e2", "E2", 896)])
def test_configs_pin_all_modes_seeds_and_common_recipe(backbone, folder, study, res):
    import yaml
    for mode in ("hydraT", "hydraR"):
        for seed in (1337, 2026, 42):
            cfg = yaml.safe_load((CODE / "configs" / folder / f"{mode}_s{seed}.yaml").read_text())
            assert cfg["run_id"] == f"j26_{backbone}_{mode}_full_{res}_s{seed}"
            assert cfg["owner"] == "veronika" and cfg["study"] == study
            assert cfg["backbone"] == backbone and cfg["method"] == "hydra"
            assert cfg["seed"] == seed and cfg["deck"] == "j26_train_mix"
            assert cfg["data_tag"] == "full" and cfg["overrides"] == {}
            assert cfg["method_cfg"] == {"mode": mode, "r": 8, "alpha": 16, "dropout": .05,
                                          "n_tasks": 6, "log_routing": True, "target_spec": None}


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
@pytest.mark.parametrize("p8", [21639168, 14901248])
def test_runtime_records_both_parameter_budgets(tmp_path, monkeypatch, mode, p8):
    from types import SimpleNamespace
    sys.path.insert(0, str(CODE / "train"))
    import journal_train
    model, spec, _ = toy_model()
    apply_method(model, {"mode": mode, "target_spec": spec})
    monkeypatch.setattr(journal_train, "gpu_snapshot", lambda: {"visible": "1"})
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2048)
    callback = journal_train.runtime_callback(str(tmp_path), "hydra", p8)
    args = SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=8)
    state = SimpleNamespace(global_step=50)
    callback.on_train_begin(args, state, None, model=model)
    callback.on_train_end(args, state, None)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["P8"] == p8
    assert runtime["stored_params"] == count_trainable(model)
    assert runtime["active_params_per_example"] == 8 * (5 + 7)


@pytest.mark.parametrize("mode, factor", [("hydraT", "B"), ("hydraR", "A")])
def test_head_similarity_marks_undefined_heads_and_preserves_task_order(tmp_path, mode, factor):
    sys.path.insert(0, str(CODE / "analysis"))
    from hydra_heads import summarize_heads, head_mapping
    model, spec, name = toy_model()
    cfg = {"mode": mode, "target_spec": spec}
    apply_method(model, cfg)
    layer = model.get_submodule(name)
    heads = layer.lora_B if mode == "hydraT" else layer.lora_A
    with torch.no_grad():
        for head in heads:
            head.weight.fill_(1)
        heads[1].weight.fill_(-1)
        heads[2].weight.zero_()
    save_method_state(model, str(tmp_path), "hydra", cfg)
    result = summarize_heads(tmp_path)
    assert result["factor"] == factor and result["n_targets"] == 1
    assert result["task_names"][0] == "classification"
    matrix = result["targets"][name]["cosine"]
    assert matrix[0][0] == pytest.approx(1)
    assert matrix[0][1] == pytest.approx(-1)
    assert matrix[0][2] is None and matrix[2][2] is None
    for group_by, group in (("target", name), ("decoder_layer", name.rsplit(".", 1)[0])):
        mapping = head_mapping(tmp_path, group_by)
        assert mapping["mode"] == mode
        assert mapping["layers"][group][result["task_names"][0]] == [f"{name}.lora_{factor}.0.weight"]
        map_path = tmp_path / f"head_map_{group_by}.json"
        map_path.write_text(json.dumps(mapping))
        from representations import head_similarity
        alex = head_similarity(tmp_path / "method_state.safetensors", map_path,
                               tmp_path / f"alex_{group_by}.json")
        assert alex["layers"][group]["cls/ml"] == pytest.approx(-1)
        assert alex["layers"][group]["cls/det"] is None


@pytest.mark.parametrize("mode, factor", [("hydraT", "B"), ("hydraR", "A")])
def test_e7_decoder_map_concatenates_multiple_targets_in_fixed_order(tmp_path, mode, factor):
    sys.path.insert(0, str(CODE / "analysis"))
    from hydra_heads import head_mapping
    from representations import head_similarity
    model, spec, name = toy_model()
    block = model.model.language_model.layers[0]
    block.v_proj = nn.Linear(5, 7)
    spec.update(target_modules=["q_proj", "v_proj"], n_modules=2)
    cfg = {"mode": mode, "target_spec": spec}
    apply_method(model, cfg)
    for target in (block.q_proj, block.v_proj):
        randomize_b(target)
    save_method_state(model, str(tmp_path), "hydra", cfg)
    mapping = head_mapping(tmp_path, "decoder_layer")
    layer = name.rsplit(".", 1)[0]
    entries = mapping["layers"][layer]
    assert len(entries["classification"]) == 2
    vectors = []
    for task in (0, 1):
        heads = [target.lora_B if factor == "B" else target.lora_A
                 for target in (block.q_proj, block.v_proj)]
        vectors.append(torch.cat([head[task].weight.detach().flatten() for head in heads]))
    expected = torch.nn.functional.cosine_similarity(*vectors, dim=0).item()
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping))
    result = head_similarity(tmp_path / "method_state.safetensors", path, tmp_path / "out.json")
    assert result["layers"][layer]["cls/ml"] == pytest.approx(expected, abs=1e-6)
