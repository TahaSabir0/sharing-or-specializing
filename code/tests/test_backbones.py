"""Unit test: backbone selection and LoRA target-module resolution (PLAN.md §1.1, §1.2, W9).

CPU only. The checks that need no torch run anywhere; the last section (methods.base on a toy
module tree) needs torch and is skipped without it.

Checks:
- q35 is untouched: no LF overrides, Qwen's target file for qwen3_5 / unknown / missing model types;
- the frozen target files are self-consistent (module count, P8 from the listed shapes);
- mg15 resolves to the Gemma3 recipe: model, template, no 768² cap, language-model targets only;
- on synthetic Qwen3.5 and Gemma3 module names (current and legacy checkpoint naming) the target
  predicate picks exactly 248 / 238 modules and never a vision-tower or projector module;
- every run config in code/configs names a known backbone and carries its tag in the run_id.

  python code/tests/test_backbones.py      (or pytest)
"""
import glob
import os
import re
import sys

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, CODE)

from methods import backbones as bb  # noqa: E402


def _p8(spec, r=8):
    total = 0
    for _, (n, shape) in spec["shapes_in_x_out"].items():
        i, o = (int(x) for x in shape.split("x"))
        total += n * r * (i + o)
    return total


def _n_modules(spec):
    return sum(n for n, _ in spec["shapes_in_x_out"].values())


def qwen_names():
    """Qwen3.5-9B: 32 layers = 24 linear-attention + 8 full-attention, MLP on all; ViT outside."""
    names = []
    for i in range(32):
        p = f"model.language_model.layers.{i}"
        attn = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj") if i % 4 == 3 else \
            ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
             "linear_attn.in_proj_b", "linear_attn.out_proj")
        names += [f"{p}.{a}" for a in attn] + [f"{p}.mlp.{m}" for m in ("gate_proj", "up_proj", "down_proj")]
        names += [f"{p}.input_layernorm"]
    for i in range(27):
        names += [f"model.visual.blocks.{i}.attn.qkv", f"model.visual.blocks.{i}.attn.proj",
                  f"model.visual.blocks.{i}.mlp.linear_fc1", f"model.visual.blocks.{i}.mlp.linear_fc2"]
    return names + ["model.visual.merger.linear_fc1", "model.language_model.embed_tokens", "lm_head"]


def gemma_names(prefix="model.language_model", vision="vision_tower.vision_model"):
    """Gemma3-4B: 34 decoder layers; SigLIP has 27 layers whose attention reuses q/k/v_proj."""
    names = []
    for i in range(34):
        p = f"{prefix}.layers.{i}"
        names += [f"{p}.self_attn.{x}_proj" for x in "qkvo"]
        names += [f"{p}.mlp.{m}_proj" for m in ("gate", "up", "down")]
        names += [f"{p}.input_layernorm", f"{p}.self_attn.q_norm", f"{p}.self_attn.k_norm"]
    for i in range(27):
        v = f"{vision}.encoder.layers.{i}"
        names += [f"{v}.self_attn.{x}_proj" for x in ("q", "k", "v")] + [f"{v}.self_attn.out_proj",
                                                                         f"{v}.mlp.fc1", f"{v}.mlp.fc2"]
    return names + [f"{prefix}.embed_tokens", "multi_modal_projector", "lm_head"]


def pick(names, spec):
    return [n for n in names if bb.matches_target(n, spec)]


# ---------------------------------------------------------------------------------- q35 unchanged
def test_q35_adds_nothing():
    assert bb.lf_overrides("q35", "/models") == {}
    assert bb.DEFAULT_BACKBONE == "q35"


def test_qwen_target_file_is_the_fallback():
    q = bb.QWEN_TARGETS_PATH
    assert os.path.basename(q) == "target_modules_qwen35.json"
    for mt in ("qwen3_5", None, "something_new"):
        assert bb.targets_path_for_model_type(mt) == q, mt
    assert os.path.basename(bb.targets_path_for_model_type("gemma3")) == "target_modules_medgemma15.json"


def test_qwen_spec_unchanged():
    s = bb.load_spec("q35")
    assert s["n_modules"] == 248 and s["P8"] == 21639168 and s["scope_prefix"] == "model.language_model.layers"
    assert _n_modules(s) == 248 and _p8(s) == 21639168


def test_qwen_resolution_248_and_no_vision():
    s = bb.load_spec("q35")
    got = pick(qwen_names(), s)
    assert len(got) == 248
    assert not any("visual" in n for n in got)


# ----------------------------------------------------------------------------------------- mg15
def test_mg15_spec_self_consistent():
    s = bb.load_spec("mg15")
    assert _n_modules(s) == s["n_modules"] == 7 * s["layers"]["total"] == 238
    assert _p8(s) == s["P8"] == 14901248
    assert s["model_type"] == bb.get_backbone("mg15").model_type == "gemma3"
    assert bb.expected_p8("mg15") == 14901248 and bb.expected_p8("q35") == 21639168


def test_mg15_lf_overrides():
    o = bb.lf_overrides("mg15", "/data/ugrip2026/models")
    assert o["model_name_or_path"] == "/data/ugrip2026/models/medgemma-1.5-4b-it"
    assert o["template"] == "gemma3"
    assert o["image_max_pixels"] > 8192 * 8192          # never triggers LF's pre-resize (its default is 768²)
    assert o["image_do_pan_and_scan"] is False
    assert o["lora_target"] == "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    assert set(o) == {"model_name_or_path", "template", "image_max_pixels", "image_do_pan_and_scan", "lora_target"}


def test_mg15_resolution_238_language_model_only():
    s = bb.load_spec("mg15")
    for prefix in ("model.language_model", "language_model.model"):   # transformers 5.x / legacy checkpoint naming
        names = gemma_names(prefix=prefix)
        got = pick(names, s)
        assert len(got) == 238, (prefix, len(got))
        assert not any("vision_tower" in n or "multi_modal_projector" in n or "lm_head" in n for n in got)
        assert all(".self_attn." in n or ".mlp." in n for n in got)


def test_scope_is_what_excludes_the_vision_tower():
    """SigLIP's q/k/v_proj share leaf names with the decoder: 27 layers x 3 = 81 would match without the scope."""
    leaves = set(bb.load_spec("mg15")["target_modules"])
    names = gemma_names()
    by_leaf = [n for n in names if n.rsplit(".", 1)[-1] in leaves]
    assert len(by_leaf) == 238 + 81


def test_unknown_backbone_rejected():
    for fn in (bb.get_backbone, bb.targets_path, bb.load_spec):
        try:
            fn("gemma9")
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted an unknown backbone")


def test_res_tags():
    assert bb.get_backbone("q35").res_tag == "768" and bb.get_backbone("mg15").res_tag == "896"


# ---------------------------------------------------------------------------------- run configs
def test_all_configs_name_a_known_backbone_in_their_run_id():
    paths = glob.glob(os.path.join(CODE, "configs", "**", "*.yaml"), recursive=True)
    assert paths
    for p in paths:
        txt = open(p).read()
        mr = re.search(r"^run_id:\s*(\S+)", txt, re.M)
        if mr is None:                      # reference templates, not run configs
            continue
        rid = mr.group(1)
        m = re.search(r"^backbone:\s*(\S+)", txt, re.M)
        tag = m.group(1) if m else bb.DEFAULT_BACKBONE
        bb.get_backbone(tag)
        assert f"_{tag}_" in rid, (p, rid, tag)
        assert f"_{bb.get_backbone(tag).res_tag}_s" in rid, (p, rid)


def test_mg15_smoke_config():
    txt = open(os.path.join(CODE, "configs", "smoke_lora_mg15.yaml")).read()
    get = lambda k: re.search(rf"^{k}:\s*(.+?)\s*$", txt, re.M).group(1)    # noqa: E731
    assert get("backbone") == "mg15" and get("method") == "lora" and get("method_cfg") == "{}"
    assert get("deck") == "j26_train_mix" and get("seed") == "1337" and get("overrides") == "{}"
    assert get("run_id") == "j26_mg15_shared_full_896_s1337"      # j26_{backbone}_{arm}_{data}_{res}_s{seed}


def test_smoke_sh_passes_the_backbone_to_the_eval():
    sh = open(os.path.join(CODE, "train", "smoke.sh")).read()
    assert "get('backbone','q35')" in sh                              # read from the YAML, Qwen default
    line = re.search(r"eval_harness\.py[^\n]*", sh).group(0)
    assert '--backbone "$BACKBONE"' in line


# ---------------------------------------------------- methods.base on a toy tree (needs torch)
def test_iter_target_modules_picks_the_file_from_model_type():
    try:
        import torch
    except ImportError:
        print("SKIP test_iter_target_modules_picks_the_file_from_model_type (no torch)")
        return
    from types import SimpleNamespace

    from methods import base as mb

    class Tree(torch.nn.Module):
        def __init__(self, names, model_type):
            super().__init__()
            self.config = SimpleNamespace(model_type=model_type)
            for n in names:
                parent = self
                *path, leaf = n.split(".")
                for p in path:
                    if not hasattr(parent, p):
                        parent.add_module(p, torch.nn.Module())
                    parent = getattr(parent, p)
                parent.add_module(leaf, torch.nn.Linear(2, 2, bias=False))

    g = Tree(gemma_names(), "gemma3")
    assert len(mb.iter_target_modules(g)) == 238
    q = Tree(qwen_names(), "qwen3_5")
    assert len(mb.iter_target_modules(q)) == 248
    assert len(mb.iter_target_modules(Tree(qwen_names(), None))) == 248        # unknown type: old behaviour


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests passed")
