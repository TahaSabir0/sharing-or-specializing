"""Tiny, randomly initialized Transformers models; no downloads, images or GPU.

These exercise the real projection shapes, model forwards and KV-cache paths;
they do not certify the production NF4/MedGemma port or its image template.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.base import load_targets, routing
from methods.hydra import apply_method


def tiny_model(backbone):
    from transformers import (Gemma3Config, Gemma3ForConditionalGeneration,
                              Qwen3_5Config, Qwen3_5ForConditionalGeneration)
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                pad_token_id=0, eos_token_id=1)
    if backbone == "q35":
        text.update(linear_num_key_heads=1, linear_num_value_heads=2,
                    linear_key_head_dim=16, linear_value_head_dim=16,
                    layer_types=["linear_attention", "full_attention"])
        config = Qwen3_5Config(
            text_config=text,
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                               out_hidden_size=32, num_position_embeddings=16),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        model = Qwen3_5ForConditionalGeneration(config)
        leaves = load_targets()["target_modules"]
    else:
        text.update(query_pre_attn_scalar=16, sliding_window=16,
                    layer_types=["sliding_attention", "full_attention"])
        config = Gemma3Config(
            text_config=text,
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                               num_attention_heads=2, image_size=8, patch_size=2),
            mm_tokens_per_image=4, image_token_index=60, boi_token_index=61, eoi_token_index=62)
        model = Gemma3ForConditionalGeneration(config)
        leaves = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    scope = "model.language_model.layers"
    n = sum(isinstance(mod, nn.Linear) and name.startswith(scope + ".")
            and name.rsplit(".", 1)[-1] in leaves for name, mod in model.named_modules())
    return model, {"scope_prefix": scope, "target_modules": leaves, "n_modules": n}


@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
@pytest.mark.parametrize("backbone", ["q35", "gemma3"])
@pytest.mark.parametrize("tasks", [[0], [0, 5]])
def test_real_transformers_training_and_cached_generation(mode, backbone, tasks):
    torch.manual_seed(1337)
    model, spec = tiny_model(backbone)
    x = torch.randint(3, 50, (len(tasks), 6))
    model.eval()
    with torch.no_grad():
        initial_logits = model(input_ids=x).logits
    apply_method(model, {"mode": mode, "r": 2, "dropout": 0, "target_spec": spec})
    try:
        with routing.use(task_id=torch.tensor(tasks)):
            with torch.no_grad():
                torch.testing.assert_close(model(input_ids=x).logits, initial_logits, rtol=0, atol=0)
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.enable_input_require_grads()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=0.01, weight_decay=0)
            losses = []
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                loss = model(input_ids=x, labels=x, use_cache=False).loss
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            assert losses[-1] < losses[0]
            model.gradient_checkpointing_disable()
            model.eval()
            args = dict(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=3,
                        do_sample=False, num_beams=1, eos_token_id=None, pad_token_id=0,
                        return_dict_in_generate=True, output_scores=True)
            cached = model.generate(**args, use_cache=True)
            uncached = model.generate(**args, use_cache=False)
            torch.testing.assert_close(cached.sequences, uncached.sequences)
            for got, want in zip(cached.scores, uncached.scores):
                torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    finally:
        routing.clear()
