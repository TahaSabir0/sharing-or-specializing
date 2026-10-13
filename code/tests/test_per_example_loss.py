"""Unit test: per-example loss under batch 1 × gradient accumulation (PLAN.md §1.5).

Two hand-made examples, one with 2 answer tokens and one with 60, go through one optimizer step
of the real HF Trainer with JournalLossMixin (bsz 1, grad-accum 2) on a tiny random Llama (CPU).
The accumulated gradient must equal the gradient of

    0.5 * (mean CE over example 1's answer tokens + mean CE over example 2's answer tokens)

and must differ from the token-weighted alternative (sum of all answer-token CE / 62), which is
what HF does when `num_items_in_batch` reaches the model. Routing ids are also checked: the
mixin must pop them before the forward and expose them via methods.base.routing.

  python code/tests/test_per_example_loss.py      (or pytest)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments  # noqa: E402

from journal_loss import JournalLossMixin  # noqa: E402
from methods.base import routing  # noqa: E402

VOCAB = 64


def make_examples():
    g = torch.Generator().manual_seed(0)
    exs = []
    for n_prompt, n_answer, task in [(5, 2, 0), (5, 60, 5)]:
        ids = torch.randint(3, VOCAB, (n_prompt + n_answer,), generator=g)
        labels = ids.clone()
        labels[:n_prompt] = -100
        exs.append({"input_ids": ids, "labels": labels, "task": task})
    return exs


class ListDS(torch.utils.data.Dataset):
    def __init__(self, exs):
        self.exs = exs

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        return self.exs[i]


def collate(features):
    assert len(features) == 1
    f = features[0]
    return {"input_ids": f["input_ids"][None], "labels": f["labels"][None],
            "attention_mask": torch.ones_like(f["input_ids"])[None],
            "task_id": torch.tensor([f["task"]]), "modality_id": torch.tensor([0]),
            "dataset_id": torch.tensor([0])}


class GradCapture(TrainerCallback):
    grads = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
        GradCapture.grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                             if p.grad is not None}


class SeenRouting(LlamaForCausalLM):
    seen = []

    def forward(self, *a, **kw):
        assert "task_id" not in kw, "routing ids must be popped before the forward"
        assert "num_items_in_batch" not in kw, "num_items_in_batch must not reach the model"
        tid = routing.get("task_id")
        SeenRouting.seen.append(None if tid is None else int(tid.item()))
        return super().forward(*a, **kw)


class T(JournalLossMixin, Trainer):
    pass


def manual_grads(model, exs, mode):
    model.zero_grad()
    losses, tokens = [], 0
    for e in exs:
        out = model(input_ids=e["input_ids"][None])
        logits = out.logits[0, :-1].float()
        tgt = e["labels"][1:]
        ce = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=-100, reduction="sum")
        n = int((tgt != -100).sum())
        losses.append((ce, n))
        tokens += n
    if mode == "per_example":
        loss = sum(ce / n for ce, n in losses) / len(losses)
    else:
        loss = sum(ce for ce, _ in losses) / tokens
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_per_example_loss(tmp_dir="/tmp/j26_loss_test"):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
    model = SeenRouting(cfg).float()
    init = {k: v.clone() for k, v in model.state_dict().items()}
    exs = make_examples()

    args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=1,
                             gradient_accumulation_steps=2, max_steps=1, learning_rate=1e-3,
                             max_grad_norm=0.0, report_to="none", save_strategy="no",
                             logging_steps=1, use_cpu=True, dataloader_num_workers=0,
                             remove_unused_columns=False, seed=0)
    tr = T(model=model, args=args, train_dataset=ListDS(exs), data_collator=collate,
           callbacks=[GradCapture()])
    assert tr.model_accepts_loss_kwargs is False
    tr.train()
    got = GradCapture.grads
    assert got, "no gradients captured"
    assert sorted(x for x in SeenRouting.seen if x is not None) == [0, 5], SeenRouting.seen

    ref = SeenRouting(cfg).float()
    ref.load_state_dict(init)
    ref.train()
    want = manual_grads(ref, exs, "per_example")
    tok = manual_grads(ref, exs, "token")

    def rel(a, b):
        num = sum(((a[k] - b[k]) ** 2).sum() for k in a)
        den = sum((b[k] ** 2).sum() for k in b)
        return float((num / den).sqrt())

    e_per, e_tok = rel(got, want), rel(got, tok)
    print(f"relative grad error vs per-example: {e_per:.2e}; vs token-weighted: {e_tok:.2e}")
    assert e_per < 1e-4, f"accumulated gradient is not the per-example mean (rel err {e_per})"
    assert e_tok > 1e-2, "test not discriminative: per-example and token-weighted coincide"
    print("PASS per-example loss")


if __name__ == "__main__":
    test_per_example_loss()
