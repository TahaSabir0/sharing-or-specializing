#!/usr/bin/env python3
"""W2: journal_train, the training wrapper around LLaMA-Factory v0.9.5 (PLAN.md §1.2–§1.5, §1.10, §3).

  CUDA_VISIBLE_DEVICES=1 python code/train/journal_train.py --config code/configs/s1/shared_s1337.yaml
  ... --max_steps 50 --no_eval --run_suffix smoke      # smoke gate item 1

What it adds on top of `llamafactory-cli train`:
- The fixed protocol (PROTOCOL below) pinned explicitly; a run config may only override keys
  through `overrides:`, and every override is recorded as a deviation in config.yaml.
- Per-example loss (journal_loss.JournalLossMixin; unit test in tests/).
- Routing ids: task_id / modality_id / dataset_id flow from the deck rows through LF's converter,
  tokenizer step and collator to the trainer, which exposes them via methods.base.routing.
- Plug-in hook: `method: <name>` other than `lora` replaces LF's LoRA setup with
  methods/<name>.py::apply_method(model, method_cfg). Only trainable parameters are saved.
- aux_loss: summed from modules and added to the loss.
- Run registry: $JOURNAL_RUNS (default /data/ugrip2026/journal/runs)/<run_id>/ with config.yaml, git_commit.txt,
  decks.json, trainer_log.jsonl (LF), task_loss.jsonl, runtime.json, registry_row.csv.

Run config (YAML):
  run_id: j26_q35_shared_full_768_s1337
  owner: taha
  study: S1
  method: lora            # or a plug-in name in code/methods/
  method_cfg: {}
  deck: j26_train_mix     # name in <data_dir>/dataset_info.json
  data_tag: full
  seed: 1337
  eval_deck: j26_devselect  # optional; dev-loss deck (default j26_devselect)
  overrides: {}           # LF/HF keys that deviate from PROTOCOL (logged)
"""
import argparse
import copy
import csv
import datetime
import hashlib
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from methods import backbones  # noqa: E402
from methods import base as mb  # noqa: E402

RUNS_ROOT = os.environ.get("JOURNAL_RUNS", "/data/ugrip2026/journal/runs")
DATA_DIR = os.environ.get("JOURNAL_DATA", "/data/ugrip2026/journal/data")
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/ugrip2026/models")
TARGETS = mb.load_targets()   # Qwen's (PROTOCOL below is the Qwen recipe); other backbones: methods/backbones.py

# PLAN.md §1.1–§1.4. Every value pinned explicitly, including framework defaults.
# This is the Qwen3.5 recipe. A run config with `backbone: mg15` gets backbones.lf_overrides("mg15")
# on top (model, template, image budget, target modules); `backbone: q35` adds nothing.
PROTOCOL = {
    # model / loading
    "model_name_or_path": f"{MODELS_DIR}/Qwen3.5-9B",
    "trust_remote_code": True,
    "image_max_pixels": 589824,              # 768² area cap
    "quantization_bit": 4,
    "quantization_method": "bnb",
    "double_quantization": False,            # NF4 is LF's bnb default type
    # method
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target": ",".join(TARGETS["target_modules"]),
    "freeze_vision_tower": True,
    "freeze_multi_modal_projector": True,
    # data / template
    "template": "qwen3_5",
    "enable_thinking": False,
    "cutoff_len": 8192,
    "preprocessing_num_workers": 16,
    "dataloader_num_workers": 4,
    "overwrite_cache": True,                 # never reuse a cache built without routing columns
    "packing": False,
    # optimisation
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2.0e-4,
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0.03,                    # <1 means a ratio of total steps (HF ≥5)
    "optim": "adamw_torch",
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1.0e-8,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    # checkpoints / logging (end-of-epoch checkpoints, dev loss for diagnostics only)
    "save_strategy": "epoch",
    "save_only_model": False,
    "eval_strategy": "epoch",
    "eval_dataset": "j26_devselect",
    "per_device_eval_batch_size": 1,
    "logging_steps": 10,
    "plot_loss": True,
    "report_to": "none",
    "ddp_timeout": 180000000,
}


# ------------------------------------------------------------------------------------ helpers
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def git_info():
    commit = sh(f"git -C {REPO} rev-parse HEAD")
    branch = sh(f"git -C {REPO} rev-parse --abbrev-ref HEAD")
    dirty = sh(f"git -C {REPO} status --porcelain --untracked-files=no")
    return commit, branch, dirty


def gpu_snapshot():
    return {
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpus": sh("nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu "
                   "--format=csv,noheader"),
        "apps": sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader"),
    }


def deck_record(name):
    info = json.load(open(os.path.join(DATA_DIR, "dataset_info.json")))
    if name not in info:
        raise SystemExit(f"deck {name!r} not in {DATA_DIR}/dataset_info.json")
    path = os.path.join(DATA_DIR, info[name]["file_name"])
    rec = {"name": name, "path": path, "md5": md5(path)}
    for manifest_name in ("MANIFEST.json", "GROUP_MANIFEST.json"):
        man_path = os.path.join(DATA_DIR, manifest_name)
        if os.path.exists(man_path):
            man = json.load(open(man_path))["decks"].get(name)
            if man is not None:
                rec["manifest_md5"] = man["md5"]
                if man["md5"] != rec["md5"]:
                    raise SystemExit(f"deck {name} md5 {rec['md5']} != manifest {man['md5']}")
    return rec


# ------------------------------------------------------------------------------------ patches
def install_patches(method, method_cfg, run_dir, backbone=backbones.DEFAULT_BACKBONE):
    """Monkeypatch LF so routing ids reach the trainer and plug-ins replace LF's LoRA."""
    import torch
    from llamafactory.data import converter as C
    from llamafactory.data import collator as K
    from llamafactory.data.processor import supervised as S
    import llamafactory.model.loader as L
    import llamafactory.train.sft.workflow as W
    from journal_loss import JournalLossMixin

    # (1) converter: keep the routing labels next to LF's aligned columns
    orig_conv = C.SharegptDatasetConverter.__call__

    def conv_call(self, example):
        out = orig_conv(self, example)
        out["_routing"] = [int(example[k]) if example.get(k) is not None else -1 for k in mb.ROUTING_KEYS]
        return out

    C.SharegptDatasetConverter.__call__ = conv_call

    # (2) tokenizer step: carry them through (same validity rule as LF's own loop)
    orig_pp = S.SupervisedDatasetProcessor.preprocess_dataset

    def pp(self, examples):
        out = orig_pp(self, examples)
        if "_routing" in examples:
            keep = [i for i in range(len(examples["_prompt"]))
                    if len(examples["_prompt"][i]) % 2 == 1 and len(examples["_response"][i]) == 1]
            out["routing"] = [examples["_routing"][i] for i in keep]
            assert len(out["routing"]) == len(out["input_ids"])
        return out

    S.SupervisedDatasetProcessor.preprocess_dataset = pp

    # (3) collator: stack into task_id / modality_id / dataset_id
    orig_coll = K.SFTDataCollatorWith4DAttentionMask.__call__

    def coll_call(self, features):
        rows = [f.pop("routing", None) for f in features]
        batch = orig_coll(self, features)
        if rows and all(r is not None for r in rows):
            t = torch.tensor(rows, dtype=torch.long)
            for i, k in enumerate(mb.ROUTING_KEYS):
                batch[k] = t[:, i]
        return batch

    K.SFTDataCollatorWith4DAttentionMask.__call__ = coll_call

    # (3b) MedGemma only: LF marks the wrong tokens as image in token_type_ids (see backbones.py)
    if backbone == "mg15":
        backbones.patch_lf_gemma3_token_type_ids()
        print("[journal_train] mg15: LF gemma3 token_type_ids patched (all image soft tokens, as the processor)", flush=True)

    # (4) plug-in hook in place of LF's adapter setup
    if method != "lora":
        plugin = importlib.import_module(f"methods.{method}")

        def init_plugin(config, model, model_args, finetuning_args, is_trainable):
            model.requires_grad_(False)
            model = plugin.apply_method(model, method_cfg)
            for p in model.parameters():          # LF upcasts trainable LoRA params to fp32 too
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
            if mb.count_trainable(model) == 0:
                raise RuntimeError(f"plug-in {method} left no trainable parameters")
            model._hf_peft_config_loaded = True   # lets HF Trainer accept the quantized model
            return model

        L.init_adapter = init_plugin

    # (5) trainer: per-example loss, routing, aux loss, per-task loss log, plug-in saving
    label_ids = json.load(open(os.path.join(CODE, "data", "label_ids.json")))

    class JournalTrainer(JournalLossMixin, W.CustomSeq2SeqTrainer):
        journal_task_names = {v: k for k, v in label_ids["task"].items()}
        journal_task_loss_path = os.path.join(run_dir, "task_loss.jsonl")

        def _save(self, output_dir=None, state_dict=None):
            if method == "lora":
                return super()._save(output_dir, state_dict)
            output_dir = output_dir or self.args.output_dir
            mb.save_method_state(self.model, output_dir, method, method_cfg)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    W.CustomSeq2SeqTrainer = JournalTrainer


def runtime_callback(run_dir, method, p8=None):
    import torch
    p8 = TARGETS["P8"] if p8 is None else p8
    from transformers import TrainerCallback

    class RuntimeCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kw):
            self.t0 = time.time()
            self.start = datetime.datetime.now().isoformat(timespec="seconds")
            self.gpu_begin = gpu_snapshot()
            self.trainable = mb.count_trainable(model) if model is not None else None
            torch.cuda.reset_peak_memory_stats()

        def on_train_end(self, args, state, control, **kw):
            wall = time.time() - self.t0
            steps = max(state.global_step, 1)
            rt = {
                "host": socket.gethostname(), "started": self.start,
                "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                "wall_s": round(wall, 1), "optimizer_steps": state.global_step,
                "s_per_it": round(wall / steps, 3),
                "s_per_example": round(wall / (steps * args.per_device_train_batch_size
                                               * args.gradient_accumulation_steps), 4),
                "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "trainable_params": self.trainable,
                "P8": p8,
                "trainable_matches_P8": (self.trainable == p8) if method == "lora" else None,
                "gpu_begin": self.gpu_begin, "gpu_end": gpu_snapshot(),
                "note": "wall_s includes dev-loss evaluation and checkpoint saving",
            }
            json.dump(rt, open(os.path.join(run_dir, "runtime.json"), "w"), indent=1)

    return RuntimeCallback()


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke runs: stop after N optimizer steps")
    ap.add_argument("--no_eval", action="store_true", help="skip dev-loss evaluation (smoke runs)")
    ap.add_argument("--run_suffix", default=None, help="appended to run_id, e.g. smoke")
    ap.add_argument("--dry", action="store_true", help="write the run folder and resolved config, don't train")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    run_id = cfg["run_id"] + (f"_{a.run_suffix}" if a.run_suffix else "")
    assert run_id.startswith("j26_"), "run ids follow j26_{backbone}_{method}_{data}_{res}_s{seed}"
    backbone = cfg.get("backbone", backbones.DEFAULT_BACKBONE)
    try:
        backbones.get_backbone(backbone)
    except ValueError as e:
        raise SystemExit(str(e))
    if f"_{backbone}_" not in run_id:
        raise SystemExit(f"run_id {run_id!r} does not carry the backbone tag _{backbone}_")
    method = cfg.get("method", "lora")
    method_cfg = cfg.get("method_cfg") or {}
    run_dir = os.path.join(RUNS_ROOT, run_id)
    if os.path.exists(os.path.join(run_dir, "runtime.json")):
        raise SystemExit(f"{run_dir} already holds a finished run; pick a new run_id")
    os.makedirs(run_dir, exist_ok=True)

    lf = copy.deepcopy(PROTOCOL)
    backbone_args = backbones.lf_overrides(backbone, MODELS_DIR)   # {} for q35
    lf.update(backbone_args)
    lf.update({"dataset": cfg["deck"], "dataset_dir": DATA_DIR, "output_dir": run_dir,
               "seed": int(cfg["seed"]), "data_seed": int(cfg["seed"])})
    if cfg.get("eval_deck"):
        lf["eval_dataset"] = cfg["eval_deck"]
    overrides = cfg.get("overrides") or {}
    recipe = {**PROTOCOL, **backbone_args}   # what a run of this backbone uses unless it overrides
    deviations = {k: {"protocol": recipe.get(k), "run": v} for k, v in overrides.items()
                  if recipe.get(k) != v}
    lf.update(overrides)
    smoke = {}
    if a.max_steps:
        smoke["max_steps"] = a.max_steps
        smoke["save_strategy"] = "no"
        # tokenize only what the smoke run can use (rows are pre-shuffled in the mix deck)
        smoke["max_samples"] = a.max_steps * lf["per_device_train_batch_size"] * lf["gradient_accumulation_steps"] * 2
    if a.no_eval:
        smoke["eval_strategy"] = "no"
        lf.pop("eval_dataset", None)
    lf.update(smoke)
    if lf.get("eval_strategy", "no") == "no":
        lf.pop("eval_dataset", None)
    if not smoke:
        # tokenize each deck once and reuse it across runs (keyed by deck md5 + protocol fields that
        # change tokenization); built with the routing patches, so the routing column is included
        key = hashlib.md5(json.dumps({k: lf.get(k) for k in (
            "dataset", "eval_dataset", "template", "enable_thinking", "cutoff_len", "image_max_pixels",
            "model_name_or_path")}, sort_keys=True).encode()).hexdigest()[:10]
        lf["tokenized_path"] = os.path.join(DATA_DIR, "tokenized", f"{lf['dataset']}_{key}")
        lf["overwrite_cache"] = False

    commit, branch, dirty = git_info()
    decks = {"train": deck_record(lf["dataset"])}
    if lf.get("eval_dataset"):
        decks["eval"] = deck_record(lf["eval_dataset"])
    started = datetime.datetime.now().isoformat(timespec="seconds")
    resolved = {"run_id": run_id, "owner": cfg.get("owner"), "study": cfg.get("study"),
                "backbone": backbone, "backbone_overrides": backbone_args,
                "method": method, "method_cfg": method_cfg,
                "data_tag": cfg.get("data_tag"), "seed": int(cfg["seed"]),
                "deviations": deviations, "smoke": smoke, "llamafactory_args": lf,
                "source_config": os.path.abspath(a.config)}
    yaml.safe_dump(resolved, open(os.path.join(run_dir, "config.yaml"), "w"), sort_keys=False)
    with open(os.path.join(run_dir, "git_commit.txt"), "w") as f:
        f.write(f"{commit}\nbranch {branch}\n" + (f"DIRTY (tracked changes):\n{dirty}\n" if dirty else "clean\n"))
    json.dump(decks, open(os.path.join(run_dir, "decks.json"), "w"), indent=1)

    row = {"run_id": run_id, "owner": cfg.get("owner", ""), "study": cfg.get("study", ""),
           "backbone": resolved["backbone"], "method": method, "data": cfg.get("data_tag", ""),
           "seed": cfg["seed"], "code_commit": commit[:7] + ("-dirty" if dirty else ""),
           "host": socket.gethostname(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
           "status": "running", "started": started, "finished": "", "output_dir": run_dir,
           "notes": "smoke" if smoke else ""}

    def write_row(r):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(r))
        w.writeheader()
        w.writerow(r)
        open(os.path.join(run_dir, "registry_row.csv"), "w").write(buf.getvalue())

    write_row(row)
    print(f"[journal_train] run {run_id} -> {run_dir}  (commit {commit[:7]}{' DIRTY' if dirty else ''})")
    if deviations:
        print(f"[journal_train] DEVIATIONS from protocol: {deviations}")
    if a.dry:
        return

    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    install_patches(method, method_cfg, run_dir, backbone)
    from llamafactory.train.tuner import run_exp
    try:
        run_exp(args=lf, callbacks=[runtime_callback(run_dir, method, backbones.expected_p8(backbone))])
        row["status"] = "done"
    except BaseException:
        row["status"] = "failed"
        raise
    finally:
        row["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        write_row(row)


if __name__ == "__main__":
    main()
