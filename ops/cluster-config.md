# Compute operations guide (journal study)

> # ★ 2026-10-14: ONE BACKEND — `mbz-yaqub` (Blackwell) GPU 0. `ws` IS NOT OURS ANY MORE.
>
> - All journal training **and** evaluation run on `mbz-yaqub`, **GPU 0 only** (assignment confirmed
>   2026-10-14). Never use GPUs 1–7. `ws` is used by others: no jobs there. Its copies of the decks
>   and env are a read-only reference.
> - Outside user `mostafasalem` also had a job on GPU 0 on 2026-10-14 (~12.4 GB). Run
>   `nvidia-smi -i 0` before each launch. We need ~17 GB per training run.
> - Journal setup on the box (all verified 2026-10-14):
>   - env `flare-blackwell-v5`: package versions identical to ws `flare-ws-v5`; see `code/env/LF_VERSION.md`.
>   - LF v0.9.5 at `/data/ugrip2026/LLaMA-Factory-v5`: release commit `7af9095` + tf5 patch;
>     source byte-identical to ws.
>   - model `/data/ugrip2026/models/Qwen3.5-9B`.
>   - repo clone `/data/ugrip2026/journal/repo`. The box can't pull from GitHub: update it with a
>     git bundle copied to `/data/ugrip2026/relay/`, then `git fetch` from it.
>   - decks `/data/ugrip2026/journal/data`: identical to the ws decks once image paths are remapped;
>     md5s in `data/DECK_MANIFEST_blackwell.json`.
>   - runs `/data/ugrip2026/journal/runs/<run_id>/`.
> - Launch: `nohup bash code/train/run_chain.sh <cfg.yaml ...> > /data/ugrip2026/journal/runs/chain_gpu0.log 2>&1 &`
>   (`run_chain.sh` defaults to GPU 0 and refuses other GPUs).
> - Everything below is the pre-2026-10-14 guide. §B is still valid for access, storage and
>   hardware; the `flare-blackwell` env (Qwen3-VL) is challenge-era.

> Extracted on 2026-10-11 from the UGRIP challenge repo's `cluster-config.md`: section A (`ws`)
> and section B (`mbz-yaqub` Blackwell) only. The retired `flare` SLURM material was left behind.
> Old "idle", "running", storage and capacity statements are dated observations, not live checks.
> Passwords were removed; ask Taha. For the journal study, the training stack is the Qwen3.5
> one (`flare-ws-v5` on ws, LF v0.9.5 + tf5 patch; see PLAN.md §1.1), not the older
> flare-qwen3 examples below.

> # ⛔ TWO COMPUTE BACKENDS (2026-08-03): `ws` (experimentation/eval) + `mbz-yaqub` (flagship training). `flare` SLURM IS CLOSED.
>
> - **`ws` — Prof. Yaqub's workstation `BioMedIA-A5000` (§A below).** 4× RTX A5000 24 GB, **no SLURM**,
>   run python directly. **The experimentation + eval + Docker-build box.** §A is its authoritative ops guide.
> - **`mbz-yaqub` — 96 GB Blackwell box (§B below).** Initially granted 2026-08-03 for
>   ~3 weeks; **Taha confirmed continued access on 2026-08-27 and September jobs ran**.
>   The original August 24 expiry estimate is obsolete; verify access before new work.
>   8× RTX PRO 6000 Blackwell 96 GB — **we get GPU 0 ONLY, no Docker.** This is the **FLAGSHIP 2048²
>   TRAINING box** — it fixes the blocker that 2048² OOMs on the 24 GB A5000s. **sm_120 breaks the pinned
>   cu121 recipe** → it runs a separate env `flare-blackwell` (torch 2.11 cu128). **§B is its authoritative
>   ops guide.** Division of labor: **train the flagship on `mbz-yaqub`, evaluate + build Docker on `ws`.**
> - **`flare` (`10.127.79.251`) can no longer run jobs.** Our SLURM account
>   (`project_8_multimodal_medical_image_parsing`) and all UGRIP user associations were **deleted from
>   slurmdbd on 2026-07-28**: `sbatch`/`srun` fail on every QOS with *"Invalid account or account/partition
>   combination specified"*, and `sacctmgr show associations user=taha.sabir` is empty. The GPUs are idle
>   but unreachable. This was the deferred end-of-UGRIP cleanup finally landing; **we are not contesting it.**
> - **Do NOT write, plan, or delegate `sbatch` / `srun` / `-p ugrip` / `-p ugrip_80` / `-q prod` / `-q debug`
>   / `-q tsqos` commands.** Everything below §A that describes requesting GPUs or submitting jobs is
>   **historical record only**, kept for reproducing what we already ran and for the paper's methods section.
> - **Capacity reality check:** we went from ~28× A100 (40/80 GB) to **4× A5000 24 GB shared by all 4
>   interns**. No more 6-way fan-outs; scope experiments to one or two cards and coordinate on Teams.
> - **Nothing is stranded.** A full asset audit (2026-07-28) copied all 36 trained adapters, the base
>   models, the GLiClass router, the frozen `internal_splits_v2`, the dataset, and the per-question
>   prediction archive onto `ws` (+ local + GitHub). `flare`'s SSH and filesystem still work, so it
>   remains reachable as a **read-only asset source of last resort** — from `ws` only, as
>   `taha.sabir@10.127.79.251` (the `flare` hostname does not resolve from `ws`).
> - Preserve `backup_flare/adapters/` and `adapters_rescue/`; do not clean them up.
>   The July loss of SLURM made reproducing those runs expensive. Later Blackwell
>   access enabled new flagship retrains, so the earlier blanket "un-retrainable"
>   statement is superseded; it does not make the saved adapters disposable.

---

## A. ★ CURRENT ENVIRONMENT — workstation `BioMedIA-A5000` (verified 2026-07-04)

A **bare single workstation** shared by the 4 interns (`alberto`, `taha`, `thinh`, `veronika`). **No
SLURM, no login/compute split** — you SSH in and run directly. Manage the 4 GPUs amongst yourselves.

### A.1 Access
- **`ssh ws`** — passwordless (ed25519 key `~/.ssh/id_ed25519`), user **`taha`**, host **`10.127.30.108`**,
  port 22. Requires the **MBZUAI VPN up** (internal `10.127.x`). Alias is in `~/.ssh/config` on Taha's
  laptop.
- The `cluster-executor` subagent connects with `ssh ws` for this machine (NOT `ssh flare`). For
  Windows→ssh quoting sanity, prefer the **Bash tool (Git Bash)** over PowerShell — PowerShell mangles
  nested quotes/`|`/parens in remote commands; wrap remote scripts in heredocs.

### A.2 Hardware
- **4× NVIDIA RTX A5000, 24 GB each** (driver 580.159.03; CUDA 12.x runtimes OK). All idle at handoff.
- Ubuntu 22.04.5 LTS, **32 CPU cores, 251 GB RAM**.
- **No scheduler.** Pick a GPU manually with `CUDA_VISIBLE_DEVICES=<n>`; coordinate who's on which card
  over Teams. Per-job VRAM ceiling is **24 GB** — every QLoRA-8B job fits (batch 1 + grad-checkpointing;
  ~15–25 GB), inference ~8 GB. The team is **parallelism-bound, not VRAM-bound**: 4 cards ≈ one per intern.
  For a single big job, multi-GPU DDP across the 4 cards works (each replica < 24 GB).

### A.3 Storage (keep HOME light — it's nearly full)
| Path | Free | Use |
|---|---|---|
| **`/share/sda/ugrip`** | ~1.2 TB | ✅ **our data home** (setgid `ugrip` group → files auto-shared). Everything big goes here. |
| `/home` | ~450 GB, **87% full** | keep light — no models/datasets/checkpoints here |
| `/share/sdb` | ~400 GB | other projects; avoid |

### A.4 Staged assets (all under `/share/sda/ugrip`, pulled from the old cluster over LAN 2026-07-04)
- **Env:** conda **`flare-qwen3`** at `/share/sda/ugrip/taha/miniconda3/envs/flare-qwen3` (python 3.10).
  **Verified:** torch **2.3.1+cu121**, `cuda.is_available()==True`, **NGPU==4**; transformers 4.57.1,
  peft 0.15.2, bitsandbytes 0.46.0 (matches the old `flare-qwen3` freeze). **No** nvrtc/cuDNN fixes
  needed here (clean build — the §3e gotchas do NOT apply on this box).
- **Models:** `/share/sda/ugrip/models/` → `Qwen3-VL-8B-Instruct` (17 G, backbone) + `CRIMSON-medgemma-4b`
  (18 G, report-gen judge). Point `MODELS_DIR=/share/sda/ugrip/models`.
- **Our outputs:** `/share/sda/ugrip/backup_flare/` → `adapters/` (**all 121 trained LoRA adapters**, final
  files only — no checkpoints; the 7 deployed are `adapters/qwen3/qwen3_final/*_full_s1337`), `router/`
  (GLiClass), `predictions/`, `splits/` (`internal_splits_v2`), `MANIFEST.md` (adapter→experiment map).
- **Rescued adapters:** `/share/sda/ugrip/taha/adapters_rescue/` (1.3 GB) — the **13 adapters that existed
  only on `flare`**, copied here in the 2026-07-28 audit before the cluster closed: `numeric` (E1),
  `regmix_A1–A4`, `regmix_V1/V2`, `regression_cot_{s1337,s42,s7}`, `regression_2048_s1337` (the bare
  regression specialist = the 12.70 study baseline), `multilabel_2048_s1337`, `multilabel_noavg_768_s1337`.
  Root files only, `checkpoint-*` excluded; each verified to hold `adapter_model.safetensors`.
  **Un-retrainable — do not delete.**
- **Repo:** `/share/sda/ugrip/taha/ME-VLIP-qwen3` @ branch **`Taha/final-pipeline`** (origin
  `github.com/TahaSabir0/MICCAI-FLARE-2026-Task-3`). Git auth via a fine-grained PAT in `~/.git-credentials`
  (chmod 600, in HOME not `/share`).
- **Dataset:** ✅ **staged** at `/share/sda/ugrip/datasets/FLARE-MLLM-2D/` (34 G) — the updated 2026 repo
  `FLARE-MedFM/FLARE-MLLM-2D` (HF, gated; Taha's Read token in `~/.cache/huggingface/token`). Pulled via a
  resume-loop with `HF_HUB_DISABLE_XET=1` (Xet backend hits a 429 "1000 API req / 5 min" limit on the ~50k
  small files — the loop resumes past it). Splits `training/`+`validation-public/`+`validation-hidden/`
  (no `testing/`, withheld). **11 image dirs shipped as `imagesTr.zip`/`imagesVal.zip` and were extracted
  in place** (chestdr, endo, CMMD, bcn20000, retino, iugc, IU_XRay, periapical, neojaundice, +2 val).
  **Split-compat verified:** our frozen `internal_splits_v2` image refs resolve **100% for all 17 datasets
  we use**; only `chromosome` (dead instance-detection task) + `endo` (excluded video) differ — both unused,
  so it's a clean drop-in. **Content diff also verified (2026-07-05):** of 45,806 split rows, **44,558 matched
  the new dataset with byte-identical answers, 0 answer changes**; the 1,248 not-found are all `chromosome`.
  The update vs `FLARE-Task5-MLLM-2D` was purely: drop dead `chromosome`, relocate `endo` image files
  (Q metadata unchanged), add ~20 detection/xray samples (BUSI-det +17, periapical +2, boneresorption +1) not
  in our splits. **⇒ frozen splits + all prior experiment results remain valid; nothing to re-run.**
  **✅ Split paths REMAPPED (2026-07-05):** the split `image` fields' old-cluster prefix
  `/home/…/shared/datasets/FLARE-Task5-MLLM-2D/` was rewritten in place to `/share/sda/ugrip/datasets/FLARE-MLLM-2D/`
  across all 13 jsonls (167,783 paths); **verified 100% resolve for all 17 used datasets** (only unused chromosome/endo
  missing). Pristine originals kept at `internal_splits_v2_orig_bak/`. Router also symlinked to
  `MODELS_DIR/FLARE-gliclass-small-v1.0` (inference.py hardcodes that path). **⇒ the box is launch-ready.** ⚠️ **Split paths still point to the OLD cluster root** (`/home/…/FLARE-Task5-MLLM-2D/`)
  — re-point to `/share/sda/ugrip/datasets/FLARE-MLLM-2D/` when running training/eval (relative structure matches).

### A.5 Activate the env (every non-interactive shell)
```bash
source /share/sda/ugrip/taha/miniconda3/etc/profile.d/conda.sh
conda activate flare-qwen3
export MODELS_DIR=/share/sda/ugrip/models
```
> Env-build gotcha (already solved, for reference): fresh conda needs `conda tos accept` OR create envs
> with `-c conda-forge --override-channels`, else `CondaToSNonInteractiveError` aborts `conda create`.

### A.6 Run jobs (NO SLURM — run directly, detach long jobs)
- **No `srun`/`sbatch`.** Just run python after activating. Choose a GPU: `CUDA_VISIBLE_DEVICES=2 python ...`.
- **Long jobs → detach** (VPN is flaky; an attached SSH drop kills the job):
  `cd <dir> && nohup bash job.sh > job.log 2>&1 &` then exit; collect later by grepping markers in the log.
  This is the standing pattern for this box (same launch-and-detach rule as the old cluster).
- Check GPU use before grabbing one: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`.

### A.7 Docker (submission container) — ✅ WORKING (2026-07-05)
- Docker **is installed and usable** — `taha` was added to the `docker` group (Yaqub), so `docker ps`/
  `docker build` work **without sudo** after a fresh login (docker 24.0.2 client+server). The submission
  container path is unblocked. ⚠️ Only `taha` is confirmed in the group so far — the other interns may need
  adding too if they build containers (`groups` should list `docker`; else ask Yaqub). No passwordless sudo
  otherwise, but nothing else needs it.

### A.8 Old cluster as a read-only LAN asset source (last resort)
`flare` **cannot run jobs any more** (SLURM account deleted 2026-07-28 — see the banner at the top),
but its SSH and filesystem still respond, and it is reachable **from the workstation only** over LAN as
`taha.sabir@10.127.79.251` (the workstation's key is authorized there; ⚠️ the `flare` hostname does
**not** resolve from `ws`). Use it *only* to pull a file that turns out to be missing here:
`rsync -aP taha.sabir@10.127.79.251:~/… /share/sda/ugrip/…`. **You should not need it** — the 2026-07-28
asset audit confirmed all 36 adapters, both base models, the router, the frozen splits, the dataset and
the per-question prediction archive already exist on `ws` (adapters also mirrored locally + on GitHub).
Assume it can be wiped without warning; never leave anything only there.

---

## B. ★ FLAGSHIP-TRAINING BOX — `mbz-yaqub` (RTX PRO 6000 Blackwell, verified 2026-08-03)

Granted by Prof. Yaqub + Ahmed Aly on **2026-08-03 for ~3 weeks** (≈ through Aug 24) to close the
classification gap before the validation board closes 08-20 / paper 08-31. This is the box that fixes
the 07-28 blocker: **2048² QLoRA needs ~28 GB and OOMs on the 24 GB A5000s of `ws`** — here we have 96 GB.
See memory [[project_blackwell_training_box]]. **Use: flagship 2048² training only. Eval + Docker stay on `ws`.**

### B.1 Access (SHARED account; local ed25519 key verified 2026-09-17)
- **Default from this machine: `ssh blackwell`**, in PowerShell or Git Bash.
  Taha installed the local ed25519 public key for `ugrip2026`; Codex verified
  `ssh -o BatchMode=yes -o ConnectTimeout=20 blackwell hostname` returned `mbz-yaqub`
  without a password. The local `~/.ssh/config` alias selects `10.67.33.33`,
  user `ugrip2026`, and `~/.ssh/id_ed25519`. The shared password is unchanged.
- Read-only GPU check:
  ```powershell
  ssh -o BatchMode=yes -o ConnectTimeout=20 blackwell 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv -i 0'
  ```
  On 2026-09-17, GPU 0 had 49,823 / 97,887 MiB used and 99% utilization.
  This is a snapshot, not an availability guarantee; recheck before any launch.
- Host **`10.67.33.33`** (hostname `mbz-yaqub`), user **`ugrip2026`** (shared with other researchers),
  password (not stored in this repo; ask Taha) — ⚠️ **do NOT change the password** (Ahmed's instruction; shared account).
- **Fallback only when the key is unavailable:** OpenSSH refuses a piped password,
  so connect via the `SSH_ASKPASS` trick from a LOCAL Bash/Git-Bash shell
  (the box is NOT reachable from `ws` — different subnet; `flare` hostname also doesn't resolve):
  ```bash
  cat > /tmp/askpass.sh <<'A'
  #!/bin/bash
  echo "$UGRIP2026_PW"   # export the shared password yourself; never commit it
  A
  chmod +x /tmp/askpass.sh
  SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
    ssh -o StrictHostKeyChecking=accept-new ugrip2026@10.67.33.33 '<cmd>' < /dev/null
  ```
  `scp` takes the same env prefix. Long jobs: `nohup … &` detached (VPN-flaky), collect later.

### B.2 Hardware / allocation
- **8× NVIDIA RTX PRO 6000 Blackwell Server Edition, ~96 GB each** (97,887 MiB), driver 590.48.01 /
  CUDA 13.1, compute capability **sm_120 (Blackwell)**.
- **★ WE ARE ALLOCATED GPU 0 ONLY** (Ahmed's rule; GPUs 1–7 belong to other researchers). **Always
  `CUDA_VISIBLE_DEVICES=0`.** Never launch on 1–7. No SLURM. **No Docker** on this box.

### B.3 Storage (HOME is FULL — everything under `/data`)
| Path | Free | Use |
|---|---|---|
| **`/data/ugrip2026`** | 25 TB | ✅ our home — conda, models, dataset, repo all here |
| `/home` | **12 GB, 100% FULL** | install NOTHING here |

### B.4 Env `flare-blackwell` (sm_120 needs a cu128 stack — the pinned cu121 recipe will NOT run)
Miniconda at `/data/ugrip2026/miniconda3`; env `flare-blackwell` (py3.11). Activate:
`source /data/ugrip2026/miniconda3/etc/profile.d/conda.sh && conda activate flare-blackwell`.
- **Heavy pins:** **torch 2.11.0+cu128** (`pip install torch torchvision --index-url
  https://download.pytorch.org/whl/cu128`), transformers **4.57.1**, peft **0.15.2**, **bitsandbytes 0.50.0**
  (NOT the recipe's 0.46.0 — that predates Blackwell), accelerate **1.11.0**, trl **0.9.6**, datasets **3.6.0**.
  Both compat gates PASS (GPU 4096² matmul kernel; bnb `Linear4bit` nf4 forward).
- **LLaMA-Factory:** the repo's **vendored** `llama_factory_internvl/` (v0.9.4.dev0, has Qwen3-VL), installed
  `pip install -e . --no-deps`.
- **★ `--no-deps` MISSING-DEP LIST (install these or `llamafactory-cli train` crashes at import/startup):**
  `rich`, `tyro`, then LF's mandatory runtime deps `matplotlib pandas scipy einops sentencepiece tiktoken
  fire av` (+ `omegaconf==2.3.1 antlr4-python3-runtime==4.9.3`). Install the small ones normally (they don't
  touch the heavy pins). ⚠️ pip prints stale LF metadata caps (`numpy<2`, `safetensors<=0.5.3`, `tyro<0.9`) —
  **these are warnings only; imports + training succeed** with numpy 2.4.4 / safetensors 0.8.0 / tyro 1.x.

### B.5 Staged assets (all downloaded from HF/GitHub over the box's internet, NOT from `ws`)
- **Base model:** `/data/ugrip2026/models/Qwen3-VL-8B-Instruct` (from HF `Qwen/Qwen3-VL-8B-Instruct`).
- **Repo:** `/data/ugrip2026/MICCAI-FLARE-2026-Task-3` on branch **`Taha/blackwell-box`** (relayed via git
  bundle local→box; the box can't `git pull` the private repo). The launcher is portable — see B.6.
- **Frozen splits:** `/data/ugrip2026/splits/internal_splits_v2` — a COPY relayed from `ws`, image paths
  `sed`-remapped `/share/sda/ugrip/datasets → /data/ugrip2026/datasets` (frozen originals on `ws` untouched).
- **Router:** NOT here (inference-only; eval runs on `ws`).
- **★ DATASET — needs UNZIPPING after the HF download.** `huggingface-cli download FLARE-MedFM/FLARE-MLLM-2D
  --repo-type dataset` (gated → needs an HF token; use it inline as a one-shot `HF_TOKEN=` env var, do NOT
  `huggingface-cli login` on the shared box). ⚠️ Use `HF_HUB_DISABLE_XET=1 --max-workers 4` or it 429-rate-limits
  and crashes ~12 GB in. **The repo ships per-dataset `imagesTr.zip` / `imagesVal.zip` (11 zips) that must be
  extracted in place** (`unzip -o -q <zip> -d <its dir>` → yields `imagesTr/<file>` matching the split paths).
  After extraction, **all 17 USED datasets resolve 100%**; the only unresolved paths are the two UNUSED
  datasets `Microscopy/chromosome` + `Endoscopy/endo` (endo is `--exclude_dataset`'d from training anyway) —
  gate a resolve-check on the DECK's images, not raw `A.jsonl`.

### B.6 Running a training job (portable launcher `scripts/train_qwen3_final.sh`)
From `.../llama_factory_internvl`, set these env overrides (the launcher defaults to `ws` paths):
```bash
export SPLITS=/data/ugrip2026/splits/internal_splits_v2 \
       CONDA_SH=/data/ugrip2026/miniconda3/etc/profile.d/conda.sh \
       QWEN3_ENV=flare-blackwell \
       QWEN3_DIR=/data/ugrip2026/models/Qwen3-VL-8B-Instruct \
       LF_DIR=/data/ugrip2026/MICCAI-FLARE-2026-Task-3/llama_factory_internvl \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cls winner recipe (stats/21): r32/α64 @ 2048². ML winner = r64/α128.
CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_PIXELS=4194304 LORA_RANK=32 LORA_ALPHA=64 \
  nohup bash scripts/train_qwen3_final.sh cls_bare 1337 > logs/<name>.log 2>&1 &
```
`LORA_RANK`/`LORA_ALPHA` are new env knobs (empty ⇒ YAML default r8/α16). `SPLITS`/`CONDA_SH`/`QWEN3_ENV`
are env-overridable (committed on `Taha/blackwell-box`).

### B.7 Measured performance (2048² cls smoke, r32/α64, GPU 0, 2026-08-03)
- **Peak GPU mem ≈ 28.2 GB** (28,206 MiB) at bsz1/ga8 — confirms *why* it OOM'd on the 24 GB A5000; here
  ~68 GB headroom. ⇒ we can **restore the deployed recipe's true `bsz 2 / grad_accum 4`** (the A100 original;
  A5000 forced us to bsz1/ga8) for a closer numerics match, and there's room for bf16 weights if ever wanted.
- **Speed ≈ 9.33 s/it @ 2048²** vs the A5000's ~38.5 s/it ⇒ **~4.1× faster.** The all-data cls retrain
  (~13,950 steps) drops from **~6 days (A5000) → ~1.5 days** here.
- ⚠️ **LF preprocessing (tokenize + image-process the full deck) is the slow stage at `preprocessing_num_workers=1`**
  (~50 min for 25,911 rows; rate collapses on the large 2048² images). It's cached per deck, but **bump
  `preprocessing_num_workers` (box has plenty of cores) for real runs.**
