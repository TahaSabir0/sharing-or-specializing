# Kickstart: from zero to your first smoke test

For Veronika, Alberto and Alex (and anyone joining). Do it once, top to bottom, and tick each
box. It takes about an hour. Stuck? Ask Taha in the team chat.

---

## 0. The 2-minute version of how we work

- **Code lives in this repo** (`TahaSabir0/sharing-or-specializing`). You work on your own
  branch, write code on your laptop, push to GitHub.
- **The GPU is one card**: GPU 1 of the Blackwell box `mbz-yaqub`, until 13 Nov. It runs the
  team's real training runs around the clock, 3 at a time.
- **You use the box only to test your code**, with one command (`smoke.sh`): a 50-step
  training run plus a tiny eval, ~20–40 minutes. One smoke test at a time for the whole team.
- **Real runs** (days of GPU) are launched only by Taha: open a pull request with your smoke
  result, Taha merges and queues it.

Why so strict: one card shared by four people, a shared login, and 3–4 weeks left. A clash or a
deleted run costs days we don't have.

---

## 1. Accounts and access

- [ ] **GitHub:** accept the collaborator invite to `TahaSabir0/sharing-or-specializing`.
- [ ] **SSH key on GitHub:** `ssh -T git@github.com` on your laptop answers
      "Hi <you>! You've successfully authenticated". If not:
      [add an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
- [ ] **MBZUAI VPN** connected (the box is on the internal network).
- [ ] **Box password:** Taha sends it to you by DM. Never write it in the repo, the chat, or a
      file on the box.

## 2. Get the repo on your laptop

```bash
git clone git@github.com:TahaSabir0/sharing-or-specializing.git
cd sharing-or-specializing
git checkout -b <your_name>/<topic>        # e.g. veronika/hydra, alberto/medgemma-port, alex/modality-decks
```

You don't need a GPU or the big Python stack locally: you write code here and test it on the box.

## 3. Read, in this order (~45 min)

1. `STATUS.md`: where the project is right now.
2. `PLAN.md`: the one-page summary at the top, §1 (the fixed protocol: every run follows it),
   **your experiments in §4**, and §7 (who does what, and the rules).
3. `code/INTERFACE.md`: how a method plugs in, run configs, smoke tests, asking for a run.
4. `code/README.md`: what is where in `code/`.
5. For your method: `references/papers/METHOD_DIGEST.md` and the paper in `references/papers/`.

`CLAUDE.md` holds the same rules written for AI coding assistants. If you use one (Claude Code,
Codex, ...) in this repo, it reads that file automatically.

## 4. One-time setup on the box

**4.1 SSH config on your laptop** (`~/.ssh/config`):

```
Host blackwell
  HostName 10.67.33.33
  User ugrip2026
  ForwardAgent yes
```

`ForwardAgent yes` lets the box use **your** GitHub key while you are logged in, so you can
`git pull` your branch there without leaving any credentials on the shared account.
On Windows, start the agent once: `Get-Service ssh-agent | Set-Service -StartupType Automatic;
Start-Service ssh-agent; ssh-add` (PowerShell as admin). On macOS/Linux: `ssh-add`.

**4.2 Log in and check:**

```bash
ssh blackwell                      # password from Taha
ssh -T git@github.com              # on the box: must greet YOU (agent forwarding works)
```

**4.3 Your own clone** (never work in `/data/ugrip2026/journal/repo`: the real runs use it):

```bash
mkdir -p /data/ugrip2026/journal/people/<your_name> && cd $_
git clone git@github.com:TahaSabir0/sharing-or-specializing.git repo
cd repo
git config user.name "<Your Name>"           # repo-local: the account is shared, so never --global
git config user.email "<your email>"
git checkout <your_name>/<topic>            # after you have pushed it from your laptop
```

**4.4 Environment:** `source /data/ugrip2026/miniconda3/etc/profile.d/conda.sh && conda activate flare-blackwell-v5`.
It is the frozen training stack: **never `pip install` into it.** If you need a package, ask Taha
(we make a separate env).

## 5. Your first smoke test (proves everything works)

Run the reference plug-in, which is known to pass:

```bash
cd /data/ugrip2026/journal/people/<your_name>/repo
bash code/train/smoke.sh <your_name> code/configs/smoke_lora_ref.yaml
```

- If another smoke test is running, it says whose: try again later.
- You should see, at the end: `loss: first ≈1.56 -> last ≈0.65 (decreasing)`, trainable params
  21,639,168, an eval line for all 6 tasks with `malformed_rate 0.0`, and a valid
  `routing_stats.npz`. (Task scores after 50 steps are meaningless: it's a format check.) Then
  you're set up.
- It takes ~20–40 min. To keep it running after you log out, use `tmux` (or `nohup ... &`).

## 6. Your daily loop

```
laptop: write code → commit → git push
box:    cd /data/ugrip2026/journal/people/<your_name>/repo && git pull
box:    bash code/train/smoke.sh <your_name> code/configs/v2/<your_config>.yaml
        → read the summary → fix → repeat
done:   open a pull request into main with the smoke summary → Taha queues the real runs
```

CPU-only work on the box (building decks, analysis scripts) is fine anytime, but cap threads:
`export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` first. The box has 512 cores and an uncapped job
once swamped it.

## 7. GPU and box rules

**Allowed:** `smoke.sh` (one at a time, it enforces that); CPU work in your own folder;
reading anything.

**Never:**
- launch anything on a GPU except `smoke.sh` (no `python ... train`, no notebooks on the GPU);
- use any GPU other than 1 (GPUs 0, 2–7 belong to other people);
- kill a process you did not start (`nvidia-smi` shows them: leave them alone);
- touch `/data/ugrip2026/journal/repo`, or anything in `/data/ugrip2026/journal/runs/` outside
  `runs/smoke/`, or `/data/ugrip2026/journal/data/` (shared decks: read only);
- install packages into shared conda envs;
- store passwords, tokens or keys on the box.

Something looks broken (the card is full, a run died, the lock is stuck)? Tell Taha; don't fix it
yourself.

## 8. Cheat sheet

| What | Where |
|---|---|
| Base models | `/data/ugrip2026/models/` (`Qwen3.5-9B`, `medgemma-1.5-4b-it`, `CRIMSON-medgemma-4b`) |
| Decks (journal data) | `/data/ugrip2026/journal/data/` (`j26_train_*.jsonl`, `j26_holdout.jsonl`, `j26_devselect.jsonl`) |
| Real runs (read only) | `/data/ugrip2026/journal/runs/<run_id>/` |
| Your smoke runs | `/data/ugrip2026/journal/runs/smoke/` |
| Your clone | `/data/ugrip2026/journal/people/<your_name>/repo` |
| Training env | `flare-blackwell-v5` |
| Results of record | `results/LEDGER.md` (numbers), `results/RUNS.csv` (every run) |
| GPU 1 status | `nvidia-smi -i 1` |

| Ask about | Person |
|---|---|
| access, the box, GPU, runs, eval, records | Taha |
| HydraLoRA | Veronika |
| MedGemma | Alberto |
| modality/dataset adapters, routing, representations, transfer matrix | Alex |
