# Method digest: the 11 papers in this folder

Written on 2026-10-09 by a reading agent working for Taha. It read all 11 PDFs (191 pages):
method, experiment and appendix pages as images, plus a text pass to search for URLs and
hyperparameters. ME-VLIP and QwenVLConnector were read in full.

**How to read it:**
- "p." numbers are **PDF page indices**. SMoES also has printed proceedings page numbers
  (24330–24337), which differ.
- "NS" = not stated in the PDF.
- "[ext]" marks a venue or year that came from outside the PDF.
- **Check the PDF before quoting a number in the paper.** How each method is adapted to our
  setup is in PLAN.md §1.3 and §4.

## Three facts that shape our implementation

1. **No paper tests its MoE-LoRA method on a 4-bit base or with HF PEFT.** All of them need
   custom modules. QLoRA appears only in ME-VLIP (plain LoRA) and QwenVLConnector.
2. **SMoES's "modality" means vision tokens vs text tokens, not imaging modality.** Its
   modality scores never feed the router; they only enter an auxiliary loss. For
   X-ray/US-style modality experts, Med-MoE's supervised modality router is the closer
   template.
3. **MedQwen (Sparse Spectral LoRA) subtracts a residual from the base weight W0 at
   initialisation.** That can't be done cleanly on NF4 weights. Our workaround (PLAN.md S6)
   keeps the residual as a frozen low-rank negative correction instead.

---

## 1. HydraLoRA

**Citation.** "HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning".
Chunlin Tian, Zhan Shi, Zhijiang Guo, Li Li, Chengzhong Xu (Univ. Macau / UT Austin /
Cambridge). arXiv:2404.19245v2, 23 May 2024, marked "Preprint. Under review" (NeurIPS 2024
[ext]). Key: `tian2024hydralora`.

**Method.**
- The authors' analysis (Obs. I/II, p.3; Table 1; Fig. 3 t-SNE) finds that, at a fixed
  parameter count, several small LoRA heads beat one big LoRA. Trained A matrices converge
  across heads, while B matrices diverge.
- So they share one A and use N separate B heads (p.4, Eq. 2):
  `W = W0 + Σ_{i=1}^N ω_i B_i A`, with `B_i ∈ R^{d×r}` and shared `A ∈ R^{r×k}`.
- Forward pass (p.5, Eq. 3): `y = W0 x + Σ_i ω_i E_i A x`.
- Router (Eq. 4): `ω = softmax(W_g^T x)`, with `W_g ∈ R^{r×N}`. It is "a dense layer with
  trainable weights followed by a softmax", taking "an intermediate token representation x".
  - The dimension is inconsistent: r×N implies the router sees the r-dim projection Ax,
    while the text says x. Pick one and document it.
- The baseline "LoRA MoE" in App. A.2 (p.16, Eqs. 5–6) uses `W_g ∈ R^{d_m×n}` on x with a
  softmax.
- Routing is dense: all B heads are weighted; there is no top-k.
- N is chosen by k-means on TF-IDF features of the training corpus with the elbow method
  (App. B, p.16–17, Eqs. 7–8). This only sets N; routing is learned end to end without
  labels.

**Hyperparameters.**
- Base: LLaMA2-7B (13B also in Table 3). Target modules: q_proj and v_proj (Fig. 3
  caption, p.4; App. C Fig. 9).
- Single-domain runs (Table 2, p.6): "HydraLoRA(r=8)", #A = 1, mean #B = 3, 0.124% params.
  That is the same budget as LoRA r=16 ("only employs rank=16 training overhead", p.8). The
  LoRA-Split baseline is 4×8, budget-matched to LoRA r=32 (0.248%).
- Multi-task Flan-v2 → BBH (Table 3, p.7; Table 4, p.18): 1 A / 10 B, 0.341% params.
  Baselines: LoRA-MoE with 48 A/B pairs (2.976%); LoraHub with 20 LoRAs (1.240%).
- k sweep 1–5: k=4 best (47.22 MMLU); k=3 only −0.42%; "not a sensitive parameter" (p.8,
  Fig. 8).
- No load-balancing or auxiliary loss is mentioned.
- Single stage: LLM frozen; A, the B_i and the router trained jointly.
- Learning rate, epochs, batch size, alpha, dropout: NS.
- Integral to the method: shared A, multiple B, softmax router. Incidental: the k-means
  choice of N and the q/v targets.

**Evaluation and analyses.**
- Benchmarks: MMLU (general, medical, law subsets), GSM8K, HumanEval, BBH 3-shot.
- Baselines: full fine-tuning, Prompt Tuning, P-Tuning, Prefix, IA3, AdaLoRA, LoRA
  r=8/16/32, LoRA-Split, LoraHub, LoRA-MoE.
- Ablations (Fig. 6, p.8): without MoE (= LoRA), without the gate (uniform ω), without
  Hydra (independent LoRAs).
- Also: t-SNE of A vs B parameters per subtask (replicable on our 6 task families), and
  robustness to the cluster count (Fig. 7). No routing heatmaps.

**Compute.**
- 4×A40. Versus LoRA r=32: 1.96× faster training and 49.6% less energy (Fig. 5, p.8).
- Limitations (p.19): needs "1 to 2 times more training iterations" than typical PEFT.
- Inference "compute[s] a weighted average of the experts [B] … then appl[ies] PEFT
  transformation" (§3.2.2, p.5). Cost is about one LoRA, but ω depends on the input, so it
  cannot be folded into W0.
- 4-bit, PEFT and LLaMA-Factory support: NS. Code URL: NS in this PDF.

**Gotchas.**
- Routing is per token, so it is KV-cache safe.
- Needs a custom wrapper around bnb `Linear4bit`: `base(x) + scaling · Σ ω_i B_i(A(x))`.
- Decide whether the router input is x or Ax.
- There is no balancing loss, so the router can collapse onto a few heads. Log routing
  entropy.

---

## 2. LoRAMoE

**Citation.** "LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via
MoE-Style Plugin". Shihan Dou, Enyu Zhou, Yan Liu, … Tao Gui, Qi Zhang, Xuanjing Huang
(Fudan / Hikvision). The arXiv ID is not printed; the MedQwen reference list gives
arXiv:2312.09979. ACL 2024 [ext]. Code: https://github.com/Ablustrund/LoRAMoE (p.1
footnote). Key: `dou2024loramoe`.

**Method.**
- The FFN linear layers are replaced by a plugin of N LoRA experts plus a router; the
  backbone is frozen (Fig. 4, p.4).
- Eq. 3: `o = W0 x + Σ_i G(x)_i E_i(x)`, with `G(x) = Softmax(x W_g)`.
- Eq. 5 (p.5): `o = W0 x + (α/r) Σ_i ω_i B_i A_i x`. Routing is dense, per token, softmax
  over all experts.
- **Localized Balancing Constraint** (p.5–6):
  - importance `Q_{n,m} = Σ_{j=1}^{T_m} exp(ω_n^j/τ) / Σ_k exp(ω_k^j/τ)`: the sum of routing
    values over the tokens of sample m (Eq. 6);
  - coefficient `I_{n,m} = 1+δ` if Type_e(n) = Type_s(m), otherwise `1−δ` (Eq. 7);
  - `L_lbc = σ²(Z)/μ(Z)` with `Z = I ∘ Q` (Eq. 8);
  - total `L = L_LM + β L_lbc` (Eq. 9).
- Experts are split into two groups (world-knowledge vs other downstream tasks). Each
  sample carries a task-type label during training only; at inference the router weights
  all experts, so no labels are needed (p.6).

**Hyperparameters** (§4.1, p.6).
- LLaMA-2-7B; only FFN linear layers replaced (which of gate/up/down: NS).
- 6 experts (3+3), r=4, α=32, dropout 0.05, lr 2e-4, batch 16 per node, β=0.1, δ=0.1.
  Epochs: NS. A 3M-sample SFT mix (Table 4, p.11).
- Only experts and router are trainable. Single stage.
- Sensitivity (Table 3, p.7):

  | Configuration | Params | Score |
  |---|---:|---:|
  | 6 experts, r=4 (default) | 0.57% | 58.21 |
  | 4 experts | | 55.84 |
  | 8 experts | | 56.58 |
  | r=8 | | 58.11 |
  | r=16 | | 58.86 |

  Six experts is best; more rank gives little.
- Integral: LoRA experts plus a softmax router. The balancing loss only matters with a
  task-type split; for us the natural analogue is a task-family (or modality) label per
  sample.

**Evaluation.**
- Compares baseline, SFT, single LoRA, and LoRAMoE with and without L_lbc (Table 2, p.7).
- Router-weight visualisation per dataset (Fig. 7, p.8): the fraction of router mass on
  each expert group, per task. This is exactly a task-to-expert probability plot.
- Coefficient of variation of expert usage over training without the constraint (Fig. 5,
  p.5): it climbs to about 3, i.e. the router collapses.

**Compute.** 0.57% trainable params. Training time, 4-bit and PEFT support: NS. Not
mergeable (dense, input-dependent routing).

**Gotchas.**
- Per-token routing is KV-cache safe.
- L_lbc needs per-sample token sums of router probabilities, so mask padding, and decide
  whether image tokens count.
- It needs a type label per sample; our task family provides one for free.
- FFN only in the paper; we wrap the common target set (PLAN.md §1.2).

---

## 3. Mixture of LoRA Experts (MoLE)

**Citation.** "Mixture of LoRA Experts". Xun Wu, Shaohan Huang, Furu Wei (Microsoft
Research Asia / Tsinghua). ICLR 2024. arXiv:2404.13628v1. Code:
https://github.com/yushuiwx/MoLE.git (p.1). Key: `wu2024mole`.

**Method.**
- Post-hoc composition of N already-trained, frozen LoRAs. Each layer or block gets a
  learnable gate; the base model and the LoRAs stay frozen and only the gates train (p.7).
- Each LoRA's block output is computed separately: `E_{Δθ_i}(x) ∈ R^{L×d}` (Eqs. 7–8).
- Gate:
  - concatenate and normalise:
    `E_Ω = Normalization(E_{Δθ_0} ⊕ … ⊕ E_{Δθ_{N−1}}) ∈ R^{N×L×d}` (Eq. 9);
  - score: `ε = Flatten(E_Ω)^T · e`, with `e ∈ R^{ξ×N}` and `ξ = N·L·d` (Eq. 10);
  - weights: `G(ε_i) = exp(ε_i/τ) / Σ_j exp(ε_j/τ)`, with learnable τ (Eq. 11);
  - output: `O(x) = F_θ(x) + Σ_i G_i(ε_i) E_{Δθ_i}(x)` (Eqs. 12–13).
- Balance loss (Eqs. 14–15, p.6): `L_balance = −log Π_i q^{(i)}`, where
  `q^{(i)} = (1/M) Σ_{k=1}^M softmax(ε^k/τ)_i` is averaged over the M gated blocks.
- Total: `L = L_D + α L_balance` (Eq. 16).

**Hyperparameters.**

| Domain | Setup | LR | Iterations | Batch | α |
|---|---|---|---:|---:|---:|
| Vision + language | DreamBooth / SD2.1, 3–6 LoRAs | 1e-5 | 400 | 2 | 0.5 |
| NLP | Flan-T5, LoRAs trained on FLAN tasks | 1e-5 | 800 | 12 | 0.5 |

- Number of LoRAs tested: 8, 24, 48, 128 (Table 6, p.12); performance drops at 128.
- No top-k; all LoRAs are weighted.
- Gate granularity (Table 9, p.13): matrix-wise, layer-wise, block-wise (best),
  network-wise (worst).
- Without L_balance: 77.57; raising τ instead also hurts (Table 7, p.12).
- The rank, alpha and target modules belong to the pre-trained LoRAs: NS.

**Evaluation.**
- Baselines: normalized linear arithmetic (NLA, `Ŵ = W + Σ w_i ΔW_i` with Σw = 1; Eq. 2),
  LoraHub, PEMs, SVDiff.
- Replicable analyses: the gate-entropy curve over training (Fig. 5a), per-gate weight bars
  per layer (Fig. 7, p.13), and an expert-masking inference mode (Fig. 8, p.14).

**Compute.** Only gate parameters train, so training is cheap. Inference runs all N LoRAs
plus the gates, i.e. N× LoRA compute. Not mergeable unless the weights are frozen to
constants (that is NLA). 4-bit and PEFT support: NS.

**Gotchas.**
- The gate flattens over the sequence length L, so `e` depends on a fixed L. It is a
  sequence-level gate that changes as tokens are generated. In practice, compute it at
  prefill and cache it for decoding, or replace Flatten with a mean pool. Either is a
  documented deviation.
- **Our adaptation (PLAN.md S4b):** the gate reads the hidden state, as Tausifa specified,
  not the LoRA outputs. It is mean-pooled at prefill and cached; α=0.5; the learning rate is
  the common one rather than MoLE's 1e-5.

---

## 4. Med-MoE

**Citation.** "Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical
Vision-Language Models". Songtao Jiang, Tuo Zheng, Yan Zhang, Yeying Jin, Li Yuan, Zuozhu
Liu (Zhejiang U / NUS / PKU). arXiv:2404.10237v3 (1 Sep 2024). EMNLP Findings 2024 [ext].
Code: https://github.com/jiangsongtao/Med-MoE. Key: `jiang2024medmoe`.

**Method: three phases** (Fig. 2, p.3; Eqs. 1–5, p.3–4).
1. **Alignment:** train only the MLP projector on image–caption pairs (Eq. 1).
2. **Instruction tuning plus router:**
   - full instruction tuning with the vision encoder frozen (Eq. 2);
   - a router trained on a small labelled subset to predict imaging modality (CT, MRI,
     pathology, X-ray) with cross-entropy: `L_Router = −Σ y_i log p(y_i | T_comb)` (Eq. 3).
3. **Domain-specific MoE tuning:**
   - the FFN is replaced by K experts plus an always-active meta-expert:
     `O_MoE = Σ_{i=1}^K G_i E_i + E_meta` (Eq. 4);
   - experts and meta-expert are initialised from the phase-2 FFN;
   - the router is frozen and only the experts train (Eq. 5).

**Hyperparameters.**
- 4 domain experts, top-2, plus 1 meta-expert (Tables 9–10, p.7). MoE in 16 of 32 layers
  (Table 13, p.13).
- Experts are full FFN copies, not LoRA. Base models: Phi-2 2.7B (3.6B activated / 5.3B
  total) and StableLM-1.6B (2.0B / 2.9B).
- Router: an MLP. 1–2 layers is best and 3 overfits (Fig. 3, p.7). About 1,000
  modality-labelled samples give 98.1% router accuracy; 50 give 82.3% (Fig. 6, p.12).
- Table 11 (p.13):

  | | Stage I | Stage II | Stage III |
  |---|---|---|---|
  | Epochs | 1 | 3 | 9 |
  | Learning rate | 1e-3 | 2e-5 | 2e-5 |
  | Data | 500K | 60K | 44K |

  Also: cosine schedule, weight decay 0, batch 2/GPU, 8× RTX 3090, bf16.
- "Med-LoRAMoE" variant (Table 7): LoRA rank 128/256 on the experts.
- No load-balancing loss; the router is supervised and then frozen. Its input is T_comb
  (the image plus instruction tokens).

**Evaluation and ablations.**
- Meta-expert on/off (Table 5): +1.3 to +4.2.
- Supervised router vs generic learned router (Table 6): +0.5 to +2.3.
- Top-1 vs top-2 (Table 9); 2/4/6 experts (Table 10).
- Router-embedding PCA silhouette: 0.173 before the router vs 0.579 after (Fig. 10).
- Top-1 expert per layer per modality (Fig. 5); per-expert text/image token share per
  layer (Figs. 4, 11).

**Compute.** Training time: 2 experts 6 h, 4 experts 8 h, 6 experts 11 h. Not mergeable.
4-bit: NS.

**For us:** the template for the modality-aware arm (PLAN.md S5a, Med-MoE-style): a
supervised (here, oracle) modality route, an always-on shared LoRA, and routed modality
LoRA experts. With a sequence-level label, route once at prefill.

---

## 5. Soft Modality-Guided Expert Specialization (SMoES)

**Citation.** "Soft Modality-Guided Expert Specialization in MoE-VLMs". Zi-Hao Bo, Yaqian
Li, Anzhou Hou, Rinyoichi Takezoe, Ertao Zhao, Tianxiang Pan, Jiale Yan, Mo Guang, Kaiwen
Long (Li Auto). CVPR Open Access version; the year is not printed (references run to Aug
2025, so likely CVPR 2026). arXiv: NS. Code: NS. Key: `bo2026smoes`.

**How the modality guidance works.**
- It is **not** router conditioning and **not** hard labels. "Modality" means vision tokens
  vs text tokens. The router stays a standard `g = softmax(W_gate x)` with top-k (§3.1,
  p.3).
- Soft modality scores `M ∈ [0,1]` (summing to 1 over {text, vision}) are estimated per
  token and per layer (p.4), in one of two ways:
  - **attention-accumulated** (Eqs. 2–4): start from the 0/1 modality ID, propagate through
    head-averaged attention, and mix by residual norms;
  - **Gaussian statistics** (Eqs. 5–9): a per-layer diagonal Gaussian per modality, updated
    by EMA (β=0.99); per-token log-likelihood; temperature softmax with τ = 0.5·D.
- **Expert binning** (Eqs. 10–11): experts are sorted by their EMA text bias
  `f_spec(e) = C̄_text,e / (C̄_text,e + C̄_vision,e)` and split into N_bins consecutive groups.
- **Inter-bin mutual-information loss** (Eqs. 12–15, p.5): it maximises I(modality; bin)
  through `L_MI = −Σ_l (1/N_batch) Σ_i I_i(M; B)`.
- **Bin-level load balancing** (Eq. 16).
- **Total** (Eq. 17): `L = L_task + α_bal L_bal + α_MI L_MI`.

**Hyperparameters.**
- N_bins = 8, τ = 0.5·D, EMA β = 0.99, α_bal = 0.001, α_MI = 0.0001, on 8×A800.
- Backbones are native fine-grained MoE LLMs (DeepSeekMoE top-6 of 64, OLMoE top-8 of 64,
  Moonlight-MoE, Qwen3-MoE), with a CLIP ViT-L/14 encoder and LLaVA two-stage training.
  The experts are full FFNs, not LoRA.
- LR, epochs, batch: NS (they are in the supplement, which is not included).

**Evaluation.**
- Baselines: no specialisation, hard routing, MoIIE, SMAR.
- **Modality Specialization Index** (Eqs. 18–19):
  `MSI = (1/L) Σ_l (1/E) Σ_e 2·|C̃_text,e − 0.5|`, in [0, 1].
- Key finding: hard modality routing costs −3.9 to −6.1% on multimodal tasks and −26% on
  language-only tasks (Table 1).
- Visualisations: token-to-expert histograms, specialisation over training, and modality
  scores across layers.

**For us:** our bases are dense, so this applies to the LoRA experts we add. Re-defining M
as the per-sample imaging modality, with an MI(modality; expert bin) loss at α ≈ 1e-4, is
**our adaptation** (PLAN.md S6, "soft modality-guided"). The attention-accumulated score
needs attention maps, which conflicts with flash/SDPA attention.

---

## 6. Sparse Spectral LoRA (MedQwen)

**Citation.** "Sparse Spectral LoRA: Routed Experts for Medical VLMs". Omid Nejati Manzari,
Hojat Asgariandehkordi, Taha Koleilat, Yiming Xiao, Hassan Rivaz (Concordia).
arXiv:2604.01310v1 (1 Apr 2026). Project page: https://omid-nejati.github.io/MedQwen/.
Key: `manzari2026medqwen`.

**Method** (p.2–5; Algorithm 1, p.13).
- **Spectral experts.** Take `W(0) = U S V^T`, h = min(m, n), N experts, total LoRA rank r,
  per-expert rank d = r/N, stride t = h/N.
  - Expert j takes the d consecutive singular components starting at k = (j−1)·t (Eq. 13).
    The slices are spread across the whole spectrum and don't overlap. The total rank r is
    split across experts, not multiplied.
  - Init (Eq. 22): `B_i(0) = sqrt(1/(sρ)) U_i S_i^{1/2}`,
    `A_i(0) = sqrt(1/(sρ)) S_i^{1/2} V_i^T`.
- **Routing:** Mixtral-style top-k (Eqs. 10–12): `z = W_z x`, softmax renormalised over the
  top-k set.
- **Output:** `MoE(x) = W̃(0) x + Σ_{i∈S_k} R(x)_i s B_i A_i x`.
- **Residual compensation**, so the initial effective weight equals W(0) (Eqs. 17–21):
  `W_res = (s/N) Σ_i B_i(0) A_i(0)` (using E[R] = 1/N), then `W̃(0) = W(0) − W_res`.
- **Scaling** (Thm 5, Eq. 24): `s* = sqrt(3nη/r)`, with η = lr_FFT / lr_LoRA. It replaces
  α/r.
- Variant MedQwen-e (p.15): a per-expert scale `s_i = s_1 sqrt(σ_0/σ_i)`, with comparable
  results.

**Hyperparameters.**
- N = 8, top-2 (Fig. 7, p.8: "2-of-8 offers a balanced trade-off"), total rank 32, so d = 4
  per expert.
- Routing alternatives (Table S5): top-2 68.24 > top-p(0.25) 66.40 > top-k plus a shared
  expert 65.67.
- ρ = 10; η = 0.1 for Qwen (lr ≈ 1e-4); η = 1 for ViT (§8.2).
- Balance loss coefficient 1e-3: `L_b = Σ_i f_i P_i` (Eqs. 25–27, p.19).
- VQA (Table S3): batch 16, rank 32, α 64, AdamW, warmup 100 steps, dropout 0.05,
  lr 1e-4.
- Base: Qwen2.5-VL-7B. Targets: q, k, v, up, down. Three stages (alignment 500K, instruct
  60K, MoE tuning 2.4M). Epochs: NS.
- Integral: the SVD-segment initialisation, W_res subtraction, the s* scaling, and top-k
  routing.

**Evaluation.**
- Baselines: LoRA r=8/16/32, MiLoRA, PiSSA, MoELoRA, AdaMoLE, HydraLoRA (Tables 4, 6).
- Budget-matched vs MoELoRA/HydraLoRA (Table 6): params 0.96 / 0.84 / 0.96%.
- Initialisation ablation, principal / minor / random / theirs (Table 5). This is the
  basis for our random-init control arm.
- Experts × top-k sweep (Fig. 7); rank sweep 8–128 (Fig. 6b).
- Per-dataset expert load ≈ 0.125 ± 10% (Fig. S1), i.e. no collapse.
- Forgetting under sequential fine-tuning: LoRA −50%, MoELoRA −20%, theirs −5%.

**Compute.** Memory 32.56 GB; epoch time 29h19m vs 29h15m (MoELoRA) and 29h17m (HydraLoRA)
on one A100, so the overhead versus other MoE-LoRA is negligible. Not mergeable. 4-bit: NS.

**Gotchas.**
- The SVD must run on the full-precision W0 (dequantise each NF4 layer once).
- `W̃(0) = W0 − W_res` edits the base weights. On a 4-bit base, keep W_res as a frozen
  low-rank correction (rank ≤ r) applied with a minus sign, rather than re-quantising.
- The scale s replaces PEFT's α/r.

---

## 7. Generalist vs Specialist VLMs (MedVLMBench)

**Citation.** "Can Generalist Vision Language Models (VLMs) Rival Specialist Medical VLMs?
Benchmarking and Strategic Insights". Yuan Zhong, Ruinan Jin, Qi Dou, Xiaoxiao Li (CUHK /
UBC / Vector). arXiv:2506.17337v4 (29 Mar 2026). Code:
https://github.com/ubc-tea/MedVLMBench. Key: `zhong2025medvlmbench`.

**Setup.**
- 18 VLMs in paired generalist/specialist families, including Gemma3 vs **MedGemma** and
  Qwen2/2.5-VL vs Lingshu.
- 6 diagnosis and 4 VQA datasets.
- Three research questions:
  - RQ1: off-the-shelf generalist vs off-the-shelf specialist;
  - RQ2: fine-tuned generalist vs off-the-shelf specialist;
  - RQ3: fine-tuned generalist vs fine-tuned specialist, out of distribution.
- Generative VLMs are fine-tuned per dataset: the MLP bridge plus LoRA on the decoder
  (§2.4).
- Statistics: 1,000-iteration bootstrap CIs and a linear mixed model.

**Findings relevant to us.**
- There is **no shared multi-task vs task-specific comparison and no data-scaling study**;
  every fine-tune is single-dataset.
- Specialists win off the shelf. After light adaptation, 9 of 12 tuned generalist diagnosis
  baselines and 10 of 15 VQA baselines beat the best off-the-shelf specialist.
- Out of distribution, 8 of 18 tuned generalists beat the best tuned specialist.
- **Gemma family** (Fig. 3b, p.9), GPT score off-the-shelf → after SFT:

  | Model | VQA-RAD | PathVQA | SLAKE | FairVLMed |
  |---|---|---|---|---|
  | MedGemma | 68.7 → 69.0 | 59.3 → 72.8 | 76.3 → 88.5 | 56.8 → 69.2 |
  | Gemma3 | 56.7 → 55.9 | 56.9 → 65.6 | 68.0 → 80.1 | 57.7 → 69.5 |
  | Qwen2.5-VL (reference) | 66.9 → 71.2 | 51.6 → 71.3 | 69.1 → 86.3 | 54.9 → 72.6 |

  In the Gemma family the specialist edge survived tuning, unlike in the LLaVA/Qwen
  families. This is the literature hook for S10.

---

## 8. Vision-Language Model for Multitask Medical Text Generation

**Citation.** Hongkun Sun, Lichen Xue, Lushuai Jia (Tomorrow Medical Network Technology,
Hangzhou). FLARE 2025 Task 5 challenge paper (MICCAI 2025). Key: `sun2025multitaskmedgen`.

**Method.**
- Frozen MedSigLIP encoder at 448².
- A linear projection that first merges 4 adjacent visual tokens.
- LLaMA2-Chat-7B with a **single shared LoRA** for all tasks, plus explicit task
  identifiers in the prompt (`[classification]`, `[detection]`, …).
- 10 epochs, batch 2, lr 1e-4, cosine schedule; one RTX 3090 for about 4 days; 52.4 M
  trainable params.
- Inference samples with top_p 0.9 and temperature 1 (not greedy).

**Relevance.**
- No shared-vs-task-specific ablation and no scaling study.
- The only interference evidence is anecdotal: they excluded instance detection because
  "it may introduce gradient interference".
- Hidden-validation results: classification 0.70, multi-label 0.54, detection 0.80,
  regression MAE 13.63.

---

## 9. Understanding Task Transfer in Vision-Language Models

**Citation.** Bhuvan Sachdeva, Karan Uppal, Abhinav Java, Vineeth N. Balasubramanian
(Microsoft Research India). arXiv:2511.18787v2 (9 Apr 2026). Project page:
https://aka.ms/task-transfer-vlms. Key: `sachdeva2025tasktransfer`.

**Metric definitions (used in our S2).**
- **Perfection Gap Factor** (Eq. 3, p.4; Algorithm 1, p.11):
  `μ_{i→j} = (Acc(M(T_i), T_j) − Acc(M, T_j)) / (U_j − Acc(M, T_j) + ε)`, ε = 1e-6, default
  ceiling U_j = 100. A.10 (p.19) gives a variant with U_j = the best observed accuracy, which
  makes the diagonal 1.
- **Transferability** of source T_i (Eq. 1):
  `Δ(i)^+ = ((1 − e^{−p/N}) / p) Σ_j μ_{i→j} 1{μ_{i→j} > 0}`, with p = #{j : μ > 0}. Δ(i)^−
  is the same over negative entries.
- **Malleability** of target T_j (Eq. 2): Θ(j)^±, the same form summed over sources.
- Whether the diagonal is included in these sums: NS.
- Properties (p.8): PGF ≤ 1; the minimum is −(m−1); asymmetric; sensitive near the ceiling.
- **Task clique** (Def. 3): a subset in which every ordered pair has the same sign; tested
  for stability with a Wilcoxon test across seeds.
- **Personas** (p.6): Donor/Pirate (source above/below the average transferability) and
  Sponge/Sieve (target with high positive/negative malleability), tested with unpaired
  t-tests across seeds.
- Also: PGF-informed data mixtures (Fig. 8); transfer rises monotonically with training
  steps (A.6); LoRA-weight cosine similarity on the last-layer o_proj (A.7).

**Setup.** Qwen2.5-VL 3B/7B/32B; 13 BLINK multiple-choice tasks; LoRA r=8, α=16, batch 16,
lr 1e-4, cosine schedule, warmup 0.03; 4 seeds; GPT-4.1 for answer extraction; no 4-bit.

---

## 10. ME-VLIP (our prior baseline)

`shaaban2025mevlip`. Mai A. Shaaban, Amal Saqib, Shahad Hardan, Darya Taratynova, Tausifa
Jan Saleem, Mohammad Yaqub (MBZUAI). FLARE 2025 Task 5. Code:
https://github.com/BioMedIA-MBZUAI/ME-VLIP.
- InternVL3-8B with QLoRA via LLaMA-Factory: rank 8 on q, k, v, o, up, gate, down.
- 3 epochs, batch 2 × grad-acc 4, lr 2e-4 linear, 3% warmup. About 24 h on one A100;
  about 20 M trainable params.
- One task-specific QLoRA per task plus a multi-task QLoRA. A GLiClass task classifier
  routes to the task adapter when its confidence is ≥ 0.3, otherwise to the multi-task one.
  This is exactly our TaskLoRA hard-routing baseline.
- Multi-image samples are flattened into single-image records.
- Hidden-validation results: classification BA 0.74, multi-label 0.57, detection 0.82,
  regression MAE 11.84.

## 11. QwenVLConnector (our prior baseline)

`nguyen2025qwenvlconnector`. Le Thien Phuc Nguyen, Hoang-Thien Nguyen et al. (Ulas Bagci,
Northwestern). FLARE 2025. Code: https://github.com/plnguyen2908/QwenConnector.
- Qwen2.5-VL-7B with a dense multi-layer connector: depth-averaged ViT blocks, through the
  shared merger, then concatenated with the encoder output.
- Three stages; FLARE tuning with LoRA (32, 64) at lr 2e-4 for 3 epochs.
- Detection F1 0.55 → 0.85; classification 0.37 → 0.51; regression worse.

---

## Implementation notes for a 4-bit Qwen-family VLM

- **No method has a native PEFT implementation.** The common pattern is a custom
  `nn.Module` wrapping each bnb `Linear4bit`: `base(x) + scale · Σ_i w_i(x) B_i A_i x`.
  Keep the router in bf16 with an fp32 softmax, and keep router and LoRA weights out of
  4-bit.
- **Per-token routers** (HydraLoRA, LoRAMoE, MedQwen, SMoES) are KV-cache safe, since each
  decoding step routes only the new token.
- **Per-sequence routers** (MoLE, Med-MoE, ME-VLIP's classifier) should compute the route at
  prefill and cache it.
- **Mergeable into W0:** only static adapters (Shared LoRA, the selected TaskLoRA, and
  fixed-weight composition).
- Decide whether image tokens count in balance and MI statistics.
- Verify Qwen3.5-9B's actual projection names before setting targets (it has hybrid
  Gated-DeltaNet layers).

## Summary table

| Method | Experts | top-k | Rank per expert | Router input | Aux loss (coef.) | Stages | Mergeable | Code |
|---|---|---|---|---|---|---|---|---|
| HydraLoRA | 1 shared A + N B (3 single-domain, 10 multi-task; via k-means) | dense | "r=8" (budget = LoRA r16) | token x (ambiguous x vs Ax) | none | 1 | no | NS |
| LoRAMoE | 6 (3 + 3) | dense | r=4, α=32 | token hidden state, FFN | L_lbc, β=0.1, δ=0.1 | 1 | no | github.com/Ablustrund/LoRAMoE |
| MoLE | N pre-trained LoRAs | dense, learnable τ | inherited | LoRA block outputs, flattened over the sequence | L_balance, α=0.5 | 2 (gates only, lr 1e-5) | no | github.com/yushuiwx/MoLE |
| Med-MoE | 4 FFN + 1 meta | 2 | full FFN (LoRA variant r=128/256) | image + instruction tokens; supervised modality MLP | router CE | 3 | no | github.com/jiangsongtao/Med-MoE |
| SMoES | native MoE (64; 8 bins) | 6 or 8 | full FFN | token x (modality not fed to router) | bin balance 1e-3 + MI 1e-4 | LLaVA 2-stage | no | NS |
| Sparse Spectral (MedQwen) | 8 | 2 | r/N (32 → 4); s* = sqrt(3nη/r) | token x, linear | Σ f_i P_i, 1e-3 | 3 | no | project page |
| ME-VLIP | 1 per task + multi-task | hard top-1 | r=8 | question → GLiClass | none | per adapter | yes | github.com/BioMedIA-MBZUAI/ME-VLIP |
