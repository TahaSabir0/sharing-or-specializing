# Integrated Qwen/MedGemma Hydra smoke gates, 3 Nov 2026

Tested commit: `87e58a07f2f15f44af1f3b500dad162f9147d26c` on
`codex/veronika-hydra`, Blackwell GPU 1, frozen `flare-blackwell-v5` environment.
Evidence source: Veronika's copied terminal summaries, the queue's PASS
messages, and the downloaded archive, verified locally against all 40 manifest
SHA256 entries. Sanitized audit: `hydra_archive_audit_20261103.json`. The initial dirty-source Qwen checks and earlier timeout remain
recorded separately in `hydra_q35_20261103.md`.

| Backbone | Mode | Run suffix | First -> last loss | Stored/trainable | Active per example | s/step | Peak reserved GiB |
|---|---|---|---|---:|---:|---:|---:|
| Qwen3.5-9B | Hydra-T | `11032258` | 1.536 -> 0.646 | 78,716,928 | 21,639,168 | 13.853 | 17.89 |
| Qwen3.5-9B | Hydra-R | `11032312` | 1.613 -> 0.631 | 72,757,248 | 21,639,168 | 13.290 | 17.79 |
| MedGemma-1.5-4B | Hydra-T | `11032325` | 3.206 -> 0.722 | 55,287,808 | 14,901,248 | 5.200 | 8.45 |
| MedGemma-1.5-4B | Hydra-R | `11032333` | 3.485 -> 0.745 | 49,020,928 | 14,901,248 | 5.393 | 8.33 |

Full run IDs:

- `j26_q35_hydraT_full_768_s1337_smoke_veronika_11032258`
- `j26_q35_hydraR_full_768_s1337_smoke_veronika_11032312`
- `j26_mg15_hydraT_full_896_s1337_smoke_veronika_11032325`
- `j26_mg15_hydraR_full_896_s1337_smoke_veronika_11032333`

All four completed 50 optimizer steps with training/eval exit codes 0/0 and
decreasing loss. Each evaluated 10 questions per task, six tasks, 60 predictions,
on training-seen dev-select. All tasks had zero malformed/truncation/error counts.
Routing statistics validated as 60 x 248 x 6 (Qwen) and 60 x 238 x 6 (MedGemma).
The queue reported PASS for each run and ALL REQUESTED SMOKES PASSED. Its verifier
checks the runtime stored/active counts, optimizer steps, finite positive runtime
fields, saved method identity, loss decrease, all task rows, and routing summary.
Active counts above were independently confirmed in all four downloaded
`runtime.json` files. The downloaded archive also confirms clean source `87e58a0`,
matching resolved recipe fields, 50 optimizer steps (400 microbatch examples),
positive training exposure for all six tasks, and exact oracle one-hot routing
weights at every target. The same 60 unique evaluation sample IDs are used in
all runs. Raw sample IDs and questions are kept locally, not committed.

Archive on Blackwell:
`/data/ugrip2026/journal/people/veronika/hydra-integration-smokes/smoke-evidence.tar.gz`.
It contains configs, commit/deck provenance, runtimes, method metadata, metrics,
routing and small training summaries, with a SHA256 manifest; no weights, images,
datasets or raw predictions. Full logs and all predictions remain in the smoke
run directories under `/data/ugrip2026/journal/runs/smoke/`.

No prompt/data change was introduced by Hydra; no new template-gate run is
claimed. These are engineering gates, not medical task-performance evidence or
controlled timing benchmarks. Full training, holdout evaluation, Taha's review,
trained-checkpoint E7 findings and final manuscript results remain separate work.

Downloaded archive SHA256: `54af449d9f4ecb490b0e96fe28ebb954d565ce2efbe46dc689947883b4a366b1`.
Only documentation has changed since tested commit `87e58a0`; implementation and
configs remain unchanged. Raw routing files contain question-based sample IDs,
so only the sanitized audit report is committed. Full artifacts remain on the
server for reviewer access.
