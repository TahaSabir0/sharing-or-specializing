# Journal results ledger

The numbers of record for the journal study. One section per study (S0–S11), appended as
runs finish.

For every result, record:
- run ID(s) from `journal/RUNS.csv`;
- backbone, method, data (subset / seed);
- code branch and commit;
- evaluation surface (holdout / dev / validation-public), with sample counts;
- per-dataset metrics plus the task macro, and the malformed, truncation and omission
  rates;
- deviations from PLAN.md §1, if any.

Raw predictions go to `predictions/<study>/<run_id>/`.

Challenge-era numbers never go here; they live in `../references/challenge_results/`.

---

*No results yet.*
