#!/usr/bin/env python3
"""Forced-format evaluation (F4 v2, results/format_analysis_plan.md): inference only, both backbones.

Same model loading, rendering, image budget and stop tokens as eval_harness.py / eval_arm.py (bare
prompt for adapters, ZS-prompt system prompt for the base; greedy; thinking off for Qwen; batch 1), but
the answer is restricted to the target task's format (code/eval/forced_format.py):
  counting / regression / multi-label: constrained greedy generation (regex), max_new_tokens 12 / 12 / 64;
  classification: no generation, each option scored by log p(option + end-of-turn | prompt).

  CUDA_VISIBLE_DEVICES=1 python code/eval/forced_eval.py --backbone q35 --mode adapter \
      --adapter /data/ugrip2026/journal/runs/j26_q35_taskdet_full_768_s1337 \
      --tasks reg,count,ml,cls --data /data/ugrip2026/journal/data/j26_holdout.jsonl --surface holdout \
      --out_dir /data/ugrip2026/journal/runs/f4/f4_q35_taskdet_s1337
  --mode zs --protocol prompt     ZS-prompt base (the reference every mu_forced uses)

Outputs in --out_dir (resumable per task): <task>.jsonl, all_predictions.json (bootstrap.load_run
format; Answer = length-normalised choice for cls), run_meta.json. Extra per-row fields: forced,
forced_pattern (ML), model_logprob_sum / _mean (unconstrained log-prob of the forced answer incl. the
stop token), dead_end; cls: option_scores {option: [sum, mean, n_tokens]}, Answer_sum.
"""
import argparse
import datetime
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import forced_format as FF  # noqa: E402
from eval_harness import (DEFAULT_BASE, SHORT_TASK, TASK_SHORT, git_commit, load_base,  # noqa: E402
                          load_peft_named)
from infer_single_adapter import parse_answer  # noqa: E402
from journal_render import (ZS_PROMPT_SYSTEM, capped_images, chat_text, encode_batch,  # noqa: E402
                            eos_ids as pinned_eos, image_rules, load_images)

EOT_TEXT = {'q35': '<|im_end|>', 'mg15': '<end_of_turn>'}
FORCED_TASKS = ('classification', 'multi-label classification', 'counting', 'regression')


class Masker(LogitsProcessor):
    def __init__(self, constraint, n_prompt):
        self.c, self.n_prompt = constraint, n_prompt

    def __call__(self, input_ids, scores):
        allowed = self.c.allowed(input_ids[0, self.n_prompt:].tolist())
        mask = torch.full_like(scores, float('-inf'))
        mask[:, allowed] = 0
        return scores + mask


def generate_forced(model, inputs, constraint, max_new, eos, pad_id):
    n_in = inputs['input_ids'].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=1, temperature=None,
                             top_p=None, top_k=None, eos_token_id=eos, pad_token_id=pad_id,
                             logits_processor=LogitsProcessorList([Masker(constraint, n_in)]),
                             return_dict_in_generate=True, output_logits=True)
    new = out.sequences[0, n_in:].tolist()
    # unconstrained log-prob of each chosen token (output_logits = raw logits, before the mask)
    lps = [float(torch.log_softmax(l[0].float(), -1)[t]) for l, t in zip(out.logits, new)]
    n_new = len(new)
    for i, t in enumerate(new):
        if t in eos:
            n_new = i + 1
            break
    new, lps = new[:n_new], lps[:n_new]
    truncated = not new or new[-1] not in eos
    return new, lps, truncated


def score_options(model, proc, tok, text, images, cands, eot_text, eot_id, backbone):
    """log p(cand + end-of-turn | prompt) for every candidate, one batch (prompt repeated, right-padded)."""
    prompt = encode_batch(proc, [text], [images], backbone)
    n_prompt = prompt['input_ids'].shape[1]
    tok.padding_side = 'right'
    try:
        full = encode_batch(proc, [text + c + eot_text for c in cands], [images] * len(cands), backbone)
    finally:
        tok.padding_side = 'left'
    ids = full['input_ids']
    p = prompt['input_ids'][0]
    lengths = []
    for k in range(len(cands)):
        if not torch.equal(ids[k, :n_prompt], p):
            raise RuntimeError(f'prompt tokens change when {cands[k]!r} is appended')
        n = int(full['attention_mask'][k].sum()) - n_prompt
        if int(ids[k, n_prompt + n - 1]) != eot_id:
            raise RuntimeError(f'answer {cands[k]!r} does not end with the end-of-turn token')
        lengths.append(n)
    keep = ids.shape[1] - n_prompt + 1
    with torch.inference_mode():
        logits = model(**full.to('cuda:0'), logits_to_keep=keep).logits.float()
    logp = torch.log_softmax(logits, -1)  # position j of logp predicts token n_prompt - 1 + j + 1
    scores = {}
    for k, c in enumerate(cands):
        n = lengths[k]
        tgt = ids[k, n_prompt:n_prompt + n].to(logp.device)
        lp = logp[k, torch.arange(n, device=logp.device), tgt]
        scores[c] = [float(lp.sum()), float(lp.mean()), n]
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backbone', choices=['q35', 'mg15'], required=True)
    ap.add_argument('--mode', choices=['zs', 'adapter'], required=True)
    ap.add_argument('--protocol', choices=['bare', 'prompt'], default='bare', help='zero-shot only')
    ap.add_argument('--adapter')
    ap.add_argument('--base_model')
    ap.add_argument('--data', required=True)
    ap.add_argument('--surface', required=True, choices=['holdout', 'devselect'])
    ap.add_argument('--tasks', required=True, help='comma list of short names: cls,ml,count,reg')
    ap.add_argument('--per_task', type=int, default=0)
    ap.add_argument('--per_dataset', type=int, default=0, help='first N rows of each dataset (smoke)')
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    if a.mode != 'zs' and a.protocol == 'prompt':
        raise SystemExit('ZS-prompt is zero-shot only; adapters are evaluated bare (PLAN.md §1.6)')
    if (a.mode == 'adapter') != bool(a.adapter):
        raise SystemExit('--adapter goes with --mode adapter')
    for v in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ.setdefault(v, '1')
    tasks = [SHORT_TASK[t.strip()] for t in a.tasks.split(',') if t.strip()]
    assert all(t in FORCED_TASKS for t in tasks), tasks
    a.base_model = a.base_model or DEFAULT_BASE[a.backbone]
    rules = image_rules(a.backbone)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_all = [json.loads(l) for l in open(a.data) if l.strip()]
    answers_by_ds = {}
    for r in rows_all:
        answers_by_ds.setdefault(r['dataset'], set()).add(r['answer'])

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(a.base_model, local_files_only=True)
    tok = proc.tokenizer
    tok.padding_side = 'left'
    eos = pinned_eos(tok, a.backbone)
    eot_text = EOT_TEXT[a.backbone]
    eot = tok.convert_tokens_to_ids(eot_text)
    assert eot == eos[0], (eot, eos)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    model = load_base(a.base_model)
    if a.mode == 'adapter':
        model = load_peft_named(model, {'a': a.adapter})
    model.eval()
    system = ZS_PROMPT_SYSTEM if (a.mode == 'zs' and a.protocol == 'prompt') else None

    t0 = time.time()
    strs = FF.token_strings(tok)
    strs_index = FF.index_strings(strs)
    num_cands = FF.number_candidates(strs)
    print(f'[forced] token table {time.time() - t0:.0f}s, {sum(s is not None for s in strs)} text tokens, '
          f'{len(num_cands)} number tokens: {[strs[i] for i in num_cands][:40]}', flush=True)

    meta = {'experiment': 'F4 v2 forced-format decoding', 'mode': a.mode, 'protocol': a.protocol if a.mode == 'zs' else None,
            'adapter': a.adapter, 'backbone': a.backbone, 'base_model': a.base_model, 'surface': a.surface,
            'data': a.data, 'tasks': tasks, 'per_task': a.per_task, 'per_dataset': a.per_dataset, 'system_prompt': system,
            'constraints': {'counting': FF.COUNT_RE, 'regression': FF.REG_RE,
                            'multi-label classification': 'question options + own separator (per row)',
                            'classification': 'log p(option + end-of-turn); Answer = length-normalised, Answer_sum = summed'},
            'max_new_tokens': FF.MAX_NEW, 'eos_token_id': eos, 'end_of_turn': eot_text,
            'decoding': 'greedy (do_sample=False, num_beams=1), constrained', 'enable_thinking': False if a.backbone == 'q35' else None,
            'image_max_pixels': rules['max_pixels'], 'image_min_pixels': rules['min_pixels'], 'batch_size': 1,
            'host': socket.gethostname(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'code_commit': git_commit(), 'started': datetime.datetime.now().isoformat(timespec='seconds')}
    json.dump(meta, open(out_dir / 'run_meta.json', 'w'), indent=1)

    timings = {}
    for task in tasks:
        rows = [r for r in rows_all if r['task'] == task]
        if a.per_task:
            rows = rows[:a.per_task]
        if a.per_dataset:
            seen = {}
            rows = [r for r in rows if seen.setdefault(r['dataset'], []).append(1) or len(seen[r['dataset']]) <= a.per_dataset]
        ckpt = out_dir / f'{TASK_SHORT[task]}.jsonl'
        done = {json.loads(l)['id'] for l in open(ckpt) if l.strip()} if ckpt.exists() else set()
        todo = [r for r in rows if r['id'] not in done]
        print(f'\n=== {task}: {len(rows)} rows, {len(done)} done, {len(todo)} to run', flush=True)
        ts, fails = time.time(), 0
        with open(ckpt, 'a', encoding='utf-8') as fout:
            for k, r in enumerate(todo):
                rec = dict(id=r['id'], ImageName=r['images'], Question=r['question'], TaskType=task,
                           dataset=r['dataset'], modality=r.get('modality'), gt_answer=r['answer'], forced=True)
                try:
                    paths = capped_images({'image': r['images']})
                    images = load_images(paths, **rules)
                    text = chat_text(proc, r['question'], len(paths), system=system, backbone=a.backbone)
                    if task == 'classification':
                        cands = FF.cls_candidates(r['question'], answers_by_ds[r['dataset']])
                        sc = score_options(model, proc, tok, text, images, cands, eot_text, eot, a.backbone)
                        best_mean = max(cands, key=lambda c: sc[c][1])
                        best_sum = max(cands, key=lambda c: sc[c][0])
                        rec.update(raw_output=best_mean, Answer=parse_answer(best_mean, task),
                                   Answer_sum=parse_answer(best_sum, task), option_scores=sc,
                                   model_logprob_sum=sc[best_mean][0], model_logprob_mean=sc[best_mean][1],
                                   n_new_tokens=sc[best_mean][2], truncated=False, dead_end=False)
                    else:
                        if task == 'multi-label classification':
                            opts, sep = FF.ml_options(r['question'])
                            pattern = FF.ml_regex(opts, sep)
                            cands = FF.ml_candidates(strs_index, opts, sep)
                            rec['forced_pattern'] = pattern
                        else:
                            sep = None
                            pattern = FF.COUNT_RE if task == 'counting' else FF.REG_RE
                            cands = num_cands
                        c = FF.RegexConstraint(pattern, cands, strs, eot)
                        inputs = encode_batch(proc, [text], [images], a.backbone).to('cuda:0')
                        new, lps, trunc = generate_forced(model, inputs, c, FF.MAX_NEW[task], eos, pad_id)
                        body = [t for t in new if t not in eos]
                        raw = tok.decode(body, skip_special_tokens=True)
                        if raw != c.text(body):
                            raise RuntimeError(f'decoded {raw!r} != constrained text {c.text(body)!r}')
                        complete = c.complete(raw)
                        ans_text = FF.dedupe_labels(raw, sep) if sep else raw
                        rec.update(raw_output=raw, Answer=parse_answer(ans_text, task), n_new_tokens=len(new),
                                   truncated=trunc, complete_match=complete, dead_end=c.dead_end,
                                   model_logprob_sum=sum(lps), model_logprob_mean=sum(lps) / max(len(lps), 1))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    fails += 1
                    torch.cuda.empty_cache()
                    rec.update(raw_output='', Answer=f'Error: {e}', n_new_tokens=0, truncated=False)
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                if (k + 1) % 25 == 0 or k + 1 == len(todo):
                    el = time.time() - ts
                    print(f'  [{task}] {k + 1}/{len(todo)} {el / (k + 1):.2f}s/q fails {fails} '
                          f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB', flush=True)
        timings[task] = {'seconds': round(time.time() - ts, 1), 'rows_run': len(todo), 'fails': fails}

    combined = []
    for task in tasks:
        p = out_dir / f'{TASK_SHORT[task]}.jsonl'
        if p.exists():
            combined += [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    json.dump(combined, open(out_dir / 'all_predictions.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    meta.update(finished=datetime.datetime.now().isoformat(timespec='seconds'), timings=timings,
                n_predictions=len(combined), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    json.dump(meta, open(out_dir / 'run_meta.json', 'w'), indent=1)
    print(f'\nDONE {len(combined)} predictions -> {out_dir}/all_predictions.json', flush=True)


if __name__ == '__main__':
    main()
