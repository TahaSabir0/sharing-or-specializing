import json, sys, glob, os, collections
root = sys.argv[1]
for d in sorted(glob.glob(root + '/f4smoke_*')):
    p = os.path.join(d, 'all_predictions.json')
    if not os.path.exists(p): print('MISSING', d); continue
    preds = json.load(open(p))
    print('\n#####', os.path.basename(d), len(preds))
    by = collections.defaultdict(list)
    for x in preds: by[x['TaskType']].append(x)
    for t, xs in by.items():
        err = sum(str(x['Answer']).startswith('Error') for x in xs)
        tr = sum(bool(x.get('truncated')) for x in xs); inc = sum(x.get('complete_match') is False for x in xs)
        dis = sum(x.get('Answer_sum', x['Answer']) != x['Answer'] for x in xs)
        print(f'== {t}: n={len(xs)} err={err} trunc={tr} incomplete={inc} sum!=mean={dis}')
        shown = collections.Counter()
        for x in xs:
            if shown[x['dataset']] >= (2 if t == 'classification' else 5): continue
            shown[x['dataset']] += 1
            print(f"  {x['dataset'][:12]:12s} gt={x['gt_answer'][:60]!r:62s} raw={x['raw_output'][:110]!r} ans={x['Answer'][:90]!r} n={x.get('n_new_tokens')} lp={x.get('model_logprob_mean', 0):.2f}")
