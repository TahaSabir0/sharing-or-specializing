"""Normal-on-abnormal rate for IU_XRay report-gen predictions -- ONE definition, applied identically
to every model (8B Qwen3-VL flagship, 9B Qwen3.5), so the numbers are comparable.

Definition (explicit; the stats/16 s3f 8B figure of 75.9% was produced by an ad-hoc agent pass whose
rule was never written down, so both models are RE-computed here with the same code):
  * A sentence is a NORMAL statement if every clause is either a negation ("no ...", "without ...",
    "negative for ...") or a normality assertion ("within normal limits", "normal heart size",
    "lungs are clear", "unremarkable", "intact", ...) and carries no un-negated positive finding.
  * A gold sentence is ABNORMAL if it is not a NORMAL statement.
  * normal_on_abnormal = P(pred is NORMAL | gold is ABNORMAL).
Also reports: gold abnormal fraction, pred normal fraction, unique-pred ratio, top-10 coverage,
and the "modal sentence" collapse fraction that reportgen_sanity_proxy.py reported for the 9B (14.5%).

Input: jsonl of {"gt":..., "pred":...} (reportgen_pairs.jsonl / predictions_true_test.jsonl format).
Usage: python reportgen_normal_on_abnormal.py <name>=<pairs.jsonl> [<name>=<pairs.jsonl> ...]
"""
import json, re, sys, collections

NEG_TRIGGERS = r"\b(no|not|without|negative for|free of|absent|clear of|resolved)\b"
NORMAL_WORDS = r"\b(normal|unremarkable|clear|intact|within normal limits|wnl|patent|preserved|well[- ]aerated|non[- ]?acute|stable|unchanged|expanded|midline|appropriate|adequate)\b"

# positive findings; if one appears in a clause with no negation trigger, the clause is abnormal
FINDINGS = [
    "opacit", "consolidation", "effusion", "pneumothorax", "edema", "atelectasis", "nodul", "mass",
    "cardiomegaly", "enlarge", "infiltrat", "fracture", "degenerative", "scoliosis", "kyphosis",
    "hyperinflat", "hyperexpan", "emphysema", "granuloma", "calcifi", "fibrosis", "scar", "thicken",
    "prominen", "tortuo", "ectatic", "elevat", "blunting", "densit", "hernia", "interstitial",
    "airspace", "air space", "congestion", "deformity", "spondylosis", "osteophyt", "hardware",
    "wire", "device", "pacemaker", "catheter", "tube", "stent", "clip", "suture", "cabg", "sternotomy",
    "surgical", "aneurysm", "widen", "low lung volume", "hypoinflat", "eventration", "lymphaden",
    "cyst", "lesion", "abnormal", "increased", "decreased", "shift", "displace", "collapse",
    "bronchiectasis", "bullae", "bleb", "cavit", "hazy", "haziness", "streaky", "reticul", "patchy",
    "chronic", "mild", "moderate", "severe", "borderline", "suspicious", "concern", "possible",
    "residual", "disease",
    "compression", "fusion", "arthropathy", "arthritis", "obscur", 
]

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def is_normal_statement(s):
    s = norm(s)
    if not s:
        return False
    clauses = [c.strip() for c in re.split(r"[.;]", s) if c.strip()]
    if not clauses:
        return False
    for c in clauses:
        has_neg = re.search(NEG_TRIGGERS, c) is not None
        has_normal = re.search(NORMAL_WORDS, c) is not None
        pos = any(t in c for t in FINDINGS)
        if has_neg:
            continue            # "no pleural effusion", "without acute abnormality"
        if pos:
            return False        # un-negated finding word -> abnormal ("mild cardiomegaly", "stable scarring")
        if not has_normal:
            return False        # neither negated nor asserting normality -> treat as abnormal/other
    return True

def profile(name, path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows)
    gt_abn = [r for r in rows if not is_normal_statement(r["gt"])]
    gt_nrm = n - len(gt_abn)
    pred_normal = sum(1 for r in rows if is_normal_statement(r["pred"]))
    noa = sum(1 for r in gt_abn if is_normal_statement(r["pred"]))
    aon = sum(1 for r in rows if is_normal_statement(r["gt"]) and not is_normal_statement(r["pred"]))
    cnt = collections.Counter(norm(r["pred"]) for r in rows)
    gcnt = collections.Counter(norm(r["gt"]) for r in rows)
    modal, modal_n = cnt.most_common(1)[0]
    gt_not_modal = [r for r in rows if norm(r["gt"]) != modal]
    modal_collapse = sum(1 for r in gt_not_modal if norm(r["pred"]) == modal)
    top10 = sum(c for _, c in cnt.most_common(10))
    boiler = {a for a, c in cnt.items() if c >= 10}
    boiler_frac = sum(1 for r in rows if norm(r["pred"]) in boiler) / n
    return dict(
        model=name, n=n,
        gold_abnormal_n=len(gt_abn), gold_abnormal_frac=round(len(gt_abn)/n, 4),
        gold_normal_n=gt_nrm,
        pred_normal_frac=round(pred_normal/n, 4),
        normal_on_abnormal_n=noa,
        normal_on_abnormal_rate=round(noa/max(1, len(gt_abn)), 4),
        abnormal_on_normal_rate=round(aon/max(1, gt_nrm), 4),
        unique_pred_ratio=round(len(cnt)/n, 4), unique_gold_ratio=round(len(gcnt)/n, 4),
        top10_pred_coverage=round(top10/n, 4),
        boilerplate_pred_frac_ge10=round(boiler_frac, 4),
        modal_pred=modal[:80], modal_frac=round(modal_n/n, 4),
        modal_collapse_fraction_sanityproxy_style=round(modal_collapse/max(1, len(gt_not_modal)), 4),
    ), rows, gt_abn

if __name__ == "__main__":
    show = "--examples" in sys.argv
    res = []
    for arg in [a for a in sys.argv[1:] if "=" in a]:
        name, path = arg.split("=", 1)
        o, rows, gt_abn = profile(name, path)
        res.append(o)
        print(json.dumps(o, indent=2))
        if show:
            print(f"  -- {name}: normal-on-abnormal examples:")
            for r in [r for r in gt_abn if is_normal_statement(r["pred"])][:6]:
                print(f"     GT:   {r['gt'][:110]}\n     PRED: {r['pred'][:110]}")
            print(f"  -- {name}: abnormal-gold, abnormal pred:")
            for r in [r for r in gt_abn if not is_normal_statement(r["pred"])][:4]:
                print(f"     GT:   {r['gt'][:110]}\n     PRED: {r['pred'][:110]}")
    json.dump(res, open("reportgen_normal_on_abnormal_results.json", "w"), indent=2)
