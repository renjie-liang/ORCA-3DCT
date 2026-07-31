#!/usr/bin/env python3
"""Score image-grounded VQA: exact-match accuracy of the generated letter(s) vs gold, per target/axis.
The ONLY genuinely VQA-specific eval (report-gen scores clinical-F1 instead). Joins inference
(raw_btb3d_output.jsonl) to the gold VQA json by SAMPLE ID.

UNIFIED schema: every gold record has `targets` and `answer_letters` LISTS (len 1 for single, 3 for grouped).
The generated answer ("A", or "1:A 2:C 3:B") is parsed positionally for standalone \\b[ABC]\\b letters ->
one scoring path for both.

Run: python score_vqa.py --pred <raw_btb3d_output.jsonl> --gold <per_family/{split}_{fam}.json>

Two hardenings (2026-07-16), both prompted by the stale-label incident where three weeks of runs trained on a
superseded label set without one loud failure:

1. **Chance is per-question-type, not a constant.** It used to be hard-coded 0.333 everywhere. `catalog_tertile`
   is 3-way (0.333) but `catalog_clinical` and `catalog_presence` are BINARY (0.5) — reporting a binary
   question against a 0.333 baseline turns coin-flipping into an apparent win. Chance now comes from the
   gold record's `qtype` and is written per target.
2. **Unmatched predictions are fatal.** They used to be counted and ignored, so pointing --gold at the wrong
   file produced a perfectly normal-looking metrics.json computed over whatever happened to join. That is
   exactly how a wrong label set stays invisible. Any unmatched prediction now aborts.
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

# options per question type -> chance. Keep in sync with data_prep/vqa_labelgen/attributes.py.
CHANCE = {
    "catalog_tertile": 1 / 3,     # (A) Low (B) Moderate (C) High
    "catalog_clinical": 1 / 2,    # binary clinical threshold, e.g. emphysema < -950 HU yes/no
    "catalog_presence": 1 / 2,    # disease present/absent
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="inference jsonl (raw_btb3d_output)")
    ap.add_argument("--gold", required=True, help="the per-family VQA gold json")
    ap.add_argument("--out", default=None, help="where to write vqa_metrics.json; default = next to --pred")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="do NOT abort on predictions with no gold record. Only for debugging a mismatch.")
    args = ap.parse_args()

    gold = json.load(open(args.gold))
    by_id = {r["id"]: r for r in gold}

    bad = [r["id"] for r in gold if r.get("qtype") not in CHANCE]
    if bad:
        sys.exit(f"FATAL: {len(bad)} gold records have an unknown/missing qtype (e.g. {bad[:3]}). "
                 f"Known: {sorted(CHANCE)}. A 'value' (free-number) record cannot be letter-scored — "
                 f"exclude it in the splitter.")

    stat = defaultdict(lambda: [0, 0])        # (axis, target) -> [correct, total]
    qtype_of = {}                             # (axis, target) -> qtype  (for the right chance)
    unmatched = []

    def credit(gr, gen):
        # standalone-letter match (\b[ABC]\b) so prompt-echo like "Answer..." is not mis-read as 'A';
        # positional: letters[i] vs gold answer_letters[i]. Works for single (1 letter) and grouped (3).
        letters = re.findall(r"\b([ABC])\b", str(gen).upper())
        for i, tgt in enumerate(gr["targets"]):
            key = (gr["axis"], tgt)
            stat[key][1] += 1
            qtype_of[key] = gr["qtype"]
            if i < len(letters) and letters[i] == gr["answer_letters"][i]:
                stat[key][0] += 1

    with open(args.pred) as f:
        for line in f:
            d = json.loads(line)
            if "conversations_out" not in d:
                unmatched.append(d.get("id", "<no id>")); continue
            for c in d["conversations_out"]:
                gr = by_id.get(c["id"])
                if gr is None:
                    unmatched.append(c["id"]); continue
                credit(gr, c.get("answer"))

    if unmatched and not args.allow_unmatched:
        sys.exit(f"FATAL: {len(unmatched)} predictions have no gold record — --gold is very likely the wrong "
                 f"file for this run.\n  e.g. {unmatched[:3]}\n  gold={args.gold}\n  pred={args.pred}\n"
                 f"  (pass --allow-unmatched to score anyway; it will be scored over the join only)")
    if not stat:
        sys.exit(f"FATAL: nothing scored — no prediction joined to {args.gold}")

    print(f"VQA exact-match accuracy (unmatched preds: {len(unmatched)})")
    print(f"  {'axis':9s} {'target':38s} {'qtype':17s} {'acc':>6s} {'chance':>7s} {'n':>6s}")
    axis_tot = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for (axis, tgt), (c, n) in sorted(stat.items()):
        ch = CHANCE[qtype_of[(axis, tgt)]]
        print(f"  {axis:9s} {tgt:38s} {qtype_of[(axis,tgt)]:17s} {c/n:6.3f} {ch:7.3f} {n:6d}")
        axis_tot[axis][0] += c; axis_tot[axis][1] += n
        overall[0] += c; overall[1] += n
    print("  --- per-axis ---")
    for axis, (c, n) in sorted(axis_tot.items()):
        print(f"  {axis:9s} {'(all)':38s} {'':17s} {c/max(n,1):6.3f} {'':7s} {n:6d}")
    print(f"  OVERALL {overall[0]/max(overall[1],1):.3f} (n={overall[1]})")

    out = Path(args.out) if args.out else Path(args.pred).with_name("vqa_metrics.json")
    metrics = {
        "pred": str(args.pred), "gold": str(args.gold),
        "n_unmatched": len(unmatched),
        # per-target chance: a binary clinical question is NOT comparable to a 3-way tertile one
        "per_target": {f"{axis}/{tgt}": {"acc": c / n, "n": n,
                                         "qtype": qtype_of[(axis, tgt)],
                                         "chance": CHANCE[qtype_of[(axis, tgt)]]}
                       for (axis, tgt), (c, n) in sorted(stat.items())},
        # per-axis mixes question types, so it has no single chance -- read per_target for that
        "per_axis": {axis: {"acc": c / max(n, 1), "n": n} for axis, (c, n) in sorted(axis_tot.items())},
        "overall": {"acc": overall[0] / max(overall[1], 1), "n": overall[1]},
    }
    out.write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
