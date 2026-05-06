"""
Aggregate failure_analysis.py JSONs across multiple checkpoints.

Usage: python tools/failure_aggregate.py <dir>
Reads every failure_*.json in <dir>, prints:
  - per-variant attr_acc/obj_acc/pair_acc means
  - top attribute confusions across all variants (which attr-attr pairs
    are persistently hard regardless of model)
  - top hardest pairs across all variants (which (attr, obj) is hardest
    in the dataset, model-agnostic signal)
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict


def main():
    if len(sys.argv) != 2:
        print("usage: failure_aggregate.py <dir>")
        sys.exit(1)
    files = sorted(glob.glob(os.path.join(sys.argv[1], "failure_*.json")))
    if not files:
        print(f"no failure_*.json under {sys.argv[1]}")
        sys.exit(1)

    print(f"=== loaded {len(files)} failure analyses ===\n")

    # Per-checkpoint summary
    print(f"{'checkpoint':<60s}  {'attr_acc':>9s}  {'obj_acc':>9s}  {'pair_acc':>9s}  "
          f"{'attr_unseen':>12s}  {'pair_unseen':>12s}")
    rows = []
    for fp in files:
        d = json.load(open(fp))
        s = d["summary"]["biased_closed"]
        tag = os.path.basename(fp).replace("failure_", "").replace(".json", "")
        rows.append((tag, s))
        print(f"{tag:<60s}  {s['attr_acc']:>9.4f}  {s['obj_acc']:>9.4f}  "
              f"{s['pair_acc']:>9.4f}  "
              f"{(s['attr_acc_unseen'] or 0):>12.4f}  "
              f"{(s['pair_acc_unseen'] or 0):>12.4f}")

    # Aggregate attribute confusions across all checkpoints
    attr_pair_count = Counter()  # (gt_attr, pred_attr) -> total count across ckpts
    attr_gt_total = Counter()    # gt_attr -> total samples
    for fp in files:
        d = json.load(open(fp))
        for r in d["attr_confusion_top"]:
            for pred, cnt in r["top_confusions"]:
                attr_pair_count[(r["gt_attr"], pred)] += cnt
            attr_gt_total[r["gt_attr"]] += r["gt_total"]

    print("\n=== TOP ATTRIBUTE CONFUSIONS (gt -> pred), summed across all ckpts ===")
    print(f"{'gt_attr':<22s} -> {'pred_attr':<22s}  {'count':>6s}  {'~per_ckpt':>10s}")
    for (gt, pred), cnt in attr_pair_count.most_common(20):
        per_ckpt = cnt / len(files)
        print(f"{gt:<22s} -> {pred:<22s}  {cnt:>6d}  {per_ckpt:>10.1f}")

    # Aggregate object confusions
    obj_pair_count = Counter()
    for fp in files:
        d = json.load(open(fp))
        for r in d["obj_confusion_top"]:
            for pred, cnt in r["top_confusions"]:
                obj_pair_count[(r["gt_obj"], pred)] += cnt

    print("\n=== TOP OBJECT CONFUSIONS (gt -> pred) ===")
    print(f"{'gt_obj':<32s} -> {'pred_obj':<32s}  {'count':>6s}  {'~per_ckpt':>10s}")
    for (gt, pred), cnt in obj_pair_count.most_common(15):
        per_ckpt = cnt / len(files)
        print(f"{gt:<32s} -> {pred:<32s}  {cnt:>6d}  {per_ckpt:>10.1f}")

    # Hardest pairs across all checkpoints
    pair_fail = Counter()
    pair_total = Counter()
    pair_unseen = {}
    for fp in files:
        d = json.load(open(fp))
        for r in d["hardest_pairs"]:
            key = (r["attr"], r["obj"])
            pair_fail[key] += r["fail"]
            pair_total[key] += r["total"]
            pair_unseen[key] = r["is_unseen"]

    print("\n=== HARDEST (attr, obj) PAIRS ACROSS ALL CKPTS ===")
    print(f"{'pair':<50s}  {'split':>6s}  {'fail/total':>12s}  {'rate':>6s}")
    pair_rows = sorted(pair_fail.items(), key=lambda kv: kv[1], reverse=True)
    for key, fail in pair_rows[:20]:
        total = pair_total[key]
        rate = fail / max(1, total)
        tag = "UNSEEN" if pair_unseen.get(key, False) else "seen"
        s = f"({key[0]}, {key[1]})"
        print(f"{s:<50s}  {tag:>6s}  {fail:>5d}/{total:<5d}  {rate*100:>5.1f}%")


if __name__ == "__main__":
    main()
