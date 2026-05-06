"""
Failure-case analysis for a single CZSL checkpoint.

Loads a trained model, runs through the test set, and writes a JSON report
with per-sample predictions plus aggregate failure statistics:

  * attr / obj / pair top-1 accuracy (closed)
  * confusion top-K for attributes (which attrs get mis-predicted as which)
  * confusion top-K for objects
  * per-test-pair failure rate (which (attr, obj) is hardest)
  * seen vs unseen breakdown
  * sample-level CSV of (path, gt_attr, gt_obj, pred_attr, pred_obj, correct)
    for downstream qualitative inspection

Usage:
  python tools/failure_analysis.py \
      --yml_path config/lhp_czsl_v3_text_utzap_l14_seed0.yml \
      --load_model checkpoint/lhp_czsl_v3_text_l14_utzap_seed0/val_best.pt \
      --output_dir logs/failure_analysis/v3_text_seed0
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_args
from parameters import parser as base_parser
from dataset import CompositionDataset
from model.model_factory import get_model
from test import Evaluator, predict_logits, predict_logits_text_first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yml_path", required=True)
    p.add_argument("--load_model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_confusion_k", type=int, default=10)
    p.add_argument("--top_failed_pairs_k", type=int, default=20)
    p.add_argument("--bias", type=float, default=1e-3,
                   help="match yml test.bias (default close to v3_text yml)")
    args = p.parse_args()

    config = base_parser.parse_args([])
    load_args(args.yml_path, config)
    config.load_model = args.load_model
    config.open_world = False

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[failure] yml={args.yml_path}")
    print(f"[failure] ckpt={args.load_model}")

    test_dataset = CompositionDataset(config.dataset_path, phase="test",
                                      split="compositional-split-natural",
                                      open_world=False)

    allattrs = test_dataset.attrs
    allobj = test_dataset.objs
    classes = [c.replace(".", " ").lower() for c in allobj]
    attributes = [a.replace(".", " ").lower() for a in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(config.load_model))
    model.eval()

    predict_fn = predict_logits
    if hasattr(config, "text_first") and config.text_first \
            and hasattr(model, "encode_text_for_open"):
        predict_fn = predict_logits_text_first

    evaluator = Evaluator(test_dataset, model=None)

    print("[failure] running inference on test set")
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, _ = predict_fn(
        model, test_dataset, config)

    # turn logits dict (keyed by (attr,obj)) into score tensor [N, n_pairs]
    pair_keys = test_dataset.pairs
    scores = torch.stack(
        [all_logits[k] for k in pair_keys], dim=1).float().cpu()

    # mimic Evaluator.generate_predictions: closed-world top-1 (with bias)
    seen_mask = evaluator.seen_mask  # [n_pairs] bool, True for train pairs
    closed_mask = evaluator.closed_mask  # [n_pairs] bool, True for in-test pairs

    # biased closed: add bias to unseen-but-in-test pairs
    biased = scores.clone()
    biased[:, ~seen_mask] += args.bias
    biased[:, ~closed_mask] = -1e10
    _, biased_top = biased.topk(1, dim=1)
    biased_top = biased_top.view(-1)
    biased_attr = evaluator.pairs[biased_top][:, 0]
    biased_obj = evaluator.pairs[biased_top][:, 1]

    # unbiased closed: just mask out non-test pairs, no bias
    unbiased = scores.clone()
    unbiased[:, ~closed_mask] = -1e10
    _, unbiased_top = unbiased.topk(1, dim=1)
    unbiased_top = unbiased_top.view(-1)
    unbiased_attr = evaluator.pairs[unbiased_top][:, 0]
    unbiased_obj = evaluator.pairs[unbiased_top][:, 1]

    # Build seen/unseen mask per sample (test pair is unseen iff not in train)
    train_pair_set = set(test_dataset.train_pairs)
    sample_is_unseen = torch.tensor([
        (allattrs[a], allobj[o]) not in train_pair_set
        for a, o in zip(all_attr_gt.tolist(), all_obj_gt.tolist())
    ])

    # Aggregate
    n = len(all_attr_gt)
    correct_attr_b = (biased_attr == all_attr_gt).int()
    correct_obj_b = (biased_obj == all_obj_gt).int()
    correct_pair_b = (correct_attr_b * correct_obj_b)

    correct_attr_u = (unbiased_attr == all_attr_gt).int()
    correct_obj_u = (unbiased_obj == all_obj_gt).int()
    correct_pair_u = (correct_attr_u * correct_obj_u)

    def acc_split(mask, vec):
        if mask.sum() == 0:
            return None
        return float(vec[mask].float().mean())

    summary = {
        "n_test_samples": int(n),
        "n_attrs": len(allattrs),
        "n_objs": len(allobj),
        "n_test_pairs": int(closed_mask.sum()),
        "n_seen": int((~sample_is_unseen).sum()),
        "n_unseen": int(sample_is_unseen.sum()),
        "biased_closed": {
            "attr_acc": float(correct_attr_b.float().mean()),
            "obj_acc": float(correct_obj_b.float().mean()),
            "pair_acc": float(correct_pair_b.float().mean()),
            "attr_acc_seen": acc_split(~sample_is_unseen, correct_attr_b),
            "attr_acc_unseen": acc_split(sample_is_unseen, correct_attr_b),
            "obj_acc_seen": acc_split(~sample_is_unseen, correct_obj_b),
            "obj_acc_unseen": acc_split(sample_is_unseen, correct_obj_b),
            "pair_acc_seen": acc_split(~sample_is_unseen, correct_pair_b),
            "pair_acc_unseen": acc_split(sample_is_unseen, correct_pair_b),
        },
        "unbiased_closed": {
            "attr_acc": float(correct_attr_u.float().mean()),
            "obj_acc": float(correct_obj_u.float().mean()),
            "pair_acc": float(correct_pair_u.float().mean()),
        },
    }

    # Confusion: which gt attr most often gets predicted as which other attr
    attr_conf = defaultdict(Counter)
    for gt, pred, ok in zip(all_attr_gt.tolist(), unbiased_attr.tolist(), correct_attr_u.tolist()):
        if not ok:
            attr_conf[allattrs[gt]][allattrs[pred]] += 1
    attr_conf_top = []
    for gt_name, cnt in attr_conf.items():
        gt_total = int((all_attr_gt == allattrs.index(gt_name)).sum())
        attr_conf_top.append({
            "gt_attr": gt_name,
            "gt_total": gt_total,
            "fail_count": int(sum(cnt.values())),
            "fail_rate": float(sum(cnt.values()) / max(1, gt_total)),
            "top_confusions": cnt.most_common(5),
        })
    attr_conf_top.sort(key=lambda r: r["fail_count"], reverse=True)

    # Same for objects
    obj_conf = defaultdict(Counter)
    for gt, pred, ok in zip(all_obj_gt.tolist(), unbiased_obj.tolist(), correct_obj_u.tolist()):
        if not ok:
            obj_conf[allobj[gt]][allobj[pred]] += 1
    obj_conf_top = []
    for gt_name, cnt in obj_conf.items():
        gt_total = int((all_obj_gt == allobj.index(gt_name)).sum())
        obj_conf_top.append({
            "gt_obj": gt_name,
            "gt_total": gt_total,
            "fail_count": int(sum(cnt.values())),
            "fail_rate": float(sum(cnt.values()) / max(1, gt_total)),
            "top_confusions": cnt.most_common(5),
        })
    obj_conf_top.sort(key=lambda r: r["fail_count"], reverse=True)

    # Per-pair failure rate (using biased closed = standard CZSL eval)
    pair_stats = []
    for i, (a_idx, o_idx) in enumerate(zip(all_attr_gt.tolist(), all_obj_gt.tolist())):
        pass
    pair_results = defaultdict(lambda: {"total": 0, "ok": 0, "is_unseen": False})
    for a_idx, o_idx, ok, is_unseen in zip(all_attr_gt.tolist(), all_obj_gt.tolist(),
                                            correct_pair_b.tolist(), sample_is_unseen.tolist()):
        key = (allattrs[a_idx], allobj[o_idx])
        pair_results[key]["total"] += 1
        pair_results[key]["ok"] += int(ok)
        pair_results[key]["is_unseen"] = bool(is_unseen)
    pair_rows = []
    for (a, o), stats in pair_results.items():
        fail = stats["total"] - stats["ok"]
        pair_rows.append({
            "attr": a, "obj": o, "is_unseen": stats["is_unseen"],
            "total": stats["total"], "ok": stats["ok"], "fail": fail,
            "fail_rate": fail / max(1, stats["total"]),
        })
    pair_rows.sort(key=lambda r: r["fail"], reverse=True)

    out = {
        "config": {
            "yml": args.yml_path,
            "ckpt": args.load_model,
            "dataset": config.dataset,
            "bias": args.bias,
        },
        "summary": summary,
        "attr_confusion_top": attr_conf_top[:args.top_confusion_k],
        "obj_confusion_top": obj_conf_top[:args.top_confusion_k],
        "hardest_pairs": pair_rows[:args.top_failed_pairs_k],
        "easiest_pairs": [r for r in pair_rows if r["total"] >= 5][-10:],
    }

    base = os.path.basename(args.load_model.rstrip("/"))
    parent = os.path.basename(os.path.dirname(args.load_model.rstrip("/")))
    tag = f"{parent}__{base}".replace("val_best.pt", "val_best")
    out_path = os.path.join(args.output_dir, f"failure_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[failure] saved -> {out_path}")

    # Console summary
    print("\n=== SUMMARY ===")
    s = summary["biased_closed"]
    print(f"  attr_acc={s['attr_acc']:.4f}  obj_acc={s['obj_acc']:.4f}  pair_acc={s['pair_acc']:.4f}")
    print(f"  unseen attr_acc={s['attr_acc_unseen']}  seen attr_acc={s['attr_acc_seen']}")
    print(f"\n=== TOP CONFUSED ATTRS (gt -> pred) ===")
    for r in attr_conf_top[:5]:
        print(f"  {r['gt_attr']:>20s}  fail={r['fail_count']:>4d}/{r['gt_total']:<4d}  "
              f"({r['fail_rate']*100:.1f}%)  top: {r['top_confusions'][:3]}")
    print(f"\n=== HARDEST PAIRS ===")
    for r in pair_rows[:5]:
        tag = "UNSEEN" if r["is_unseen"] else "seen"
        print(f"  ({r['attr']:>15s}, {r['obj']:<25s})  [{tag}]  "
              f"{r['ok']}/{r['total']}  fail_rate={r['fail_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
