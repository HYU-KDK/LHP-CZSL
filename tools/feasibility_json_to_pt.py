"""
Convert tools/llm_feasibility.py JSON output -> the .pt tensor format that
test.py expects for open-world feasibility filtering.

Format expected by test.py:639:
    torch.load(path)['feasibility']  -> [n_pairs] tensor aligned to dataset.pairs

Pairs come in the order returned by CompositionDataset.parse_split() (sorted
across train+val+test pair lists). This script emits scores in that exact
order and normalizes the LLM 0-10 scale to a unit-interval feasibility:
    feasibility = score / 10.0

Compositions absent from the LLM JSON (resume failure, etc.) get score=0.5
(neutral) so they are not silently dropped from the candidate set.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import CompositionDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--feasibility_json", required=True,
                   help="Output of tools/llm_feasibility.py")
    p.add_argument("--output_pt", required=True)
    p.add_argument("--open_world", action="store_true",
                   help="Use full attr x obj product (open-world). "
                        "Default uses dataset.pairs (closed-world test pair list).")
    args = p.parse_args()

    ds = CompositionDataset(args.dataset_path, "train",
                            open_world=args.open_world)
    pairs = ds.pairs
    print(f"[convert] dataset_path={args.dataset_path}  open_world={args.open_world}")
    print(f"[convert] expected pair count: {len(pairs)}")

    raw = json.load(open(args.feasibility_json))
    score_map = {}
    for r in raw["scores"]:
        if r.get("score") is None:
            continue
        score_map[(r["attr"], r["obj"])] = float(r["score"]) / 10.0
    print(f"[convert] scored pairs in JSON: {len(score_map)}")

    feas = torch.zeros(len(pairs), dtype=torch.float32)
    missing = []
    for i, p in enumerate(pairs):
        if p in score_map:
            feas[i] = score_map[p]
        else:
            feas[i] = 0.5
            missing.append(p)
    print(f"[convert] missing pairs (filled with 0.5): {len(missing)}")
    if missing[:5]:
        print(f"  e.g. {missing[:5]}")

    out = {"feasibility": feas, "pairs": pairs}
    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
    torch.save(out, args.output_pt)
    print(f"[convert] saved -> {args.output_pt}")
    print(f"[convert] feas stats: min={feas.min():.3f}  max={feas.max():.3f}  mean={feas.mean():.3f}")


if __name__ == "__main__":
    main()
