"""
Prototype-collapse diagnostic for ClusPro / LHP-CZSL checkpoints.

For each saved checkpoint:
  - Load every per-primitive prototype buffer (`attr_queue{i}`, `obj_queue{i}`).
  - Each is [K, D]. Compute, per primitive:
      * mean off-diagonal cosine between the K prototypes  (1 = collapsed, 0 = orthogonal)
      * min off-diagonal cosine                            (worst-case separation)
      * effective K via participation ratio of singular values:
            eff_K = (sum s)^2 / sum s^2     ∈ [1, K]
        eff_K ≈ 1 → all prototypes lie on a single direction (collapse)
        eff_K ≈ K → prototypes span K independent directions
  - Aggregate per checkpoint (mean across primitives).
  - Print a comparison table across all checkpoints.
"""
import argparse
import json
import os
import re
from glob import glob

import torch
import torch.nn.functional as F


def primitive_metrics(P):
    """P: [K, D] prototype matrix → dict of metrics."""
    K = P.shape[0]
    Pn = F.normalize(P.float(), dim=-1)
    sim = Pn @ Pn.T                                     # [K, K]
    iu = torch.triu_indices(K, K, offset=1)
    off = sim[iu[0], iu[1]]                             # [K*(K-1)/2]
    # Effective K via singular values of P
    s = torch.linalg.svdvals(P.float())
    eff_k = (s.sum() ** 2 / (s ** 2).sum()).item() if (s ** 2).sum() > 0 else 1.0
    return {
        "mean_offdiag_cos": float(off.mean()),
        "min_offdiag_cos":  float(off.min()),
        "max_offdiag_cos":  float(off.max()),
        "eff_K":            float(eff_k),
        "K":                int(K),
    }


def diagnose_checkpoint(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    qkeys = [k for k in sd.keys() if re.match(r"^(attr|obj)_queue\d+$", k)]
    per_prim = []
    for k in qkeys:
        m = primitive_metrics(sd[k])
        m["key"] = k
        m["kind"] = "attr" if k.startswith("attr") else "obj"
        per_prim.append(m)
    if not per_prim:
        return None
    summary = {"n_primitives": len(per_prim)}
    for kind in ["attr", "obj", "all"]:
        rows = per_prim if kind == "all" else [r for r in per_prim if r["kind"] == kind]
        if not rows:
            continue
        for metric in ["mean_offdiag_cos", "min_offdiag_cos", "eff_K"]:
            vals = [r[metric] for r in rows]
            summary[f"{kind}_{metric}_mean"] = float(sum(vals) / len(vals))
        summary[f"{kind}_eff_K_min"] = float(min(r["eff_K"] for r in rows))
        summary[f"{kind}_eff_K_max"] = float(max(r["eff_K"] for r in rows))
    summary["per_primitive"] = per_prim
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", default="checkpoint")
    ap.add_argument("--output", default="logs/k_validation/prototype_diagnostic.json")
    args = ap.parse_args()

    folders = sorted(glob(os.path.join(args.checkpoint_root, "*")))
    folders = [f for f in folders if os.path.isdir(f) and os.path.exists(os.path.join(f, "val_best.pt"))]
    print(f"[diag] inspecting {len(folders)} checkpoints under {args.checkpoint_root}/")
    results = {}
    for f in folders:
        name = os.path.basename(f)
        s = diagnose_checkpoint(os.path.join(f, "val_best.pt"))
        if s is None:
            print(f"  [skip] {name}: no prototype buffers")
            continue
        results[name] = s
        print(f"  {name:50s}  attr off-cos={s['attr_mean_offdiag_cos_mean']:.3f}  obj off-cos={s['obj_mean_offdiag_cos_mean']:.3f}  attr eff_K={s['attr_eff_K_mean']:.2f}/{s['per_primitive'][0]['K']}  obj eff_K={s['obj_eff_K_mean']:.2f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        # Drop per_primitive details from JSON to keep it small (kept in memory above)
        slim = {n: {k: v for k, v in s.items() if k != "per_primitive"} for n, s in results.items()}
        json.dump(slim, f, indent=2)
    print(f"[diag] saved → {args.output}")

    # Pretty comparison table
    print("\n=== SUMMARY (attr | obj) ===")
    print(f"{'checkpoint':<50s} {'attr off-cos':>12s} {'attr effK':>10s} {'obj off-cos':>12s} {'obj effK':>10s}")
    print("-" * 100)
    for name in sorted(results):
        s = results[name]
        K = s["per_primitive"][0]["K"]
        print(f"{name:<50s} {s['attr_mean_offdiag_cos_mean']:>12.3f} {s['attr_eff_K_mean']:>7.2f}/{K} {s['obj_mean_offdiag_cos_mean']:>12.3f} {s['obj_eff_K_mean']:>7.2f}/{K}")


if __name__ == "__main__":
    main()
