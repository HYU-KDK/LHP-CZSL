"""
K-validation: compare LLM-derived K (from sub_meanings_*.json) against
data-driven K (silhouette-based clustering of CLIP features over each primitive's
training images).

Outputs per-primitive (k_visual, k_LLM) pairs, Spearman correlation, and a
disagreement breakdown — used to decide whether LLM K is a reasonable proxy for
visual diversity per primitive.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import CompositionDataset, transform_image, ImageLoader
from clip_modules.clip_model import load_clip


SIL_THRESHOLD = 0.05  # below this, treat as "no meaningful clusters" → k_visual=1
MAX_IMAGES_PER_PRIM = 300
BATCH_SIZE = 64


@torch.no_grad()
def extract_features(image_paths, root, transform, clip_model, device):
    loader = ImageLoader(os.path.join(root, "images") + "/")
    feats = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = torch.stack([transform(loader(p)) for p in batch_paths]).to(device)
        f = clip_model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 768))


def pick_k_visual(features, k_max=5):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(features)
    if n < 4:
        return 1, {}
    sil_by_k = {}
    for k in range(2, min(k_max, n - 1) + 1):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(features)
            if len(set(km.labels_)) < 2:
                continue
            s = silhouette_score(features, km.labels_, metric="cosine")
            sil_by_k[k] = float(s)
        except Exception as e:
            sil_by_k[k] = None
    valid = {k: v for k, v in sil_by_k.items() if v is not None}
    if not valid:
        return 1, sil_by_k
    best_k, best_s = max(valid.items(), key=lambda kv: kv[1])
    if best_s < SIL_THRESHOLD:
        return 1, sil_by_k
    return best_k, sil_by_k


def collect_images(train_data, kind):
    by_prim = defaultdict(list)
    idx = 1 if kind == "attr" else 2
    for img, attr, obj in train_data:
        prim = (attr, obj)[idx - 1]
        by_prim[prim].append(img)
    return by_prim


def run(dataset_path, sub_meanings_path, output_dir, max_per_prim, device):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[k_validation] dataset={dataset_path}  sub_meanings={sub_meanings_path}")
    ds = CompositionDataset(dataset_path, "train")
    transform = transform_image("test")

    print("[k_validation] loading CLIP ViT-L/14")
    clip_model, _ = load_clip("ViT-L/14", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    sub = json.load(open(sub_meanings_path))

    results = {"attrs": {}, "objs": {}}
    for kind, key in [("attr", "attrs"), ("obj", "objs")]:
        prims_by_img = collect_images(ds.train_data, kind)
        prim_names = sorted(set(sub[key].keys()) & set(prims_by_img.keys()))
        print(f"[k_validation] {key}: {len(prim_names)} primitives to evaluate")

        for i, name in enumerate(prim_names):
            imgs = prims_by_img[name]
            if len(imgs) > max_per_prim:
                rng = np.random.default_rng(0)
                imgs = list(rng.choice(imgs, size=max_per_prim, replace=False))
            feats = extract_features(imgs, dataset_path, transform, clip_model, device)
            k_vis, sil = pick_k_visual(feats)
            k_llm = sub[key][name]["K"]
            results[key][name] = {
                "k_visual": int(k_vis),
                "k_llm": int(k_llm),
                "n_images": int(len(imgs)),
                "silhouettes": sil,
            }
            if (i + 1) % 20 == 0 or (i + 1) == len(prim_names):
                print(f"  [{i+1}/{len(prim_names)}] {kind}: '{name}' k_vis={k_vis} k_llm={k_llm} n={len(imgs)}")

    # Summary stats
    summary = {}
    for key in ["attrs", "objs"]:
        rows = list(results[key].values())
        if not rows:
            continue
        kv = np.array([r["k_visual"] for r in rows])
        kl = np.array([r["k_llm"] for r in rows])
        from scipy.stats import spearmanr
        rho, p = spearmanr(kv, kl) if len(kv) >= 3 else (float("nan"), float("nan"))
        agree = int((kv == kl).sum())
        summary[key] = {
            "n": int(len(rows)),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "exact_agreement": agree,
            "exact_agreement_pct": float(agree / len(rows)),
            "mean_k_visual": float(kv.mean()),
            "mean_k_llm": float(kl.mean()),
            "k_visual_dist": {int(k): int((kv == k).sum()) for k in sorted(set(kv.tolist()))},
            "k_llm_dist": {int(k): int((kl == k).sum()) for k in sorted(set(kl.tolist()))},
        }

    out = {"summary": summary, "details": results,
           "config": {"dataset_path": dataset_path, "sub_meanings_path": sub_meanings_path,
                      "sil_threshold": SIL_THRESHOLD, "max_per_prim": max_per_prim}}

    base = os.path.basename(sub_meanings_path).replace(".json", "")
    out_path = os.path.join(output_dir, f"k_validation_{base}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[k_validation] saved → {out_path}")

    print("\n=== SUMMARY ===")
    for key, s in summary.items():
        print(f"\n{key}: n={s['n']}")
        print(f"  Spearman ρ = {s['spearman_rho']:.3f} (p={s['spearman_p']:.3g})")
        print(f"  Exact agreement: {s['exact_agreement']}/{s['n']} ({s['exact_agreement_pct']*100:.1f}%)")
        print(f"  k_visual dist: {s['k_visual_dist']}")
        print(f"  k_llm dist:    {s['k_llm_dist']}")
        print(f"  mean k_visual={s['mean_k_visual']:.2f}  mean k_llm={s['mean_k_llm']:.2f}")

    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--sub_meanings_path", required=True)
    p.add_argument("--output_dir", default="logs/k_validation")
    p.add_argument("--max_per_prim", type=int, default=MAX_IMAGES_PER_PRIM)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.dataset_path, args.sub_meanings_path, args.output_dir, args.max_per_prim, args.device)
