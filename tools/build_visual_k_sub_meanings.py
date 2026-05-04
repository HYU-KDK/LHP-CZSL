"""
Build a sub_meanings JSON whose K per primitive is determined by visual
silhouette clustering on CLIP features (from tools/k_validation.py output)
instead of by an LLM.

Used to falsify the variable-K thesis under correct K determination:
  - If LHP-CZSL trained with visual-K beats fixed-K=5 baseline,
    the LLM K choice was the issue.
  - If it ties baseline, the variable-K mechanism itself is not the lever.

Sub-meaning *names* are kept simple — replicate the primitive name K times.
This isolates the K-determination effect from the semantic priors carried
by LLM-generated names. (Use --pad_with_llm to mix LLM names instead.)
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_validation_json", required=True,
                    help="Path to a k_validation_*.json produced by tools/k_validation.py")
    ap.add_argument("--llm_sub_meanings_json", required=True,
                    help="The original LLM-derived sub_meanings JSON (for primitive list / d_sem fallback)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pad_with_llm", action="store_true",
                    help="Use LLM sub-meaning names where possible (truncate/pad to visual K), "
                         "instead of repeating the primitive name.")
    args = ap.parse_args()

    kv = json.load(open(args.k_validation_json))["details"]
    llm = json.load(open(args.llm_sub_meanings_json))

    out = {"attrs": {}, "objs": {}}
    for kind in ["attrs", "objs"]:
        if kind not in kv or kind not in llm:
            continue
        for name, info in kv[kind].items():
            k_vis = int(info["k_visual"])
            llm_entry = llm[kind].get(name, {"sub": [name], "K": 1, "d_sem": {}})
            llm_subs = llm_entry.get("sub", [name])
            if args.pad_with_llm:
                # Use LLM names; truncate to k_vis or pad with primitive name.
                sub = list(llm_subs[:k_vis])
                while len(sub) < k_vis:
                    sub.append(name)
            else:
                # Replicate primitive name k_vis times — clean K-only test.
                sub = [name] * k_vis
            out[kind][name] = {"sub": sub, "K": k_vis, "d_sem": llm_entry.get("d_sem", {})}

    # Add primitives that exist in LLM but not in k_validation (skipped silently above).
    # Use LLM K for those (fallback).
    for kind in ["attrs", "objs"]:
        for name, llm_entry in llm.get(kind, {}).items():
            if name not in out[kind]:
                out[kind][name] = llm_entry

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Summary
    summary = {}
    for kind in ["attrs", "objs"]:
        ks = [v["K"] for v in out[kind].values()]
        if not ks: continue
        from collections import Counter
        summary[kind] = {"n": len(ks), "mean_K": sum(ks)/len(ks), "dist": dict(Counter(ks))}
    print(f"Wrote {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
