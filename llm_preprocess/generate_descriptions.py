"""
Phase 1: Offline LLM description generator for VP-CMJL contextual prototypes.

Generates three artifacts per dataset (resumable, idempotent):
  A) seen_comp.json         {"attr_obj": "description"}
  B) attr_in_context.json   {"attr": {"obj": "context-specific description"}}
  C) unseen_comp.json       {"attr_obj": "description"}

Composition keys use VP-CMJL's normalized form: lowercase, dots replaced by
spaces, joined with underscore (e.g. "Faux.Fur"+"Boots.Mid-Calf" -> "faux fur_boots mid-calf").

Usage:
    GEMINI_API_KEY=... python llm_preprocess/generate_descriptions.py \
        --dataset_path data/ut-zappos \
        --out_dir data/llm_descriptions/ut-zappos \
        --provider gemini \
        --feasibility_pt VP-CMJL/data/feasibility_ut-zappos.pt \
        --unseen_topk 60

Unseen filtering: feasibility_pt is the per-pair feasibility tensor (shape [N_attr*N_obj])
from `tools/llm_feasibility.py`; the top-K plausible non-train pairs become C's input.
"""
import argparse
import json
import os
import re
import sys
import time
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VP-CMJL"))
from dataset import CompositionDataset


# -------------------- Prompts --------------------

SYSTEM_VISUAL = """You are a visual recognition expert for compositional zero-shot learning.
Describe how a primitive or composition appears in real photographs.
Be concrete: mention color, texture, shape, material, common patterns.
1-2 sentences. No prefacing phrases. No quotes around words."""

PROMPT_SEEN_COMP = """Describe how a "{attr} {obj}" looks in a real photograph.
Focus on color, texture, shape, and material that distinguish this composition.
Output: a single sentence (15-30 words), no preamble, no quotes."""

PROMPT_ATTR_IN_CONTEXT = """Describe how the attribute "{attr}" specifically manifests on a "{obj}" in real photographs.
Compare implicitly: a "{attr}" {obj} differs from {other_examples} -- focus on the visual cue that signals "{attr}" *on this object*.
Output: a single sentence (15-30 words), no preamble, no quotes."""

PROMPT_UNSEEN_COMP = """A "{attr} {obj}" composition is not in the training set, but training set contains: {seen_examples}.
Based on those references, describe what a "{attr} {obj}" would look like in a real photograph.
Focus on color, texture, shape, and material consistent with the references.
Output: a single sentence (15-30 words), no preamble, no quotes."""


# -------------------- Provider backends --------------------

def call_gemini(prompt_user, model="gemini-2.5-flash-lite", system=SYSTEM_VISUAL):
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("set GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=system + "\n\n" + prompt_user,
            )
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            msg = str(e)
            if "503" in msg or "429" in msg or "UNAVAILABLE" in msg.upper():
                time.sleep(5 * (attempt + 1))
                continue
            break
    raise last_err


def call_anthropic(prompt_user, model="claude-sonnet-4-5", system=SYSTEM_VISUAL):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=200, system=system,
        messages=[{"role": "user", "content": prompt_user}],
    )
    return resp.content[0].text.strip()


def call_openai(prompt_user, model="gpt-4o-mini", system=SYSTEM_VISUAL):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model, max_tokens=200,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_user},
        ],
    )
    return resp.choices[0].message.content.strip()


# -------------------- Helpers --------------------

def normalize_name(s):
    return s.replace(".", " ").lower()


def comp_key(attr_norm, obj_norm):
    return f"{attr_norm}_{obj_norm}"


def clean_response(text):
    """Strip surrounding quotes, leading 'a/an', trailing periods normalized."""
    t = text.strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


# -------------------- Generators --------------------

def gen_seen_comp(train_pairs_norm, call_fn, out_path, sleep_s, limit=None):
    """A) seen composition descriptions (one per train pair)."""
    data = load_json(out_path, {})
    pairs = list(train_pairs_norm)
    if limit:
        pairs = pairs[:limit]
    todo = [(a, o) for a, o in pairs if comp_key(a, o) not in data]
    print(f"[seen_comp] total={len(pairs)} done={len(data)} todo={len(todo)}")
    for i, (a, o) in enumerate(todo):
        prompt = PROMPT_SEEN_COMP.format(attr=a, obj=o)
        try:
            txt = clean_response(call_fn(prompt))
        except Exception as e:
            print(f"  [WARN] {a} {o}: {e}")
            txt = f"a {a} {o}"
        data[comp_key(a, o)] = txt
        if (i + 1) % 5 == 0 or (i + 1) == len(todo):
            save_json(out_path, data)
            print(f"  [{i+1}/{len(todo)}] {a} {o} -> {txt[:80]}")
        time.sleep(sleep_s)
    save_json(out_path, data)
    return data


def gen_attr_in_context(train_pairs_norm, attrs_norm, objs_norm, call_fn, out_path, sleep_s):
    """B) per-(attr, obj) contextual description. Only for (attr, obj) pairs in train."""
    data = load_json(out_path, {})
    # Build attr->[objs] from train
    attr_to_objs = {}
    for a, o in train_pairs_norm:
        attr_to_objs.setdefault(a, []).append(o)

    todo = []
    for a, objs in attr_to_objs.items():
        for o in objs:
            if data.get(a, {}).get(o):
                continue
            todo.append((a, o, objs))
    print(f"[attr_in_context] todo={len(todo)}")

    for i, (a, o, sibling_objs) in enumerate(todo):
        # other_examples = up to 3 sibling objs (excluding o), as comma-separated "attr {other}"
        others = [f'"{a} {oo}"' for oo in sibling_objs if oo != o][:3]
        other_str = ", ".join(others) if others else f"other types of {o}"
        prompt = PROMPT_ATTR_IN_CONTEXT.format(attr=a, obj=o, other_examples=other_str)
        try:
            txt = clean_response(call_fn(prompt))
        except Exception as e:
            print(f"  [WARN] {a}|{o}: {e}")
            txt = f"a {a} {o}"
        data.setdefault(a, {})[o] = txt
        if (i + 1) % 5 == 0 or (i + 1) == len(todo):
            save_json(out_path, data)
            print(f"  [{i+1}/{len(todo)}] {a}|{o} -> {txt[:80]}")
        time.sleep(sleep_s)
    save_json(out_path, data)
    return data


def gen_unseen_comp(unseen_pairs_norm, train_pairs_norm, call_fn, out_path, sleep_s):
    """C) unseen composition descriptions. Provide seen pairs sharing attr or obj as references."""
    data = load_json(out_path, {})
    train_set = set(train_pairs_norm)
    train_by_attr = {}
    train_by_obj = {}
    for a, o in train_set:
        train_by_attr.setdefault(a, []).append(o)
        train_by_obj.setdefault(o, []).append(a)

    todo = [(a, o) for a, o in unseen_pairs_norm if comp_key(a, o) not in data]
    print(f"[unseen_comp] total={len(unseen_pairs_norm)} todo={len(todo)}")

    for i, (a, o) in enumerate(todo):
        # Build seen_examples: up to 2 same-attr-different-obj, 2 different-attr-same-obj
        same_attr = [f'"{a} {oo}"' for oo in train_by_attr.get(a, [])][:2]
        same_obj = [f'"{aa} {o}"' for aa in train_by_obj.get(o, [])][:2]
        refs = same_attr + same_obj
        ref_str = ", ".join(refs) if refs else "no direct references"
        prompt = PROMPT_UNSEEN_COMP.format(attr=a, obj=o, seen_examples=ref_str)
        try:
            txt = clean_response(call_fn(prompt))
        except Exception as e:
            print(f"  [WARN] {a} {o}: {e}")
            txt = f"a {a} {o}"
        data[comp_key(a, o)] = txt
        if (i + 1) % 5 == 0 or (i + 1) == len(todo):
            save_json(out_path, data)
            print(f"  [{i+1}/{len(todo)}] {a} {o} -> {txt[:80]}")
        time.sleep(sleep_s)
    save_json(out_path, data)
    return data


# -------------------- Main --------------------

def select_unseen_pairs(attrs_norm, objs_norm, train_pairs_norm,
                        feasibility_pt, topk, n_attr_raw, n_obj_raw):
    """Filter unseen pairs by feasibility tensor; return top-K most plausible."""
    train_set = set(train_pairs_norm)
    all_pairs = []
    if feasibility_pt is None or not os.path.exists(feasibility_pt):
        # fall back: all unseen pairs
        for a in attrs_norm:
            for o in objs_norm:
                if (a, o) not in train_set:
                    all_pairs.append((a, o, 1.0))
        all_pairs.sort()
        return [(a, o) for a, o, _ in all_pairs[:topk] if topk else all_pairs]

    import torch
    pt = torch.load(feasibility_pt, map_location="cpu", weights_only=False)
    feas = pt["feasibility"]  # shape [N_attr * N_obj] flat in (attr, obj) row-major
    assert feas.numel() == n_attr_raw * n_obj_raw, (
        f"feasibility size {feas.numel()} != {n_attr_raw}*{n_obj_raw}"
    )
    feas = feas.view(n_attr_raw, n_obj_raw)

    scored = []
    for ai, a in enumerate(attrs_norm):
        for oi, o in enumerate(objs_norm):
            if (a, o) in train_set:
                continue
            scored.append((a, o, float(feas[ai, oi])))
    scored.sort(key=lambda x: -x[2])
    if topk and topk < len(scored):
        scored = scored[:topk]
    print(f"[unseen_select] scored={len(scored)} | top score={scored[0][2]:.3f} | "
          f"min score in selection={scored[-1][2]:.3f}")
    return [(a, o) for a, o, _ in scored]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--provider", choices=["gemini", "anthropic", "openai"], default="gemini")
    p.add_argument("--model", default=None)
    p.add_argument("--feasibility_pt", default=None,
                   help="path to per-pair feasibility .pt (optional, used to filter unseen)")
    p.add_argument("--unseen_topk", type=int, default=60,
                   help="number of unseen pairs to describe (top-K by feasibility)")
    p.add_argument("--sleep", type=float, default=4.1,
                   help="seconds between API calls (4.1s = 15 RPM Gemini free tier)")
    p.add_argument("--skip", choices=["a", "b", "c", "ab", "ac", "bc", "none"], default="none")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = CompositionDataset(args.dataset_path, "train")
    raw_attrs, raw_objs = ds.attrs, ds.objs
    raw_train_pairs = ds.train_pairs

    attrs_norm = [normalize_name(a) for a in raw_attrs]
    objs_norm = [normalize_name(o) for o in raw_objs]
    train_pairs_norm = [(normalize_name(a), normalize_name(o)) for a, o in raw_train_pairs]

    print(f"[setup] {len(attrs_norm)} attrs, {len(objs_norm)} objs, "
          f"{len(train_pairs_norm)} train pairs")

    # provider
    if args.provider == "gemini":
        if args.model:
            call_fn = lambda u: call_gemini(u, model=args.model)
        else:
            call_fn = call_gemini
    elif args.provider == "anthropic":
        call_fn = lambda u: call_anthropic(u, model=args.model or "claude-sonnet-4-5")
    else:
        call_fn = lambda u: call_openai(u, model=args.model or "gpt-4o-mini")

    sleep_s = args.sleep if args.provider == "gemini" else 0.5

    # A: seen comp
    if "a" not in args.skip:
        gen_seen_comp(train_pairs_norm, call_fn,
                      os.path.join(args.out_dir, "seen_comp.json"), sleep_s)

    # B: attr in context
    if "b" not in args.skip:
        gen_attr_in_context(train_pairs_norm, attrs_norm, objs_norm, call_fn,
                            os.path.join(args.out_dir, "attr_in_context.json"), sleep_s)

    # C: unseen comp
    if "c" not in args.skip:
        unseen = select_unseen_pairs(attrs_norm, objs_norm, train_pairs_norm,
                                     args.feasibility_pt, args.unseen_topk,
                                     n_attr_raw=len(raw_attrs), n_obj_raw=len(raw_objs))
        gen_unseen_comp(unseen, train_pairs_norm, call_fn,
                        os.path.join(args.out_dir, "unseen_comp.json"), sleep_s)

    print("done.")


if __name__ == "__main__":
    main()
