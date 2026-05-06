"""
Generate LLM visual descriptions for CZSL primitives (axis E Phase A).

For each attribute and object in a dataset, ask an LLM to produce K visual
descriptions (~30-50 tokens each, CLIP text encoder friendly). The output
JSON is drop-in compatible with `data/sub_meanings_*.json` so that
`model/lhp_czsl.py:_load_sub_meanings` reads it directly when
`text_ensemble=True`.

Output format (compatible with existing _load_sub_meanings):
    {
      "attrs": {
        "Canvas": {
          "sub": ["<desc 1>", "<desc 2>", "<desc 3>"],
          "K": 3,
          "d_sem": {}
        },
        ...
      },
      "objs": {...}
    }

Usage:
    GEMINI_API_KEY=... python tools/llm_descriptions.py \\
        --dataset_path /home/student/dongki/DPAS/data/ut-zap50k \\
        --output data/descriptions_utzap.json \\
        --K 3 \\
        --tokens 40

Token budget: descriptions are kept short (default ~40 tokens) so K=3-5
descriptions concatenated stay under CLIP's 77-token context length when
ensembled.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import CompositionDataset


SYSTEM_PROMPT = """You are a visual recognition expert generating CLIP-friendly
descriptions for compositional zero-shot learning. Each description should:
1. Be concrete and visual (texture, color, shape, material, context)
2. Stay under ~40 tokens to fit CLIP's 77-token context length
3. Use simple language CLIP was trained on (avoid jargon)
4. Vary across the K descriptions to capture distinct visual sub-types
"""

ATTR_PROMPT_TEMPLATE = """Generate {K} distinct visual descriptions for the
attribute "{attr}" as it appears in {domain_hint} images.

Each description should depict a different VISUAL SUB-TYPE of "{attr}". For
example, "leather" has smooth/grained/patent sub-types with distinct visual
appearance. If the attribute has only one visual mode, generate {K}
descriptions emphasizing different visual axes (texture vs color vs context).

Respond with a JSON list of exactly {K} short visual descriptions (no extra
prose):
[
  "description 1 (~40 tokens, concrete visual)",
  "description 2",
  ...
]"""

OBJ_PROMPT_TEMPLATE = """Generate {K} distinct visual descriptions for the
object category "{obj}" as it appears in {domain_hint} images.

Each description should depict a different VISUAL SUB-TYPE of "{obj}". For
example, "sandals" has flat/sport/heeled sub-types with distinct visual
appearance.

Respond with a JSON list of exactly {K} short visual descriptions (no extra
prose):
[
  "description 1 (~40 tokens, concrete visual)",
  "description 2",
  ...
]"""


def domain_hint(dataset_path):
    """Heuristic domain description based on dataset name."""
    name = os.path.basename(dataset_path.rstrip("/")).lower()
    if "ut-zap" in name or "zap" in name:
        return "footwear / shoe product"
    if "mit-states" in name or "mit_states" in name:
        return "everyday object"
    if "c-gqa" in name or "cgqa" in name:
        return "natural scene"
    return "natural"


def _format_name(name):
    """Match LHP-CZSL's _format_sub_name behavior — replace _ . with space."""
    return name.replace("_", " ").replace(".", " ").strip()


def _call_gemini_one(client, model, prim_name, K, prompt_template, hint):
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=SYSTEM_PROMPT + "\n\n" + prompt_template.format(
                    attr=_format_name(prim_name) if "{attr}" in prompt_template
                         else prim_name,
                    obj=_format_name(prim_name) if "{obj}" in prompt_template
                        else prim_name,
                    K=K,
                    domain_hint=hint,
                ),
            )
            text = resp.text or ""
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON list in response: {text[:200]}")
            descs = json.loads(m.group(0))
            descs = [str(d).strip() for d in descs if str(d).strip()]
            if len(descs) < K:
                raise ValueError(f"got {len(descs)} descs, need {K}: {descs}")
            return descs[:K]
        except Exception as e:
            last_err = e
            msg = str(e)
            if "503" in msg or "429" in msg or "UNAVAILABLE" in msg.upper():
                time.sleep(5 * (attempt + 1))
                continue
            break
    print(f"  [WARN] failed for '{prim_name}': {last_err}")
    return [_format_name(prim_name)] * K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--K", type=int, default=3,
                   help="number of descriptions per primitive")
    p.add_argument("--tokens", type=int, default=40,
                   help="target tokens per description (advisory, in prompt)")
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--pace_sec", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=10)
    args = p.parse_args()

    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("set GEMINI_API_KEY (or GOOGLE_API_KEY)")
    client = genai.Client(api_key=api_key)

    ds = CompositionDataset(args.dataset_path, "train")
    hint = domain_hint(args.dataset_path)
    print(f"[descriptions] dataset attrs={len(ds.attrs)} objs={len(ds.objs)} "
          f"K={args.K} domain='{hint}'")

    # Resume from existing
    if os.path.exists(args.output):
        with open(args.output) as f:
            data = json.load(f)
        print(f"[descriptions] resuming, "
              f"already-done attrs={len(data.get('attrs', {}))} "
              f"objs={len(data.get('objs', {}))}")
    else:
        data = {"attrs": {}, "objs": {}}

    todo_attrs = [a for a in ds.attrs if a not in data["attrs"]]
    todo_objs = [o for o in ds.objs if o not in data["objs"]]
    print(f"[descriptions] todo attrs={len(todo_attrs)} objs={len(todo_objs)}")

    # Attrs
    for i, attr in enumerate(todo_attrs):
        descs = _call_gemini_one(client, args.model, attr, args.K,
                                 ATTR_PROMPT_TEMPLATE, hint)
        data["attrs"][attr] = {"sub": descs, "K": len(descs), "d_sem": {}}
        if (i + 1) % args.save_every == 0 or i == len(todo_attrs) - 1:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [attr {i+1}/{len(todo_attrs)}] {attr}: {descs[0][:60]}...")
        time.sleep(args.pace_sec)

    # Objs
    for i, obj in enumerate(todo_objs):
        descs = _call_gemini_one(client, args.model, obj, args.K,
                                 OBJ_PROMPT_TEMPLATE, hint)
        data["objs"][obj] = {"sub": descs, "K": len(descs), "d_sem": {}}
        if (i + 1) % args.save_every == 0 or i == len(todo_objs) - 1:
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [obj {i+1}/{len(todo_objs)}] {obj}: {descs[0][:60]}...")
        time.sleep(args.pace_sec)

    print(f"\n[done] wrote {args.output}")
    print(f"  attrs: {len(data['attrs'])} primitives × K={args.K}")
    print(f"  objs:  {len(data['objs'])} primitives × K={args.K}")
    print(f"\n[next] Use as drop-in replacement for sub_meanings_*.json:")
    print(f"  v3_text config sub_meanings_path → {args.output}")
    print(f"  Example sub_meanings entry kept format-compatible.")


if __name__ == "__main__":
    main()
