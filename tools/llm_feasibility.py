"""
LLM-based composition feasibility scoring for CZSL open-world evaluation.

For each (attr, obj) composition in a dataset's full candidate space, ask an
LLM to rate how visually plausible / realistic the composition is on a 0-10
scale, and write the result to a JSON file. The score is later applied as a
multiplicative mask on open-world logits to suppress nonsense compositions
like ('fluffy', 'steel') or ('ripe', 'cement') without affecting
well-formed pairs like ('green', 'apple').

Usage:
    ANTHROPIC_API_KEY=sk-... python tools/llm_feasibility.py \
        --dataset_path /home/student/dongki/DPAS/data/ut-zap50k \
        --output data/feasibility_utzap.json \
        --provider anthropic \
        --batch_size 16

The script is incremental: existing entries in the output file are skipped
on re-runs, so it's safe to interrupt and resume.
"""
import argparse
import json
import os
import re
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import CompositionDataset


SYSTEM_PROMPT = """You are a visual recognition expert. Given an
(attribute, object) composition, judge how visually realistic / plausible it
is in real-world photographs on a 0-10 scale.

Scoring guide:
  0-1   physically impossible or never occurs
        ("fluffy steel", "ripe cement", "wooden water")
  2-4   very unusual, only in artistic/fictional contexts
        ("dripping mountain", "metallic flower")
  5-7   uncommon but plausible
        ("rusty banana" - exists if a banana sticker on rust, etc.)
  8-10  common everyday composition
        ("green apple", "leather boots", "wet road")

Be calibrated: most random pairs should score 0-3. Reserve 8-10 for
compositions you'd actually expect to find in image search results."""

USER_PROMPT_TEMPLATE = """Rate the visual plausibility of this composition:

  Composition: "{attr} {obj}"

Respond ONLY with valid JSON (no extra prose), in this exact form:
{{
  "score": <integer 0-10>,
  "reason": "<one short sentence>"
}}"""


def call_anthropic_batch(pairs, model="claude-sonnet-4-5"):
    """Call Anthropic API once per (attr, obj) pair. Returns list of dicts."""
    import anthropic
    client = anthropic.Anthropic()
    out = []
    for attr, obj in pairs:
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=128,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": USER_PROMPT_TEMPLATE.format(attr=attr, obj=obj)}],
            )
            text = resp.content[0].text
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON in response: {text[:200]}")
            data = json.loads(m.group(0))
            score = int(data.get("score", 0))
            reason = data.get("reason", "")
            out.append({"attr": attr, "obj": obj, "score": max(0, min(10, score)),
                        "reason": reason})
        except Exception as e:
            print(f"  [WARN] failed for ({attr}, {obj}): {e}")
            out.append({"attr": attr, "obj": obj, "score": None, "reason": str(e)})
    return out


def call_openai_batch(pairs, model="gpt-4o-mini"):
    import openai
    client = openai.OpenAI()
    out = []
    for attr, obj in pairs:
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=128,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": USER_PROMPT_TEMPLATE.format(attr=attr, obj=obj)},
                ],
            )
            text = resp.choices[0].message.content
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON in response: {text[:200]}")
            data = json.loads(m.group(0))
            score = int(data.get("score", 0))
            reason = data.get("reason", "")
            out.append({"attr": attr, "obj": obj, "score": max(0, min(10, score)),
                        "reason": reason})
        except Exception as e:
            print(f"  [WARN] failed for ({attr}, {obj}): {e}")
            out.append({"attr": attr, "obj": obj, "score": None, "reason": str(e)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default=None,
                   help="override default model (anthropic: claude-sonnet-4-5, openai: gpt-4o-mini)")
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--max_pairs", type=int, default=None,
                   help="cap number of pairs (for quick smoke test)")
    args = p.parse_args()

    ds = CompositionDataset(args.dataset_path, "train")
    all_pairs = list(product(ds.attrs, ds.objs))
    print(f"[feasibility] dataset attrs={len(ds.attrs)} objs={len(ds.objs)} "
          f"total candidate pairs={len(all_pairs)}")

    # Resume from existing
    if os.path.exists(args.output):
        with open(args.output) as f:
            data = json.load(f)
        scored = {(e["attr"], e["obj"]) for e in data["scores"] if e.get("score") is not None}
        print(f"[feasibility] resuming, already-scored pairs: {len(scored)}")
    else:
        data = {"meta": {"dataset_path": args.dataset_path, "provider": args.provider},
                "scores": []}
        scored = set()

    todo = [p for p in all_pairs if p not in scored]
    if args.max_pairs:
        todo = todo[:args.max_pairs]
    print(f"[feasibility] pairs remaining: {len(todo)}")

    call_fn = call_anthropic_batch if args.provider == "anthropic" else call_openai_batch

    # Process in chunks, save incrementally
    CHUNK = args.save_every
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i+CHUNK]
        results = call_fn(batch, model=args.model) if args.model else call_fn(batch)
        data["scores"].extend(results)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        ok = sum(1 for r in results if r.get("score") is not None)
        print(f"  [{i+len(batch)}/{len(todo)}]  ok={ok}/{len(batch)}  "
              f"sample: {results[0]['attr']} {results[0]['obj']} -> {results[0].get('score')}")
        time.sleep(0.5)

    # Summary
    scores = [e["score"] for e in data["scores"] if e.get("score") is not None]
    if scores:
        from collections import Counter
        print(f"\n[summary] n={len(scores)}  mean={sum(scores)/len(scores):.2f}  "
              f"score dist: {sorted(Counter(scores).items())}")


if __name__ == "__main__":
    main()
