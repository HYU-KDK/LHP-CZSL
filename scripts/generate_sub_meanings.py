"""
Phase 0: LLM Sub-meaning Generation Script

MIT-States의 모든 attr/obj에 대해 시각적 하위 의미를 생성합니다.

Usage:
  # With Anthropic Claude API
  ANTHROPIC_API_KEY=sk-... python scripts/generate_sub_meanings.py \
      --dataset_path data/mit-states \
      --output data/sub_meanings_mit.json \
      --provider anthropic

  # With OpenAI API
  OPENAI_API_KEY=sk-... python scripts/generate_sub_meanings.py \
      --dataset_path data/mit-states \
      --output data/sub_meanings_mit.json \
      --provider openai
"""

import argparse
import json
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import CompositionDataset


SYSTEM_PROMPT = """You are a visual recognition expert. Your task is to analyze visual primitives (attributes or objects) and identify their visually distinct sub-types.

IMPORTANT RULES:
1. Only include sub-meanings that are VISUALLY distinguishable in images
2. If the primitive has no meaningful visual sub-types, return just the primitive itself (K=1)
3. Maximum 5 sub-meanings
4. Semantic distances should reflect VISUAL difference (0=identical appearance, 1=completely different appearance)
5. Be conservative: fewer distinct sub-types is better than many overlapping ones"""

USER_PROMPT_TEMPLATE = """Analyze the {prim_type} "{name}" for compositional zero-shot learning.

List its visually distinct sub-meanings (how this {prim_type} appears differently in images).

For example:
- "old" (attribute) → worn, faded, aged (3 visually distinct appearances)
- "knife" (object) → kitchen knife, pocket knife, butter knife (3 visually distinct shapes)
- "bright" (attribute) → bright (only 1, no meaningful visual sub-types)

Respond ONLY with valid JSON, no explanation:
{{
  "sub": ["sub1", "sub2", ...],
  "K": <number of sub-meanings>,
  "d_sem": {{
    "sub1-sub2": <distance 0-1>,
    ...
  }}
}}"""


def call_anthropic(system_prompt, user_prompt, model="claude-sonnet-4-20250514"):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text


def call_openai(system_prompt, user_prompt, model="gpt-4o"):
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def parse_llm_response(response_text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response_text.strip()
    # Remove markdown code block if present
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    # Try to find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def generate_for_primitive(name, prim_type, provider, call_fn):
    """Generate sub-meanings for a single primitive."""
    user_prompt = USER_PROMPT_TEMPLATE.format(prim_type=prim_type, name=name)
    try:
        response = call_fn(SYSTEM_PROMPT, user_prompt)
        result = parse_llm_response(response)
        # Validate
        assert 'sub' in result and 'K' in result
        result['K'] = min(result['K'], 5)
        result['sub'] = result['sub'][:5]
        if 'd_sem' not in result:
            result['d_sem'] = {}
        return result
    except Exception as e:
        print(f"  [WARN] Failed for {prim_type} '{name}': {e}")
        return {"sub": [name], "K": 1, "d_sem": {}}


def generate_all(dataset_path, output_path, provider, resume=True):
    """Generate sub_meanings for all primitives in MIT-States."""
    ds = CompositionDataset(dataset_path, 'train')
    attrs = ds.attrs
    objs = ds.objs

    # Resume from existing file if available
    if resume and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
        print(f"Resuming from {output_path}")
    else:
        data = {"attrs": {}, "objs": {}}

    if provider == 'anthropic':
        call_fn = call_anthropic
    elif provider == 'openai':
        call_fn = call_openai
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Generate attrs
    for i, attr in enumerate(attrs):
        if attr in data['attrs']:
            continue
        print(f"[{i+1}/{len(attrs)}] attr: {attr}")
        result = generate_for_primitive(attr, "attribute", provider, call_fn)
        data['attrs'][attr] = result
        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        time.sleep(0.5)

    # Generate objs
    for i, obj in enumerate(objs):
        if obj in data['objs']:
            continue
        print(f"[{i+1}/{len(objs)}] obj: {obj}")
        result = generate_for_primitive(obj, "object", provider, call_fn)
        data['objs'][obj] = result
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        time.sleep(0.5)

    print(f"\nDone! Saved to {output_path}")
    print(f"  Attrs: {len(data['attrs'])}/{len(attrs)}")
    print(f"  Objs: {len(data['objs'])}/{len(objs)}")

    # Summary
    attr_ks = [v['K'] for v in data['attrs'].values()]
    obj_ks = [v['K'] for v in data['objs'].values()]
    print(f"  Attr K distribution: min={min(attr_ks)}, max={max(attr_ks)}, mean={sum(attr_ks)/len(attr_ks):.1f}")
    print(f"  Obj K distribution: min={min(obj_ks)}, max={max(obj_ks)}, mean={sum(obj_ks)/len(obj_ks):.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provider', choices=['anthropic', 'openai'], required=True)
    args = parser.parse_args()
    generate_all(args.dataset_path, args.output, args.provider)
