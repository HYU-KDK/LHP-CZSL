"""
Phase 1 step 2: encode generated descriptions to CLIP text embeddings (.pt).

Reads three JSON files produced by generate_descriptions.py, encodes each
string with CLIP ViT-L/14's text encoder (frozen, the same encoder VP-CMJL
uses for proxy init), and saves three tensors:

  seen_comp_emb.pt:
      {"keys": [N_seen],                  # ["attr_obj", ...]
       "emb":  Tensor[N_seen, 768]}       # CLIP text embedding

  attr_in_context_emb.pt:
      {"attrs": [N_attr norm names],
       "objs":  [N_obj norm names],
       "emb":   Tensor[N_attr, N_obj, 768],
       "mask":  Tensor[N_attr, N_obj] (1 where description exists, 0 otherwise),
       "attr_fallback": Tensor[N_attr, 768]  # mean over masked entries per attr
      }

  unseen_comp_emb.pt:
      {"pair_idx": Tensor[N_unseen, 2] (attr_idx, obj_idx),
       "emb":      Tensor[N_unseen, 768]}
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VP-CMJL"))
from clip_modules.clip_model import load_clip
from clip_modules.tokenization_clip import SimpleTokenizer
from dataset import CompositionDataset


def encode_texts_clip(texts, clip, tokenizer, context_length=77, batch_size=32, device="cuda"):
    """Encode a list of strings via CLIP frozen text encoder. Returns float32 tensor on CPU."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        toks = tokenizer(chunk, context_length=context_length).to(device)
        with torch.no_grad():
            emb = clip.encode_text(toks)
        out.append(emb.float().cpu())
    return torch.cat(out, dim=0)


def normalize_name(s):
    return s.replace(".", " ").lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--desc_dir", required=True,
                   help="dir containing seen_comp.json / attr_in_context.json / unseen_comp.json")
    p.add_argument("--out_dir", default=None,
                   help="output dir for .pt files (default: same as desc_dir)")
    p.add_argument("--clip_model", default="ViT-L/14")
    p.add_argument("--context_length", type=int, default=77)
    args = p.parse_args()

    out_dir = args.out_dir or args.desc_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[clip] loading {args.clip_model} ...")
    clip = load_clip(name=args.clip_model, context_length=args.context_length).cuda()
    clip.eval()
    tokenizer = SimpleTokenizer()
    emb_dim = clip.visual.output_dim
    print(f"[clip] emb_dim={emb_dim}")

    ds = CompositionDataset(args.dataset_path, "train")
    attrs_norm = [normalize_name(a) for a in ds.attrs]
    objs_norm = [normalize_name(o) for o in ds.objs]
    attr2idx = {a: i for i, a in enumerate(attrs_norm)}
    obj2idx = {o: i for i, o in enumerate(objs_norm)}

    # ---- A: seen_comp ----
    seen_path = os.path.join(args.desc_dir, "seen_comp.json")
    if os.path.exists(seen_path):
        with open(seen_path) as f:
            seen = json.load(f)
        keys = list(seen.keys())
        texts = [seen[k] for k in keys]
        print(f"[seen_comp] N={len(keys)}")
        emb = encode_texts_clip(texts, clip, tokenizer, context_length=args.context_length)
        torch.save({"keys": keys, "emb": emb},
                   os.path.join(out_dir, "seen_comp_emb.pt"))
        print(f"  saved seen_comp_emb.pt  shape={tuple(emb.shape)}")
    else:
        print(f"[seen_comp] missing {seen_path}, skip")

    # ---- B: attr_in_context ----
    aic_path = os.path.join(args.desc_dir, "attr_in_context.json")
    if os.path.exists(aic_path):
        with open(aic_path) as f:
            aic = json.load(f)
        N_a, N_o = len(attrs_norm), len(objs_norm)
        emb = torch.zeros(N_a, N_o, emb_dim, dtype=torch.float32)
        mask = torch.zeros(N_a, N_o, dtype=torch.bool)
        flat_texts, flat_idx = [], []
        for a_name, sub in aic.items():
            if a_name not in attr2idx:
                print(f"  [warn] unknown attr '{a_name}' in attr_in_context, skip")
                continue
            ai = attr2idx[a_name]
            for o_name, txt in sub.items():
                if o_name not in obj2idx:
                    continue
                oi = obj2idx[o_name]
                flat_texts.append(txt)
                flat_idx.append((ai, oi))
        print(f"[attr_in_context] N entries={len(flat_texts)}")
        if flat_texts:
            flat_emb = encode_texts_clip(flat_texts, clip, tokenizer,
                                         context_length=args.context_length)
            for (ai, oi), e in zip(flat_idx, flat_emb):
                emb[ai, oi] = e
                mask[ai, oi] = True
        # per-attr fallback: mean over masked entries; if no entries, leave zeros
        attr_fallback = torch.zeros(N_a, emb_dim, dtype=torch.float32)
        for ai in range(N_a):
            m = mask[ai]
            if m.any():
                attr_fallback[ai] = emb[ai][m].mean(dim=0)
        torch.save({"attrs": attrs_norm, "objs": objs_norm,
                    "emb": emb, "mask": mask, "attr_fallback": attr_fallback},
                   os.path.join(out_dir, "attr_in_context_emb.pt"))
        print(f"  saved attr_in_context_emb.pt  emb={tuple(emb.shape)} "
              f"mask_density={mask.float().mean().item():.3f}")
    else:
        print(f"[attr_in_context] missing {aic_path}, skip")

    # ---- C: unseen_comp ----
    unseen_path = os.path.join(args.desc_dir, "unseen_comp.json")
    if os.path.exists(unseen_path):
        with open(unseen_path) as f:
            unseen = json.load(f)
        keys, texts, idx = [], [], []
        for k, txt in unseen.items():
            # k = "attr_obj" with normalized names; split by last "_"... but obj names
            # may contain spaces too. We split by exact attr/obj membership instead.
            # Find match via attrs/objs lookup.
            matched = None
            for a in attrs_norm:
                if k.startswith(a + "_"):
                    rest = k[len(a) + 1:]
                    if rest in objs_norm:
                        matched = (a, rest)
                        break
            if matched is None:
                print(f"  [warn] could not parse key '{k}' into (attr, obj), skip")
                continue
            a, o = matched
            keys.append(k)
            texts.append(txt)
            idx.append((attr2idx[a], obj2idx[o]))
        print(f"[unseen_comp] N={len(keys)}")
        if texts:
            emb = encode_texts_clip(texts, clip, tokenizer,
                                    context_length=args.context_length)
            pair_idx = torch.tensor(idx, dtype=torch.long)
            torch.save({"keys": keys, "pair_idx": pair_idx, "emb": emb},
                       os.path.join(out_dir, "unseen_comp_emb.pt"))
            print(f"  saved unseen_comp_emb.pt  emb={tuple(emb.shape)} "
                  f"pair_idx={tuple(pair_idx.shape)}")
    else:
        print(f"[unseen_comp] missing {unseen_path}, skip")

    # Also derive an "attr LLM init" tensor for Phase 3 alpha-blend:
    # per-attr mean of all its contextual descriptions (== attr_fallback above).
    # Same for objects via transposed attr_in_context (or via standalone object descriptions).
    # We synthesize the obj fallback symmetrically: per-obj mean over attrs that have desc.
    aic_pt = os.path.join(out_dir, "attr_in_context_emb.pt")
    if os.path.exists(aic_pt):
        d = torch.load(aic_pt, map_location="cpu", weights_only=False)
        emb_b = d["emb"]   # [N_a, N_o, dim]
        mask_b = d["mask"] # [N_a, N_o]
        N_a, N_o, dim = emb_b.shape

        attr_init = torch.zeros(N_a, dim)
        obj_init = torch.zeros(N_o, dim)
        for ai in range(N_a):
            m = mask_b[ai]
            if m.any():
                attr_init[ai] = emb_b[ai][m].mean(dim=0)
        for oi in range(N_o):
            m = mask_b[:, oi]
            if m.any():
                obj_init[oi] = emb_b[m, oi].mean(dim=0)
        torch.save({"attrs": d["attrs"], "objs": d["objs"],
                    "attr_init": attr_init, "obj_init": obj_init},
                   os.path.join(out_dir, "proxy_init_emb.pt"))
        print(f"  saved proxy_init_emb.pt  attr={tuple(attr_init.shape)} obj={tuple(obj_init.shape)}")

    print("done.")


if __name__ == "__main__":
    main()
