# LHP-CZSL

**LLM-Hierarchical-Prototype Compositional Zero-Shot Learning** — does giving each primitive a different number of prototypes (decided by an LLM) help over a fixed-K ClusPro baseline?

This repo is a research workspace investigating that question on MIT-States and UT-Zappos. **Current empirical answer: tie** (gain within seed noise across 4 method variants × 2 datasets). See [§ Current state](#current-state) below — this README is honest about the negative result and lays out the diagnostic plan.

---

## Table of Contents
- [Background](#background)
- [The idea (variable K)](#the-idea-variable-k)
- [Method variants in this repo](#method-variants-in-this-repo)
- [Current state](#current-state)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Reproducing the headline numbers](#reproducing-the-headline-numbers)
- [Where to read more](#where-to-read-more)

---

## Background

**Compositional Zero-Shot Learning (CZSL)** asks a model trained on `(attr, obj)` pairs like `(red, apple)` and `(green, banana)` to recognize unseen pairs like `(green, apple)` at test time. Each image is labeled by an attribute (e.g., *red*, *old*, *canvas*) and an object (e.g., *apple*, *boots*).

We build on **ClusPro** (cluster-prototype CZSL): each primitive (attr or obj) is represented by `K` learned prototypes, and a contrastive loss pulls features near their primitive's prototypes. The original ClusPro fixes `K=5` for every primitive.

## The idea (variable K)

ClusPro's fixed `K=5` is wasteful for primitives that are *visually homogeneous* (e.g., the material *Cotton* — basically one look) and possibly insufficient for primitives that are *visually diverse* (e.g., the attribute *bright* — luminous, reflective, vivid). We hypothesize that letting `K` vary per primitive — `K=1` for narrow concepts, `K=3~5` for broad ones — should outperform fixed `K=5`.

**How K is currently chosen**: an LLM (Claude / GPT-4o) is asked, for each primitive name, "what visually distinct sub-meanings does this have?" It returns a list (size 1–5) and pairwise semantic distances. See [`scripts/generate_sub_meanings.py`](scripts/generate_sub_meanings.py).

> **Caveat**: this LLM call sees only the primitive *name*, never any image. Its output isn't validated against the actual data distribution. A `tools/k_validation.py` experiment (in flight at the time of writing) checks whether LLM-derived `K` correlates with `K` chosen by silhouette-based clustering of CLIP image features — see [§ Current state](#current-state).

## Method variants in this repo

| Tag | What changes vs ClusPro baseline | Where |
|-----|----------------------------------|-------|
| `cluspro_baseline` | Original ClusPro, fixed `K=5` | [`model/cluspro_baseline.py`](model/cluspro_baseline.py) |
| `lhp_czsl_v1` | + variable `K` (LLM) + `L_sem` semantic-distance alignment + cosine decorrelation between sub-prototypes | [`model/lhp_czsl.py`](model/lhp_czsl.py) |
| `lhp_czsl_v1_init_only` | v1 with `decorr_weight=0`, `sem_weight=0` — only the LLM-derived **prototype init** is kept | yml: `init_lamda=0.1`, `decorr_weight=0`, `sem_weight=0` |
| `lhp_czsl_v2` | v1 + LLM re-prompted to force `K≥3` (richer sub-meaning lists, [`data/sub_meanings_mit_v2.json`](data/sub_meanings_mit_v2.json)) | yml: `sub_meanings_path` swap |
| `lhp_czsl_v3_text` | v1_init_only + **text-side** sub-meaning ensemble: each primitive's prompt is averaged across its sub-meaning prompts | yml: `text_ensemble: True` |

All variants train the same way (`train.py`); the model is selected by `model_name` in the yml.

## Current state

### Headline numbers

**MIT-States (ViT-L/14, 15 epochs, seed 0)** — closed-world test:

| Run | HM | AUC |
|-----|-----|-----|
| ClusPro baseline (K=5) | 0.3893 | 0.2169 |
| ClusPro baseline (K=3) | 0.3855 | 0.2158 |
| LHP-CZSL v1 (variable K + L_sem + decorr) | 0.3886 | 0.2182 |
| LHP-CZSL v1_init_only | 0.3874 | 0.2155 |
| LHP-CZSL v2 (K≥3 forced) | 0.3897 | 0.2179 |

All variants land within **HM ±0.005, AUC ±0.003** — within seed noise. (Full table with all metrics in [`RESEARCH_LOG.md`](RESEARCH_LOG.md) §1.)

**UT-Zappos (ViT-L/14, 15 epochs, 3-seed mean ± std)** — closed-world test:

| Run | HM | AUC |
|-----|-----|-----|
| ClusPro baseline (K=5, fp16+GradScaler+guard) | 0.5481 | 0.4279 |
| LHP-CZSL v1_init_only | 0.5553 | 0.4330 |
| LHP-CZSL v3_text (text ensemble, n=2 + 1 in flight) | 0.5456 | 0.4260 |

Δ vs baseline ≤ 0.007 HM, all inside seed std. **Tie pattern reproduces from MIT-States to UT-Zap.**

### Why tie? Two candidate explanations under test

| # | Hypothesis | Diagnostic | Status |
|---|------------|------------|--------|
| **A** | LLM-derived K doesn't track visual diversity → K assignment is noisy | [`tools/k_validation.py`](tools/k_validation.py): per-primitive Spearman ρ between `K_LLM` and `K_visual` (silhouette over CLIP features) on UT-Zap + mit-states v1/v2 | ⏳ runs after seed2 finishes (auto-triggered by `scripts/wait_seed2_then_kvalidation.sh`) |
| **B** | K is fine; sub-prototypes collapse during training (so variable K acts like K=1) | enable `decorr_weight`, ablate K>1 subset only | next, after A |

If ρ is high → focus on (B). If ρ is low → revisit how K is determined (multimodal LLM, data-driven, or hybrid).

### Engineering note: AMP NaN fix

UT-Zap baseline runs prior to 5/2 had **NaN-corrupted loss** during training — fp16 autocast without GradScaler / explicit isfinite guards let occasional inf×0 patterns through. The fix combines:
1. `torch.cuda.amp.GradScaler` for backward,
2. `torch.isfinite(loss)` skip in the train loop ([`train.py:63`](train.py)),
3. EMA prototype-update guard mirrored in [`model/lhp_czsl.py`](model/lhp_czsl.py).

**Pre-fix UT-Zap baseline numbers in any older run are noise-corrupted and have been re-run.** See `RESEARCH_LOG.md` §2 for the timeline.

## Repository layout

```
LHP-CZSL/
├── README.md                         ← you are here
├── RESEARCH_LOG.md                   ← full chronological experiment log
├── train.py / test.py                ← training & evaluation entry points
├── parameters.py                     ← yml → argparse Namespace
├── dataset.py                        ← CompositionDataset (mit-states / ut-zap50k)
│
├── model/
│   ├── cluspro_baseline.py           ← ClusPro baseline (fixed K)
│   ├── lhp_czsl.py                   ← LHP-CZSL (variable K, L_sem, ensemble)
│   ├── hsic.py / nce_loss.py / ...   ← shared losses
│   └── otgcc.py                      ← optimal-transport greedy cluster ctr
│
├── clip_modules/                     ← OpenAI CLIP (vendored)
│
├── config/
│   ├── cluspro_baseline_*.yml        ← baseline configs (mit, utzap, lr/K probes)
│   └── lhp_czsl_v{1,2,3}_*.yml       ← variant configs, per dataset & seed
│
├── data/
│   ├── sub_meanings_mit.json         ← v1 LLM output, mit-states (mostly K=1)
│   ├── sub_meanings_mit_v2.json      ← v2 LLM output, mit-states (forced K≥3)
│   ├── sub_meanings_utzap.json       ← v1 LLM output, ut-zappos
│   └── sub_meanings_example.json     ← format reference
│
├── scripts/
│   ├── generate_sub_meanings.py      ← Phase 0 — call LLM to produce sub_meanings_*.json
│   ├── run_baseline_utzap_3seeds.sh  ← multi-seed launchers
│   ├── run_v1_init_only_utzap_3seeds.sh
│   ├── run_v3_text_utzap_3seeds.sh
│   ├── wait_seed2_then_kvalidation.sh  ← chain k-validation after training
│   └── sanity_v1.py                  ← LHP v1 model sanity check
│
├── tools/
│   ├── k_validation.py               ← LLM K vs visual-cluster K diagnostic
│   ├── mixup.py / optimization.py    ← training utilities
│
└── docs/
    ├── model_design.md               ← Stage 1 design (variable K + L_sem)
    ├── future_work.md                ← prioritized open questions
    ├── CHANGELOG.md                  ← method-version changelog (v1 → v2)
    └── research_log.md               ← earlier project-side narrative
```

`logs/` and `checkpoint/` are produced by training and `.gitignore`d.

## Quick start

### 1. Environment

```bash
conda create -n lhp_czsl python=3.9 -y
conda activate lhp_czsl
pip install torch torchvision  # CUDA build matching your driver
pip install ftfy regex pyyaml tqdm scikit-learn scipy numpy pillow
```

CLIP weights are downloaded by [`clip_modules/clip_model.py:load_clip`](clip_modules/clip_model.py) on first use (cached under `~/.cache/clip/`).

### 2. Datasets

Expected layout (paths configured per yml under `dataset_path`):
```
<DATA_ROOT>/
├── mit-states/
│   ├── images/
│   ├── compositional-split-natural/{train,val,test}_pairs.txt
│   └── metadata_compositional-split-natural.t7
└── ut-zap50k/
    ├── images/
    ├── compositional-split-natural/{train,val,test}_pairs.txt
    └── metadata_compositional-split-natural.t7
```

Standard CZSL splits — see e.g. the original ClusPro / Czsl-CGE datasets.

### 3. (Optional) regenerate sub-meanings

The LLM-derived sub-meanings are committed as JSON, so you don't need an API key to reproduce. To regenerate:

```bash
ANTHROPIC_API_KEY=sk-... python scripts/generate_sub_meanings.py \
    --dataset_path <DATA_ROOT>/mit-states \
    --output data/sub_meanings_mit.json \
    --provider anthropic
```

## Reproducing the headline numbers

All training is single-GPU. Set `CUDA_VISIBLE_DEVICES` as needed.

### ClusPro baseline (UT-Zap, 3 seeds)

```bash
bash scripts/run_baseline_utzap_3seeds.sh
# checkpoints land in checkpoint/cluspro_baseline_l14_utzap_v2_seed{0,1,2}/
```

### LHP-CZSL v1_init_only (UT-Zap, 3 seeds)

```bash
bash scripts/run_v1_init_only_utzap_3seeds.sh
```

### LHP-CZSL v3_text (UT-Zap, 3 seeds)

```bash
bash scripts/run_v3_text_utzap_3seeds.sh
```

### MIT-States baseline / variants

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --yml_path config/cluspro_baseline_mit_l14.yml
CUDA_VISIBLE_DEVICES=0 python train.py --yml_path config/lhp_czsl_v1_init_only_mit_l14.yml
CUDA_VISIBLE_DEVICES=0 python train.py --yml_path config/lhp_czsl_v2_mit_l14.yml
```

### Run the K-validation diagnostic

```bash
CUDA_VISIBLE_DEVICES=0 python tools/k_validation.py \
    --dataset_path <DATA_ROOT>/ut-zap50k \
    --sub_meanings_path data/sub_meanings_utzap.json \
    --output_dir logs/k_validation
# → prints Spearman ρ, exact-agreement %, and per-primitive (k_visual, k_llm)
```

## Where to read more

- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — full chronological experiment log, every run's metrics, debugging history (AMP NaN, invalid v3_text run, …)
- [`docs/model_design.md`](docs/model_design.md) — Stage 1 design rationale: variable K, L_sem, decorrelation
- [`docs/future_work.md`](docs/future_work.md) — prioritized open questions and ablation plans
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — method-version notes (v1 → v2 sub-meaning generation)

## Status

Research project, results pending. Not for production use. Tie-with-baseline result is the empirical state as of 2026-05; variable-K thesis under active diagnosis (see [§ Current state](#current-state)).
