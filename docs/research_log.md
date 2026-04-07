# LHP-CZSL 연구 로그

## 1. 프로젝트 개요

ClusPro baseline을 기반으로 LLM 지식을 활용하여 CZSL 성능을 개선하는 연구.

- **main branch**: ClusPro baseline (train/eval 포함, 바로 실행 가능)
- **lhp-czsl branch**: LLM-guided Hierarchical Prototype 모델 (Stage 1)

## 2. Baseline 실행

```bash
# Training
python train.py --yml_path config/cluspro_baseline_mit.yml

# Testing
python test.py --yml_path config/cluspro_baseline_mit.yml --load_model checkpoint/cluspro_baseline_b16_mit/val_best.pt
```

## 3. ClusPro Baseline 구성

- Visual Adapter (LoRA-style, ViT 매 블록)
- Attr/Obj Disentangler (MLP)
- Prototype Clustering (K=5, momentum update)
- Prototype Contrastive Loss (NCE) + HSIC Decorrelation
- 단일 Soft Prompt

## 4. 향후 연구 방향

LLM-guided Hierarchical Prototype (lhp-czsl 브랜치):
- Stage 1: Variable K + L_sem (하위 의미 기반)
- Stage 2: 계층 트리 + L_hier (long-tail 보완)

상세 설계는 `lhp-czsl` 브랜치의 `docs/` 참조.
