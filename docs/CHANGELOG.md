# LHP-CZSL 변경 이력

---

## v2 — K_min=3 + LLM 재생성 (2026-04-08)

### 배경

Stage 1 첫 실험(v1) 결과, ClusPro Baseline 대비 소폭 개선을 확인:
- ClusPro Baseline: AUC 0.1977, HM 0.3659
- LHP-CZSL v1: AUC 0.1997, HM 0.3691 (Best epoch 14)

하지만 v1에서 LLM이 대부분의 primitive에 K=1을 배정하여 (attrs 83%, objs 66%),
ClusPro의 핵심 강점인 multi-prototype 표현이 약화된 상태였다.
K 하한선을 두고 LLM에게 최소 3개의 시각적 하위 의미를 생성하도록 재요청.

### 변경사항

#### 1. 모델 코드 (`model/lhp_czsl.py`)

**cluster_min (K_min) 파라미터 추가:**
```python
# __init__에서
self.k_min = getattr(config, 'cluster_min', 1)

# _load_sub_meanings에서
k = max(min(info['K'], self.k_max), self.k_min)
```
config에서 `cluster_min`을 설정하면 LLM이 결정한 K가 이보다 작아도 최소 K_min개의 prototype을 사용한다.
v1에서는 cluster_min=1 (기본값), v2에서는 cluster_min=3.

**L_sem 배치 연산 최적화 (이전 커밋에서 수정됨):**
기존에 Python for-loop으로 360개 primitive를 순회하며 L_sem을 계산하여 539ms/step이 소요되었다.
d_sem 정보를 초기화 시 GPU 텐서로 캐싱하고, `torch.bmm`으로 배치 연산하여 16ms/step으로 34배 개선.
ClusPro Baseline과 동일한 학습 속도를 달성.

#### 2. LLM Sub-meanings 재생성 (`data/sub_meanings_mit_v2.json`)

LLM에게 "반드시 3개 이상의 시각적 하위 의미를 생성하라"는 프롬프트로 재요청.
단순히 K_min을 코드에서 강제하는 것이 아니라, LLM이 context-dependent한 시각적 차이를 고려하여 sub-meaning과 d_sem을 생성.

**K 분포 변화 (v1 → v2):**

| K | v1 (K>=1) | v2 (K>=3) |
|---|-----------|-----------|
| K=1 | 258개 (72%) | 0개 |
| K=2 | 80개 (22%) | 0개 |
| K=3 | 20개 (6%) | 335개 (93%) |
| K=4 | 2개 (0.6%) | 23개 (6%) |
| K=5 | 0개 | 2개 (0.6%) |
| **평균 K** | **1.4** | **3.1** |

**예시 변화:**
- `bright` (v1): K=1, sub=["bright"] → (v2): K=3, sub=["bright_reflective", "bright_luminous", "bright_vivid"]
- `apple` (v1): K=1, sub=["apple"] → (v2): K=3, sub=["red_apple", "green_apple", "sliced_apple"]

이로써 모든 primitive에 L_sem이 적용되며, prototype 간 의미 거리 정렬이 전체적으로 작동.

#### 3. Config (`config/lhp_czsl_v2_mit.yml`)

v1 config 대비 변경:
- `cluster_min: 3` 추가
- `sub_meanings_path`: `data/sub_meanings_mit.json` → `data/sub_meanings_mit_v2.json`
- `save_path`: `checkpoint/lhp_czsl_b16_mit` → `checkpoint/lhp_czsl_v2_b16_mit`

나머지 하이퍼파라미터는 v1과 동일 (lr, epochs, loss weights 등).

#### 4. Phase 0 생성 스크립트 (`scripts/generate_sub_meanings.py`)

변경 없음. 추후 API로 재생성 시 동일 스크립트 사용 가능.

### 실험 계획

v2 학습 후 v1과 비교:
- 전체 성능 (AUC, HM, Seen, Unseen)
- Per-primitive 성능 분석 (K=1이었던 primitive에서 K=3으로 바뀐 효과)
- L_sem의 기여도 변화 (v1에서는 K>=2인 22개 primitive에만 적용 → v2에서는 360개 전체)

---

## v1 — 초기 구현 (2026-04-07)

### 구현 내용

ClusPro Baseline 대비 3가지 변경:
1. **Variable K**: LLM이 결정한 primitive별 prototype 수 (K=1~5)
2. **L_sem**: 같은 primitive 내 prototype 간 의미 거리 정렬 손실
3. **Cosine Decorrelation**: HSIC 대체 (소배치 안정성)

### 파일 목록

- `model/lhp_czsl.py` — LHP-CZSL 모델 코드
- `config/lhp_czsl_mit.yml` — v1 config
- `data/sub_meanings_mit.json` — v1 LLM sub-meanings (K>=1)
- `data/sub_meanings_example.json` — JSON 포맷 예시
- `scripts/generate_sub_meanings.py` — LLM API 생성 스크립트 (Anthropic/OpenAI)
- `docs/model_design.md` — Stage 1 상세 설계서
- `docs/stage2_hierarchy.png` — Stage 2 계층 트리 다이어그램

### 실험 결과 (MIT-States, ViT-B/16, 15 epochs)

| 모델 | AUC | HM | Seen | Unseen |
|------|------|------|------|--------|
| ClusPro Baseline | 0.1977 | 0.3659 | 0.4431 | 0.5253 |
| **LHP-CZSL v1** | **0.1997** | **0.3691** | **0.4517** | 0.5220 |

Baseline 대비 AUC +0.002, HM +0.003, Seen +0.009.
