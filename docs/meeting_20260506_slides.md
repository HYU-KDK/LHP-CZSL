# Advisor 미팅 슬라이드 자료 — 2026-05-06 (수)

작성: 2026-05-05 22:00 KST → 재구성: 2026-05-06 (narrative flow)
구조: **시간 순 흐름** — ClusPro → v2 → v1 → v3_text → Phase 1 → 다음 axis
예상 발표: 14~17분 + Q&A
보조 자료: `RESEARCH_LOG.md`, `textpair.png`, `UT-Zap결과.png`

---

## Slide 1 — Title

### LHP-CZSL: LLM 진입 지점 검증 한 달 라운드

발표자: 동기 (2026-05-06 advisor 미팅)

> 한 달간 LLM의 다섯 가지 진입 지점을 차례로 검증했고, 그 흐름을 따라 thesis axis 재정의를 제안.

---

## Slide 2 — TL;DR

### 한 달 라운드의 흐름

```
ClusPro baseline (K=5 고정)
    │
    ▼
v2: LLM이 K 정함        → ❌ tie
    │
    ▼
v1: K_p + sub-meaning aux loss → ❌ tie
    │
    ▼
v1_init_only: K_p만        → ❌ tie
    │
    ▼
v3_text: text encoder INPUT 풍성화 → △ borderline +1.5pp HM
    │
    ▼
Phase 1: LLM post-hoc feasibility mask → ❌ −0.015 AUC (7σ)
    │
    ▼
[다음 axis 결정 中] description prompt / image-conditional VLM
```

**4개 negative + 1개 fragile small** → "**LLM 영향력이 prototype/post-hoc filter 같은 외곽에선 죽고, CLIP representation 안쪽으로 들어가야 신호가 잡힘**" 패턴 확인.

**speaker note**: "오늘 흐름을 빠르게 보고 들어가실게요. LLM의 다섯 진입 지점을 차례로 막다른 길까지 가본 결과예요."

---

## Slide 3 — 출발점: ClusPro Baseline

### 기존 모델의 작동 방식

CLIP ViT-L/14 backbone + primitive별 prototype 메모리.

UT-Zap 예시: 16 attr × 12 obj. 각 primitive마다 **K=5개의 768차원 prototype vector**.

```
"Suede"      메모리 = [vec_0, vec_1, vec_2, vec_3, vec_4]
"Sandals"    메모리 = [vec_0, vec_1, vec_2, vec_3, vec_4]
...총 28×5 = 140개 prototype
```

**학습**: 이미지가 들어오면 그 primitive의 K=5 prototype 중 가장 가까운 것 EMA로 업데이트
**추론**: 이미지 ↔ 모든 prototype cosine similarity → 가장 가까운 (attr, obj) 선택

### 우리의 reference (UT-Zap, post-Gumbel-fix 3-seed)
- closed-world test HM **0.5374** / AUC **0.4131**
- 이게 모든 LHP variant 비교의 기준선

**speaker note**: "ClusPro는 모든 primitive에 K=5 prototype을 무조건 두는 모델이에요. 우리 reference 성능은 test HM 0.5374."

---

## Slide 4 — Hypothesis 1: "K=5는 낭비, LLM이 K를 정해주자" (v2)

### 직관
- **"Sandals"** = flat / sport / dress 등 시각 sub-type 다양 → K 많이 필요
- **"Hair.Calf"** = 매우 specific → K=1로 충분

→ "primitive별로 K를 다르게 줘야 효율적이다"

### 어떻게 LLM에게 K를 받나
LLM(Gemini)에게 각 primitive별 sub-meaning을 enumerate시킴:

```json
"Faux.Fur": {"K": 1, "subs": ["faux fur"]}
"Leather":  {"K": 2, "subs": ["smooth leather", "patent leather"]}
"Sandals":  {"K": 2, "subs": ["flat sandals", "sport sandals"]}
```

UT-Zap 결과: 13 attr K=1 / 3 attr K=2; 8 obj K=1 / 4 obj K=2.

### 코드 구현 — "버퍼는 K=5 그대로, 마스킹으로 처리"

PyTorch 텐서가 균일 shape이어야 batched ops 가능 → **버퍼는 K_max=5 통일**, primitive별 K_p만 따로 저장하고 학습/추론 시 첫 K_p 슬롯만 active로 사용.

```
"Leather" (K_p=2): [active_0, active_1, --random, never updated--, --random--, --random--]
"Faux.Fur" (K_p=1): [active_0, --random--, --random--, --random--, --random--]
```

→ K_p 마스킹 = primitive별 prototype 수를 LLM이 줄여줌

**speaker note**: "원래 LHP의 첫 thesis가 이거였어요. K=5 강제는 일률적이니까 LLM이 primitive 의미에 맞게 K를 정해주자."

---

## Slide 5 — v2 결과 → ❌ K는 lever 아님

### v2 결과 (mit-states, 4-28)
- HM **0.3897** / AUC 0.2179
- baseline (K=5): HM 0.3893 / AUC 0.2169
- → **시드 노이즈 안의 tie**

### Sanity check — "LLM 없이 그냥 K=3으로 줄여도 되는 거 아닌가?" (4-29)

| 변형 | K | mit HM | mit AUC |
|---|---|---|---|
| ClusPro baseline | 5 | 0.3893 | 0.2169 |
| ClusPro K=3 ablation | 3 | 0.3855 | 0.2158 |
| LHP v2 (K_LLM, mostly K=3) | 1~5 | 0.3897 | 0.2179 |

→ 셋 다 tie. **K 값이 무엇이든 비슷한 성능**.

### 첫 결론 (negative finding 1)
> **"LLM이 K를 결정해주는 것 자체는 contribution 아님. prototype 개수 K는 CZSL 성능의 lever가 아니다."**

→ 원래 thesis "variable K" **폐기**.

**speaker note**: "K를 LLM이 정하든, 그냥 3으로 내리든, 5로 두든 다 똑같았어요. K 자체가 lever 아니라는 게 첫 negative finding입니다."

---

## Slide 6 — Pivot 1: "K가 아니라 sub-meaning 의미 정보가 lever"

### 새 가설
LLM이 출력한 건 **K 숫자만 아님**. sub-meaning의 **이름들** ("smooth leather", "patent leather")도 함께 나옴. 이 텍스트 정보를 활용 안 한 거 아닌가?

### v1 — 두 추가 loss 도입

**(a) L_sem — semantic distance 보존**

LLM이 알려주는 sub-meaning 간 의미 거리:
- "smooth leather" ↔ "patent leather" 거리 0.3 → prototype 간 cosine 0.7 되도록 강제
- "flat sandals" ↔ "sport sandals" 거리 0.7 → prototype 간 cosine 0.3 되도록 강제

`d_sem` 매트릭스로 미리 계산해 두고 학습 중 prototype 간 cosine을 (1 − d_sem)에 맞게 끌어당김. `sem_weight=0.05`.

**(b) L_decorr — prototype 직교화**

같은 primitive의 K_p prototype이 collapse하지 않도록 cosine decorrelation. ClusPro 원작 HSIC 대체. `decorr_weight=0.1`.

→ "K_p 마스킹 + L_sem + L_decorr" 가 v1.

**speaker note**: "K 자체가 lever 아니라는 게 확인됐으니까, LLM이 알려준 sub-meaning 의미를 prototype 간 거리로 강제하는 두 loss를 추가했어요."

---

## Slide 7 — v1 / v1_init_only 결과 → ❌ aux loss도 lever 아님

### v1 결과 (mit-states, 5-1)
- HM 0.3886 / AUC 0.2182, baseline 대비 **tie**

### "두 loss 중 어느 게 효과 있나? 아니면 둘 다 무용?" — ablation으로 v1_init_only

`sem_weight=0`, `decorr_weight=0` (둘 다 0 처리). K_p 마스킹만 남김.

| 변형 | mit HM | utzap test HM (3-seed) |
|---|---|---|
| baseline | 0.3893 | 0.5374 |
| v1 (full) | 0.3886 | (UT-Zap 미실행) |
| **v1_init_only** | **0.3874** | **0.5458** |

mit/utzap 모두 baseline 대비 시드 std 안. **Aux loss 빼도 같은 성능** = aux loss가 기여 안 함.

### 두 번째 결론 (negative finding 1 강화)
> **"prototype 메커니즘 안에서 LLM이 어디로 들어와도 (K 결정, sub-meaning 거리, K_p 마스킹) 모두 tie"**

→ visual side prototype-level 개입은 dead end로 확정.

**speaker note**: "L_sem과 L_decorr를 둘 다 끄고 K_p 마스킹만 남겨도 같은 성능. 즉 prototype 메모리 쪽에 LLM이 어떻게 들어와도 lever 아님이 확인됐어요."

---

## Slide 8 — Pivot 2: "Visual side 안 되니 Text side로" (v3_text)

### 새 진입 지점

지금까지 LLM은 **prototype 메모리 (visual side)**에만 영향:
- K 결정 → 어떤 슬롯 active?
- L_sem → prototype 간 거리?

→ 다른 입구: **CLIP text encoder의 INPUT** 자체.

### v3_text 메커니즘 (`text_ensemble=True`)

**기존 (v1_init_only 포함)**: 각 primitive마다 prompt 1개
```
"a photo of [Suede]"  → CLIP text encoder → 1 vector
```

**v3_text**: 각 primitive당 K_p개 sub-meaning을 prompt에 넣고 평균
```
"Leather" (K_p=2):
  "a photo of [smooth leather]" → vec_0
  "a photo of [patent leather]" → vec_1
  → 평균이 "Leather" representation
```

총 sum_Ka + sum_Ko = 19+16 = 35 prompt → primitive별로 평균 → 28개 text feature.

→ **LLM의 sub-meaning 이름이 CLIP text encoder의 INPUT 자체로 들어감**. prototype 메모리는 v1_init_only와 동일.

**speaker note**: "이번엔 prototype 쪽이 아니라 CLIP text encoder가 받는 INPUT 자체에 LLM 정보를 넣는 시도예요."

---

## Slide 9 — v3_text 결과 → △ fragile small gain

### Closed-world (UT-Zap, 3-seed, val_best.pt @ val_pairs HM peak)

| variant | HM (mean ± std) | AUC (mean ± std) | vs baseline |
|---|---|---|---|
| baseline v2 | 0.6694 ± 0.001 | 0.5201 ± 0.004 | — |
| v1_init_only | 0.6646 ± 0.013 | 0.5113 ± 0.020 | tie |
| **v3_text** | **0.6735 ± 0.005** | **0.5272 ± 0.004** | **+0.4 / +0.7 pp** |

### Test_pairs convention (CZSL paper standard)

| variant | TEST HM | TEST AUC | vs baseline |
|---|---|---|---|
| baseline v2 | 0.5374 | 0.4131 | — |
| v1_init_only | 0.5458 | 0.4216 | +0.8 / +0.9 pp |
| **v3_text** | **0.5523** | **0.4312** | **+1.5 / +1.8 pp** |

### 통계 정직 보고
- paired t-test (n=3, df=2):
  - HM: t = 4.31, **p ≈ 0.05** (borderline)
  - AUC: t = 1.86, **p ≈ 0.20** (NOT 유의 — seed 1 outlier가 driving)
- **3/3 seed 모두 baseline 우위 방향**, 격차 fragile

### 평가
> **"처음으로 양의 방향 신호가 나옴, 다만 격차는 시드 std 경계. main claim 못 들고 가지만 부정도 못 함."**

→ "더 검증할까 vs 다른 LLM 진입 지점 시도할까" 분기점.

**speaker note**: "처음으로 baseline 위로 올라간 신호가 나왔는데, 격차가 작아서 main으로 못 들고 가요. n=3 한계. 그래서 더 검증보다 다른 LLM 진입 지점을 시도하기로 했어요."

---

## Slide 10 — 또 다른 진입 지점 시도: Phase 1 (LLM feasibility post-hoc filter)

### 직관 — FLM (ECCV-W 2024) 식 접근

LLM에게 "이 (attr, obj) 페어가 시각적으로 그럴듯한가" 0~10점 받고, 추론 단계에서 nonsense 페어 마스킹.

```
LLM(Gemini Flash): (Suede, Slippers) → 8점 (그럴듯)
                   (Suede, lake)    → 0점 (말이 안 됨)
```

### 셋업
- mit-states 28,175 + UT-Zap 192 페어 LLM 점수화 (총 ~24h)
- `data/feasibility_{dataset}_llm.pt` 변환
- `test.py`에 `--feasibility_path` 플래그 추가
- **9 ckpt × {mask off, mask on} = 18 eval** (UT-Zap, 3-seed × 3 모델)
- **장점**: 학습 변경 없음, ~10분 만에 결과. v3_text 추가 검증보다 cheaper + 더 많은 정보

**speaker note**: "v3_text 추가 검증보다 cheaper한 다른 LLM 진입 지점 — inference 단계 post-hoc filter — 시도했어요. 학습 안 건드리고 10분이면 끝나니까."

---

## Slide 11 — Phase 1 결과 → ❌ post-hoc filter도 lever 아님

### Mask 기여도 (mask on − mask off, 3-seed, test_pairs)

| variant | val ΔHM | val ΔAUC | test ΔHM | **test ΔAUC ± std** | 통계 신뢰도 |
|---|---|---|---|---|---|
| baseline v2 | +0.002 | +0.003 | −0.013 | **−0.0151 ± 0.002** | **≈ 7σ** |
| v1_init_only | +0.001 | +0.001 | −0.011 | −0.0130 ± 0.011 | ≈ 1σ (noisy) |
| v3_text | +0.001 | +0.001 | −0.015 | **−0.0164 ± 0.004** | **≈ 4.5σ** |

### 패턴
- val에서 미세 + (threshold가 val AUC max로 선택됐으니 당연)
- **test에선 모든 모델 일관되게 −0.015 ± 0.002** (baseline 7σ, v3 4.5σ)
- baseline에서도 같은 패턴 → mask 자체의 일반화 한계, 모델 변경 문제 아님

### 두 번째 robust negative finding
> **"image-agnostic LLM feasibility는 val→test transfer 실패. inference 단계 LLM 후처리도 lever 아님."**

FLM류 한계를 우리 데이터로 직접 재현.

**speaker note**: "결과는 깔끔한 negative — 모든 3 모델에서 mask가 test AUC를 일관되게 −0.015 깎아요. baseline 기준 7σ로 robust한 negative finding입니다."

---

## Slide 12 — 패턴 종합

### 한 달간 시도한 5개 LLM 진입 지점

| LLM이 어디로 들어가나 | 변형 | 결과 |
|---|---|---|
| K 결정 (prototype 개수) | v2 | ❌ tie |
| Sub-meaning 거리 (prototype 의미 위치) | v1 (L_sem + L_decorr) | ❌ tie |
| K_p 마스킹만 | v1_init_only | ❌ tie |
| **Text encoder INPUT 풍성화** | **v3_text** | **△ +1.5pp HM (fragile)** |
| Inference 단계 post-hoc filter | Phase 1 | ❌ −0.015 AUC (7σ) |

### 패턴
- **prototype 메모리 쪽 LLM 개입 모두 tie** (v2, v1, v1_init_only)
- **Inference 단계 LLM 후처리 negative** (Phase 1)
- **Text encoder INPUT만 양의 방향 신호** (v3_text)

### 자연스러운 해석
> **"LLM 영향력이 prototype 메모리 / post-hoc filter 같은 외곽에 들어갈 땐 모두 죽고, CLIP의 representation 흐름 안쪽 (text encoder INPUT)에 들어갈 때만 작은 신호가 잡힘."**

→ **다음 axis는 text encoder INPUT을 더 풍성화하거나, image-conditional 흐름을 더 깊이 만드는 것**.

**speaker note**: "5개 진입 지점을 정리하면 패턴이 보여요. LLM이 모델 깊은 곳, CLIP representation 안쪽으로 들어갈수록 신호가 잡혀요."

---

## Slide 13 — 다음 axis: v3_text refinement

### v3_text가 유일한 신호 잡힌 진입 지점 → "조금 더 만져보기"

v3_text의 +1.5pp가 작은 이유 진단:
- **Sub-meaning이 너무 짧음** ("flat sandals" 두 단어)
- **K_p=1 primitive가 75%** (UT-Zap 28개 중 21개) → ensemble이 사실상 baseline과 동일
- **단순 mean pooling** — 이미지와 무관

→ 3개 lever를 차례로 실험해서 v3_text 신호를 정량화.

### Phase A — Lever 1+2 minimal pilot (3일)

**Lever 1**: sub-meaning을 LLM **visual description**으로 교체
```
"Leather": ["smooth fine-grained leather with subtle natural texture, deep brown or black tones, often supple and matte finish",
            "glossy patent leather with high-shine polished surface, reflective and slick"]
```

**Lever 2**: 모든 primitive에 **K_p ≥ 3** 강제 — 28개 모두에서 진짜 ensemble 작동

| Day | 작업 |
|---|---|
| 1 | LLM description 생성 (28 primitive × K_p=3 = 84 desc, Gemini Flash ~$0.5) |
| 1 | 새 `data/descriptions_utzap.json` (기존 sub_meanings 형식 유지) |
| 2 | v3_text 학습 (코드 수정 0줄, JSON path만 변경), 3-seed |
| 3 | 평가: description-v3_text vs 기존 v3_text 직접 비교 |

### Phase B — Lever 4 image-conditional selection (3-4일) ★차별화

현재 mean pooling을 **image-aware top-k selection**으로 교체:
```
test image x → CLIP visual → vec_x
descriptions {d_1, ..., d_K} → CLIP text → {vec_d_1, ..., vec_d_K}
similarity(vec_x, vec_d_i) → top-1 또는 softmax weighted
이미지마다 다른 description으로 primitive feature 결정
```

→ **axis B (VLM image-conditional) 효과를 LLM only cost로 달성**. CDS-CZSL 등 선행 연구와의 차별점.

| Day | 작업 |
|---|---|
| 4 | `model/lhp_czsl.py`에 image-conditional pooling 구현 |
| 5 | 학습/추론 변형 시도 (학습 mean / 추론 image-cond, 또는 학습부터 image-cond) |
| 6-7 | 3-seed 평가 + closed/open-world 둘 다 + 분석 |

### 1주 끝 분기 결정 (binary outcome)

| Phase A | Phase B | 결론 |
|---|---|---|
| ❌ | ❌ | "v3_text +1.5pp는 진짜 노이즈" → axis G (negative paper framing) 본격화 |
| ❌ | ✅ | image-conditional selection이 lever (axis B 본질, LLM only) |
| ✅ | ✅ | description content + image-conditional 둘 다 lever, 자연스러운 main paper |
| ✅ | ❌ | description content가 lever, 차별점은 axis G로 보강 필요 |

### 위험
- CLIP 77 token limit → description 50 token 이내 압축
- LLM hallucination → Phase A 후 quality filter 도입
- 학습 시간 sum_K 28→84로 3배 → UT-Zap 7h → 10h, 1주 안 가능

**speaker note**: "v3_text가 유일한 양의 신호니까 거기를 만져봅니다. Phase A는 description content가 lever인지, Phase B는 image-conditional structure가 lever인지 따로 측정해서 1주 안에 binary 결론 가져옵니다."

---

## Slide 14 — Advisor 질문

### Q1. Negative finding stack으로 thesis axis 재정의 정당화 충분한가?

5개 LLM 진입 지점 중 4개 negative + 1개 fragile borderline. "LLM 영향력이 representation 안쪽일수록 신호 강함"이라는 패턴 해석이 합리적인지.

### Q2. v3_text의 fragile +1.5pp gain을 어떻게 다룰지

(a) 더 seed 늘려 검증 (시간 비용 ↑)
(b) axis E (description prompt) 안의 ablation chapter로 흡수
(c) "방향 일관, 격차 fragile"로 보조 보고만

### Q3. v3_text refinement 1주 plan 적절한가? (Slide 13)

제안 plan:
- Phase A (3일): description 내용 풍성화 + K_p≥3 강제
- Phase B (3-4일): image-conditional description selection (차별화 포인트)

advisor가 검토할 점:
- Phase A vs B 순서 — content 먼저 vs structure 먼저?
- description 1차 생성에 시각 axis 5개 (texture/color/material/shape/context) 사용 적절?
- Phase B의 image-conditional selection이 axis B (VLM)와 본질 같은지

대안 — Axis B (VLM image-conditional) 직진:
| | v3_text refinement (E) | VLM (B) |
|---|---|---|
| 비용 | LLM only, 1주 | VLM API, 1~2주 |
| 우리 인프라 | v3_text 그대로 재사용 | 새 셋업 |
| 차별화 | Phase B의 image-conditional selection | 자체로 새 axis |
| 위험 | CDS-CZSL과 유사 (Phase B 차별화 필수) | API 비용 + 셋업 |

### Q4. FLM 후속작과 차별화 — 어느 axis가 가장 깨끗한가?

- (E) Description prompt 풍성화 — text encoder INPUT 변화
- (B) Image-conditional VLM — per-image feasibility 새 axis
- 우리가 갖고 갈 main contribution을 어디에 잡을지

**speaker note**: "이 4개가 advisor 답 듣고 결정해야 하는 핵심이에요. 특히 Q3가 다음 1주 시간 사용처 결정합니다."

---

## Slide 15 — Appendix

### A. K_p 메커니즘 상세 (Q&A 대비)

```
Buffer shape: (5, 768) per primitive — 균일 (batched ops)
attr_k[i]    = K_p (LLM JSON으로 결정, [k_min=1, k_max=5] clamp)
slots [0:K_p] = active (similarity / EMA / contrastive loss 참여)
slots [K_p:5] = unused (torch.randn 초기값 그대로, 학습 중 절대 안 건드림)
```

### B. Diagnostic — 왜 baseline post-fix eff_K가 회복됐나
- Pre-Gumbel-fix baseline: soft-Gumbel → prototype collapse, eff_K 1.06–1.37
- Post-Gumbel-fix (hard one-hot): eff_K 2.21–2.66 (회복)
- LHP variants eff_K 3.89/3.74 — 단 caveat: K_p<5인 primitive에선 unused random 슬롯이 SVD에 기여, 실제 학습된 prototype은 K_p=1~2개

### C. FLM 비교

| | FLM (ECCV-W 2024) | 우리 (Phase 1) |
|---|---|---|
| LLM | Vicuna / ChatGPT | Gemini 2.5-flash-lite |
| 출력 | yes/no logit | 0~10 정수 |
| 통합 | post-hoc threshold | post-hoc threshold (`test.py:508`) |
| 결과 | UT-Zap SOTA 갱신 주장 | 모든 baseline 일관되게 −0.015 AUC |

### D. 모든 자산

| 자산 | 경로 |
|---|---|
| Research Log | `RESEARCH_LOG.md` (§1, §5, §6.1~6.3) |
| 체크포인트 | `checkpoint/{baseline_v2, v1_init_only, v3_text}_l14_utzap_seed{0,1,2}/` |
| Open-world raw | `checkpoint/{run}/val_best.open.{masked,unmasked}.json` |
| Feasibility .pt | `data/feasibility_{mit-states,ut-zap50k}_llm.pt` |
| Diagnostic | `logs/k_validation/prototype_diagnostic_20260505_211757.json` |
| Pilot logs | `logs/openworld_pilot/master_*.log` |

---

## 발표 흐름 요약 (cheat sheet)

```
[Slide 1-2]   Title + TL;DR (2분)
[Slide 3]     ClusPro baseline 출발점 (1분)
[Slide 4]     Hypothesis 1: v2 — LLM이 K 정함 (1.5분)
[Slide 5]     v2 결과 → ❌ K는 lever 아님 (1.5분)
[Slide 6]     Pivot 1: sub-meaning 의미 활용 (v1) (1.5분)
[Slide 7]     v1/v1_init_only → ❌ aux loss도 lever 아님 (1.5분)
[Slide 8]     Pivot 2: text side로 (v3_text 메커니즘) (1.5분)
[Slide 9]     v3_text → △ fragile +1.5pp HM (2분)
[Slide 10]    Phase 1 (post-hoc filter) 셋업 (1분)
[Slide 11]    Phase 1 → ❌ −0.015 AUC 7σ (1.5분)
[Slide 12]    패턴 종합 — "안쪽일수록 신호" (1분)
[Slide 13]    다음 axis 제안 + 1주 plan (1.5분)
[Slide 14]    Q1-Q4 advisor 질문 (Q&A 동안)
[Slide 15]    Appendix
```

총 15 슬라이드, 발표 17~18분 + Q&A.

핵심 메시지:
> "한 달간 LLM의 5개 진입 지점을 차례로 시도했고, 4개 negative + 1개 fragile small. 패턴은 'LLM 영향력이 CLIP representation 안쪽일수록 신호가 잡힘'. 다음 axis는 text encoder INPUT을 description으로 풍성화 (axis E) 또는 image-conditional VLM (axis B)."
