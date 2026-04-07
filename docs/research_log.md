# LHP-CZSL 연구 로그

## 1. 프로젝트 개요

**LHP-CZSL** (LLM-guided Hierarchical Prototype CZSL)
- CLUSPRO의 시각적 prototype 클러스터링 + LLM의 언어적 의미 계층을 결합
- 핵심 철학: LLM을 추론 경로에 넣지 않고, 훈련 전 지식 증류 도구로만 사용
- CLUSPRO의 zero inference cost 장점 유지

---

## 2. 단계적 개발 계획

### Stage 1 (우선 구현)
- LLM으로 각 primitive의 **하위 의미만** 생성
- 하위 의미 수 → primitive별 K 자동 결정
- 하위 의미 간 거리 → L_sem 손실
- CLUSPRO 대비 변경: variable K + L_sem 추가

### Stage 2 (추후, long-tail 해결 시)
- LLM으로 **상위 개념 그룹핑** 추가 (3단 이상 계층 트리)
- 상위 prototype: CLIP text + level-wise shared projection
- L_hier 손실 추가
- Long-tail primitive 성능 개선 목적

---

## 3. Stage 1 모델 설계

### 3.1 Phase 0: LLM 하위 의미 생성 (Offline, 1회)

```
Input:  primitive list (attrs + objs from dataset)
Output: sub_meanings.json

예시:
{
  "old":         {"sub": ["worn", "faded", "aged"],       "K": 3, "d_sem": {"worn-faded": 0.3, "worn-aged": 0.5, "faded-aged": 0.4}},
  "broken":      {"sub": ["cracked", "shattered"],        "K": 2, "d_sem": {"cracked-shattered": 0.6}},
  "translucent": {"sub": ["translucent"],                 "K": 1, "d_sem": {}},
  ...
}
```

**K 결정 규칙:**
- 하위 의미 수 = K (LLM이 생성한 수 그대로)
- 상한선 K_max = 5
- 희귀/단순 primitive → K=1~2
- 다양한 primitive → K=3~5

### 3.2 Architecture

```
Vision:
  CLIP ViT (frozen)
      ↓
  Visual Adapter (LoRA-style, per block)
      ↓
  f_global
      ├── attr_disentangler → f_attr
      └── obj_disentangler  → f_obj

Prototype System:
  attr별 K_a개 prototype (momentum update)
  obj별 K_o개 prototype (momentum update)
  → K는 primitive마다 다름 (LLM 결정)

Classification:
  comp_logit = max_k sim(f_global, p_k^comp)
  attr_logit = max_k sim(f_attr, p_k^attr)
  obj_logit  = max_k sim(f_obj, p_k^obj)
  최종: p(c_ij|x) + p(a_i|x)·p(o_j|x)  (CLUSPRO Eq.13)
```

### 3.3 Loss Functions

```
L = L_BAS + αL_PCL + βL_PDL + γL_sem

L_BAS:  3-path CE (comp, attr, obj) — CLUSPRO 동일
L_PCL:  Prototype Contrastive Loss — CLUSPRO 동일
L_PDL:  Cosine Decorrelation (HSIC 대체, 소배치 안정성)
L_sem:  Intra-Primitive Semantic Distance Alignment (신규)
```

**L_sem 상세:**
```
L_sem = Σ_p Σ_{i,j ∈ sub(p)} |sim(p_i, p_j) - d_sem(s_i, s_j)|²

- 같은 primitive 내 leaf prototype pair에만 적용
- d_sem: LLM이 Phase 0에서 제공한 하위 의미 간 거리
- K=1인 primitive는 pair 없으므로 자동 skip
```

### 3.4 CLUSPRO 대비 변경점

| | CLUSPRO | LHP-CZSL Stage 1 |
|---|---|---|
| K per primitive | 고정 (K=5) | LLM 결정 (K=1~5) |
| L_sem | 없음 | 신규 |
| Decorrelation | HSIC | Cosine |
| 나머지 | 동일 | 동일 |

### 3.5 기대 효과
- K 자동 결정으로 희귀 primitive의 prototype 분산 방지 (K=1)
- L_sem으로 prototype 간 배치가 의미적으로 정렬
- Cosine decorrelation으로 소배치 안정성 확보

---

## 4. Stage 2 설계 (메모)

Stage 1 결과 확인 후 진행. Long-tail 성능이 부족할 때:

- LLM으로 primitive → category → super-category 그룹핑
- Non-leaf prototype: CLIP text + level-wise shared projection (레벨당 Linear(d,d))
- L_hier: max(0, sim(f, p_parent) - sim(f, p_child) + margin)
- margin은 레벨별로 다르게 (상위일수록 작게)

---

## 5. 향후 작업

- [ ] Phase 0 LLM 프롬프트 설계 및 sub_meanings.json 생성
- [ ] CLUSPRO baseline에 variable K 구현
- [ ] L_sem 구현
- [ ] Cosine decorrelation 적용 (HSIC 대체)
- [ ] Stage 1 학습 및 평가
- [ ] Long-tail primitive별 성능 분석 → Stage 2 필요성 판단
