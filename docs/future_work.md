# 향후 작업 계획 (Future Work)

## 0. 현재 상태 요약 (MIT-States, ViT-B/16, Test 기준)

| Model | best_hm | AUC | Seen | Unseen |
|-------|:---:|:---:|:---:|:---:|
| ClusPro Baseline | 0.3362 | 0.1651 | 0.4261 | 0.4679 |
| Ablation (Cosine only, no L_sem, K=5 fixed) | **0.3385** | **0.1668** | 0.4315 | 0.4670 |
| **LHP-CZSL v1** (variable K, K_min=1) | 0.3365 | 0.1662 | 0.4311 | 0.4676 |
| LHP-CZSL v2 (K_min=3, LLM 재생성) | 0.3350 | 0.1657 | 0.4282 | 0.4692 |

### 핵심 관찰

1. **Ablation (cosine only)이 LHP-CZSL v1/v2보다 살짝 좋음**
   - L_sem과 variable K를 빼고 cosine decorrelation만 적용해도 거의 동일한 성능
   - 즉 **L_sem + variable K의 기여가 사실상 없음**

2. **v2 (K_min=3)이 v1 (K_min=1)보다 미세하게 하락**
   - LLM이 강제로 K=3 이상 만든 sub-meaning이 noise로 작용한 가능성
   - K=1로 충분한 primitive에 억지로 prototype을 늘리면 표현력 분산

3. **ClusPro Baseline 대비 개선폭이 작음**
   - 모든 변종이 baseline 근처 ±0.003 HM 범위
   - 통계적으로 유의미한 개선이라 보기 어려움

---

## 1. 우선순위: 높음

### 1.1 L_sem 작동 여부 정밀 분석
**문제**: L_sem이 의도대로 prototype 간 의미 거리를 정렬하고 있는지 불명확

- [ ] 학습된 prototype 간 cosine similarity가 LLM의 d_sem과 실제로 상관 있는지 측정
- [ ] L_sem 가중치(γ) sweep: 0.0 / 0.1 / 0.5 / 1.0 / 2.0
- [ ] L_sem만 적용하고 variable K는 빼는 ablation (K=5 고정 + L_sem)
- [ ] K>=2인 primitive에 한해서만 따로 성능 분석 (L_sem이 적용되는 대상)

**기대**: γ가 너무 작아서 효과가 묻히는 건지, 본질적으로 L_sem이 무의미한지 분리

---

### 1.2 LLM Sub-meaning 품질 검증
**문제**: LLM이 생성한 sub-meaning과 d_sem이 실제 시각적 차이를 반영하는지 미검증

- [ ] Sub-meaning을 사람이 직접 정의한 buckets로 교체 (예: 색깔, 재질, 상태 카테고리)
- [ ] d_sem을 균등하게(uniform) 두면 어떻게 되는지 (LLM 거리 vs uniform)
- [ ] 다른 LLM (Claude vs GPT-4 vs Gemini)으로 재생성하여 일관성 측정
- [ ] Sub-meaning의 시각적 구분 가능성을 CLIP 유사도로 사전 검증

**아이디어**: LLM이 단순히 "동의어 분리"만 하고 있을 가능성. 시각적으로 구분 가능한 sub-meaning만 남기는 필터링 필요할지도.

---

### 1.3 Stage 2 (계층 구조) 진입 판단
**현재**: Stage 1에서 baseline 대비 미미한 개선 → Stage 2로 갈 만큼 의미 있는지 결정 필요

- [ ] Per-primitive 성능 분석: long-tail (training instance가 적은) primitive에서 LHP-CZSL이 baseline보다 잘 하는지
- [ ] Long-tail에서 효과가 있다면 Stage 2 (super-category 그룹핑) 진행
- [ ] 효과가 없다면 다른 방향 (대안 1.4 참고)

**Stage 2 설계** (research_log.md 참고):
- LLM으로 primitive → category → super-category 그룹핑
- Non-leaf prototype: CLIP text + level-wise shared projection
- L_hier: max(0, sim(f, p_parent) - sim(f, p_child) + margin)

---

### 1.4 대안 방향 (Stage 1이 효과 없을 경우)
- [ ] **Backbone 키우기 (ViT-L/14)** → 백본 효과로 baseline 자체 끌어올리기
- [ ] **다른 LLM-guidance**: prototype 초기화에만 LLM 사용 (학습 중 L_sem 없음)
- [ ] **Inverse 접근**: LLM이 hard negative pair를 생성 → 강한 contrastive loss로 활용
- [ ] **Prototype 수 자체에 의미 부여하지 않기**: K 자동 학습 (gating 등)

---

## 2. 우선순위: 중간

### 2.1 데이터셋 확장
- [ ] UT-Zappos (단순, 빠른 검증)
- [ ] C-GQA (대규모, 일반화 검증)

각 dataset에서 ClusPro Baseline부터 우리 환경 수치 확보 → LHP-CZSL 효과 측정.

---

### 2.2 Loss 가중치 재튜닝
v1/v2 모두 ClusPro baseline의 loss weight를 그대로 사용. LHP-CZSL 전용 튜닝 필요.

- [ ] L_PCL (contrastive) 가중치 sweep
- [ ] L_PDL (cosine decorr) 가중치 sweep  
- [ ] L_sem 가중치 sweep (1.1과 함께)

---

### 2.3 학습 안정성 개선
- [ ] Seed 여러 개로 결과 분산 측정 (현재 seed 0만)
- [ ] Best epoch이 아닌 마지막 epoch 결과로도 비교 (ckpt selection bias 제거)
- [ ] Val/Test gap 분석 (현재 모든 모델 val→test 약 0.03 HM 하락)

---

## 3. 우선순위: 낮음 (탐색적)

### 3.1 Prototype 시각화
- [ ] 학습된 prototype을 t-SNE/UMAP으로 시각화
- [ ] LLM의 sub-meaning 라벨이 시각화에서 군집을 이루는지 확인
- [ ] Variable K가 prototype 다양성에 실제로 기여하는지

### 3.2 LLM cost-effectiveness
- [ ] LLM 호출 비용 vs 성능 향상 분석
- [ ] 다른 zero-shot 방법 (CLIP text encoder만 활용) 대비 LLM이 본질적으로 더 나은지

---

## 4. 환경/인프라

- [ ] **체크포인트 정리**: epoch별 .pt가 각 실험당 10+ → val_best, final만 유지 (현재 9.8GB+)
- [ ] **로그 표준화**: tqdm 출력이 train 로그에 섞여 있어 grep 불편 → epoch별 metric을 별도 파일로
- [ ] **재현성**: 학습 시 사용한 sub_meanings 버전, config, seed를 결과 파일에 자동 기록

---

## 5. 정량 목표

| 지표 | 현재 (ClusPro) | 1차 목표 | 2차 목표 |
|------|:---:|:---:|:---:|
| best_hm (Test) | 0.3362 | 0.345+ | 0.36+ |
| AUC (Test) | 0.1651 | 0.175+ | 0.19+ |

**1차 목표**: Stage 1 ablation으로 LHP-CZSL의 기여 명확히 입증 (현재는 baseline 수준)
**2차 목표**: Stage 2 또는 백본 업그레이드로 의미 있는 SOTA 도달

---

## 6. 추천 진행 순서

ROI 높은 순서:
1. **L_sem 분석 (1.1)** — 현재 핵심 의문. 효과 있는지 확실히 확인
2. **Per-primitive 분석 (1.3 일부)** — Long-tail 효과로 Stage 2 진입 결정
3. **Sub-meaning 품질 검증 (1.2)** — LLM 의존성의 본질 점검
4. **Loss tuning (2.2)** — 위 결과 후 마지막 미세 조정
5. **Stage 2 또는 대안 (1.3, 1.4)** — Stage 1 결론 후 결정

---

## 7. 현재 미해결 핵심 질문

1. **L_sem이 정말 효과가 있는가?** (Ablation cosine only가 거의 동등)
2. **Variable K가 효과가 있는가?** (K_min=3인 v2가 v1보다 떨어짐)
3. **LLM sub-meaning이 시각적으로 의미 있는가?** (검증 자체가 안 됨)
4. **ClusPro 대비 0.003 HM 향상이 통계적으로 유의미한가?** (다중 seed 필요)
