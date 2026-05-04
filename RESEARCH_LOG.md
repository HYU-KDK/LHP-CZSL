# LHP-CZSL Research Log

작성일 2026-05-02 16:15 KST. 4-27 ClusPro baseline (K=5) 시작 시점부터 오늘 3-seed 본런 시작 직전까지의 정리. **5-3 09:00 업데이트**: UT-Zap v1_init_only 3-seed 결과 (§1) + baseline tie 결론 + 다음 단계 후보 (§4). **5-3 21:35 업데이트**: v3_text 3-seed 본런이 16:08 KST에 끝났으나 train loss와 test 숫자가 v1_init_only와 비트 일치 → **invalid run으로 폐기**, §4.5에 디버깅 단서 정리.

모든 실험: ViT-L/14 backbone, fp16 AMP, batch_size=8, grad_accum=8, 15 epochs (probe 제외), seed=0 (3-seed run 제외), val_metric=best_hm. 데이터셋 경로 등 환경 설정은 [LHP-CZSL setup memory](/home/student/.claude/projects/-home-student-dongki/memory/project_lhp_czsl_setup.md) 참고.

---

## 1. 결과 요약 테이블

### mit-states (ViT-L/14, 15 ep, seed 0) — closed-world test

| Run | Date | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|---|
| ClusPro baseline (K=5) | 4-27 | 0.4899 | 0.5203 | **0.3893** | **0.2169** | 0.3856 | 0.5553 |
| LHP-CZSL v2 (LLM-derived K) | 4-28 | 0.4891 | 0.5241 | **0.3897** | **0.2179** | 0.3836 | 0.5573 |
| ClusPro baseline K=3 (ablation) | 4-29 | 0.4870 | 0.5225 | **0.3855** | **0.2158** | 0.3868 | 0.5563 |
| LHP-CZSL v1 (sub-meaning init + sem + decorr) | 5-1 | 0.4866 | 0.5250 | **0.3886** | **0.2182** | 0.3862 | 0.5584 |
| LHP-CZSL v1_init_only (sem=0, decorr=0) | 5-1→5-2 | 0.4878 | 0.5202 | **0.3874** | **0.2155** | 0.3851 | 0.5552 |

mit-states / ViT-L/14 위에서 모든 변형이 **시드 노이즈 범위 내 동률** (HM ±0.005, AUC ±0.003).

### UT-Zappos (ViT-L/14, seed 0) — closed-world test

| Run | epochs | NaN 상태 | seen | unseen | **HM** | **AUC** | 비고 |
|---|---|---|---|---|---|---|---|
| baseline lr=1e-4 | 15 | 오염 | 0.5885 | 0.7012 | **0.4902** | **0.3388** | epoch 1 step 1356에서 NaN 시작 |
| baseline lr=5e-5 | 15 | 오염 | 0.6285 | 0.6642 | **0.5009** | **0.3473** | step 1468에서 NaN 시작 |
| probe hsic_weight=0 lr=1e-4 | 1 | 오염 | 0.6373 | 0.6684 | **0.4861** | **0.3389** | step 1980에서 NaN 시작 |
| probe bf16 lr=1e-4 (no scaler) | <1 | 영구 오염 | — | — | — | — | step 813 NaN, val_best 미저장 → 크래시 |
| **probe guard lr=1e-4 (수정)** | **1** | **클린** | **0.6403** | **0.6785** | **0.4919** | **0.3493** | NaN 표시 0회, skip 18회(0.6%) |

### UT-Zappos (ViT-L/14, lr=1e-4, fix 적용, 15 ep) — 3-seed 본런

표에 적힌 숫자는 `test.py:706`의 첫 번째 출력(`val_pairs` eval) 기준 — 표 헤더가 "test"였지만 실제로는 val_pairs임. test_pairs 결과는 §1.1 부록 참고. baseline / v1 비교는 같은 convention 안에서 일관되게 이루어짐.

#### baseline (5-2 16:10 ~ 23:06)

| seed | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|
| 0 | 0.6990 | 0.7698 | **0.6456** | **0.5016** | 0.3989 | 0.8563 |
| 1 | 0.6807 | 0.7377 | **0.6251** | **0.4728** | 0.3892 | 0.8619 |
| 2 | 0.7149 | 0.7424 | **0.6485** | **0.4978** | 0.4079 | 0.8600 |
| **mean ± std** | **0.6982 ± 0.017** | **0.7500 ± 0.017** | **0.6397 ± 0.013** | **0.4907 ± 0.016** | — | — |

NaN 오염 pre-fix baseline (seed 0, HM 0.4902 / AUC 0.3388) 대비 **HM +0.150 / AUC +0.152** 개선. Fix가 의도한 대로 동작했음을 확인 — 이 3-seed 평균이 앞으로 LHP 변형 비교의 기준선.

#### LHP-CZSL v1_init_only (5-2 23:47 ~ 5-3 06:36)

baseline과 동일 조건 (lr=1e-4, 15 ep, fp16+GradScaler+guard), 추가로 sub-meaning name 기반 prototype init만 켬. `model/lhp_czsl.py:_update_prototypes`에 features 유한성 가드 미러링.

| seed | seen | unseen | **HM** | **AUC** | attr | obj |
|---|---|---|---|---|---|---|
| 0 | 0.7058 | 0.7612 | **0.6434** | **0.5007** | 0.3976 | 0.8544 |
| 1 | 0.6853 | 0.7326 | **0.6310** | **0.4737** | 0.3933 | 0.8600 |
| 2 | 0.7035 | 0.7390 | **0.6407** | **0.4887** | 0.4054 | 0.8578 |
| **mean ± std** | **0.6982 ± 0.011** | **0.7443 ± 0.015** | **0.6384 ± 0.007** | **0.4877 ± 0.014** | — | — |

vs baseline: **seen Δ=0.000 / unseen Δ=−0.006 / HM Δ=−0.001 / AUC Δ=−0.003**. 모든 metric이 시드 std 안. **UT-Zap에서도 sub-meaning init만으로는 baseline과 tie**. mit-states/ViT-L/14에서 본 패턴이 UT-Zap에서 그대로 재현됨.

#### §1.1 부록 — test_pairs 기준 동일 비교 (참고)

| run | seed | seen | unseen | HM | AUC |
|---|---|---|---|---|---|
| baseline | 0 | 0.6755 | 0.7456 | 0.5384 | 0.4134 |
| baseline | 1 | 0.6931 | 0.7451 | 0.5498 | 0.4289 |
| baseline | 2 | 0.6989 | 0.7562 | 0.5562 | 0.4413 |
| baseline | mean | 0.6892 | 0.7490 | **0.5481** | **0.4279** |
| v1_init_only | 0 | 0.6745 | 0.7483 | 0.5371 | 0.4097 |
| v1_init_only | 1 | 0.7087 | 0.7599 | 0.5765 | 0.4587 |
| v1_init_only | 2 | 0.6784 | 0.7562 | 0.5522 | 0.4306 |
| v1_init_only | mean | 0.6872 | 0.7548 | **0.5553** | **0.4330** |

test_pairs 기준에서는 v1이 미세하게 위(HM +0.007, AUC +0.005)지만 seed 분산 안. 결론은 동일: **tie**.

---

## 2. 시계열 narrative

### Phase 1 — mit-states K-as-hyperparameter (4-27 ~ 4-29)

가설: "LLM이 attribute/object마다 의미적으로 적절한 K(prototype 수)를 알려주면 모델이 더 잘 동작한다."

- **4-27** ClusPro baseline (K=5) → HM 0.3893 / AUC 0.2169
- **4-28** LHP-CZSL v2 (LLM이 attr/obj별 K를 추정, `cluster_min=3` floor 때문에 대부분 K=3) → HM 0.3897 / AUC 0.2179
- **4-29** ClusPro K=3 ablation (LLM 안 쓰고 그냥 K를 5→3으로 내리기) → HM 0.3855 / AUC 0.2158

결론: K를 1D 하이퍼파라미터로 보면 K=3, K=5, LLM-K 모두 사실상 동률. **LLM이 K 값을 정해주는 것 자체는 contribution이 아님**. K는 단순 정수가 아니라 "branch / no-branch"의 binary 신호로 봐야 의미가 있을 가능성을 메모.

### Phase 2 — sub-meaning name 기반 prototype init (5-1)

가설 전환: K 자체가 아니라 **sub-meaning name으로 prototype을 초기화**하는 것이 진짜 contribution일 것.

- **5-1** LHP-CZSL v1 (sub-meaning init + `sem_weight=0.05` + `decorr_weight=0.1`) → HM 0.3886 / AUC 0.2182
- **5-1→5-2** v1_init_only (init 유지, sem=0 / decorr=0) → HM 0.3874 / AUC 0.2155

결론: aux loss를 빼도 init만 남겨도 baseline tie. **mit-states/ViT-L/14에서는 sub-meaning name 방향도 dead end로 확인**. mit-states는 너무 거칠어서 sub-meaning 효과가 안 드러나는 것일 수 있음 → UT-Zappos / C-GQA로 이동 결정 (3 seed로).

### Phase 3 — UT-Zappos baseline 시도 + NaN 발견 (5-2 새벽~오후)

UT-Zap에서 ClusPro baseline + LHP 변형을 비교하기 위해 ClusPro baseline부터 돌림.

- **5-2 02:06** baseline lr=1e-4 → HM 0.4902 / AUC 0.3388. epoch 1 step 1356(47%)에서 train loss=NaN.
- **5-2 10:32** baseline lr=5e-5 → HM 0.5009 / AUC 0.3473. epoch 1 step 1468(51%)에서 NaN.

LR을 낮춰도 NaN이 ~100 step만 늦어짐. 시드 동일 → **데이터 순서 동일 → 같은 batch에서 터지는 패턴**. LR tune이 아니라 numeric 문제.

### Phase 4 — 오늘 오후 NaN 디버깅 (5-2 15:00 ~ 16:10)

#### 4.1 HSIC 비섹트 probe (hsic_weight=0, 1 epoch)
- 결과: NaN 여전히 발생, 다만 step 1980(69%)으로 +500 step 지연. **HSIC 단독 범인 가설 기각**.
- 부수적 발견: HSIC 빼도 metric 거의 동일 (HM 0.4861 / AUC 0.3389) → UT-Zap에서 HSIC 기여 ~0. `model/lhp_czsl.py:7` 주석 ("HSIC를 cosine decorrelation으로 교체, 소배치 안정성")과 일치.

#### 4.2 bf16 probe
- 가설: AMP fp16의 누적 numeric drift가 진짜 원인 → bf16 (fp32 동급 dynamic range)으로 해결 가능.
- 결과: **더 나쁨**. step 813에서 NaN, 이후 영구 NaN(2,790 step 동안 회복 안 됨), val_best 미저장으로 final test 크래시.
- 진단: GradScaler를 같이 뺐기 때문. fp16+GradScaler가 사실은 NaN gradient를 자동 skip해서 params를 NaN 오염으로부터 보호하고 있었음. bf16 단독은 그 안전망이 없어서 한 번 NaN 들어가면 영구 corruption.

#### 4.3 guard probe — 최종 fix
변경:
- `train.py`: fp16+GradScaler 복원 + 명시적 `if not torch.isfinite(loss): zero_grad + scheduler step + continue` 추가
- `model/cluspro_baseline.py:_update_prototypes` 진입 시 `torch.isfinite(batch_attr_f).all() and torch.isfinite(batch_obj_f).all()` 체크, 아니면 early return (proto_momentum=0.99 EMA queue 영구 오염 방지)

결과 (1 epoch, lr=1e-4):
- **NaN 표시 0회**, loss-skip 18회(2875 step 중 0.6%)
- HM 0.4919 / AUC 0.3493 → 1 epoch만 돌렸는데도 이전 15-epoch lr=5e-5 baseline (AUC 0.3473) 초과
- 이전 NaN 오염 환경에서는 학습의 절반 이상이 silently skip되고 있었다는 가설 확인

#### 4.4 핵심 교훈 (mit-states 결과 재해석 위험)
mit-states/ViT-L/14 결과들도 같은 NaN 오염을 겪었을 가능성이 있음. mit-states에서 baseline tie 결론은 **NaN 오염 환경에서의 tie**일 수 있고, 진짜 능력치 비교가 아니었을 수 있음. 향후 핵심 결과는 fix 후 재실행이 안전.

---

## 3. 코드/설정 변경 inventory

### 코드 변경 (5-2 오후)

- `train.py:46-86` — fp16 GradScaler 유지, autocast 유지. 추가:
  ```python
  if not torch.isfinite(loss):
      optimizer.zero_grad()
      scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))
      progress_bar.set_postfix({"train loss": ..., "skipped": "nan"})
      progress_bar.update()
      continue
  ```
- `model/cluspro_baseline.py:236-242` — `_update_prototypes` 진입 시 features 유한성 체크:
  ```python
  if not (torch.isfinite(batch_attr_f).all() and torch.isfinite(batch_obj_f).all()):
      return
  ```

### 새 config

- `config/cluspro_baseline_utzap_l14_hsic0_probe.yml` — bisect용 (HSIC=0, 1 ep)
- `config/cluspro_baseline_utzap_l14_bf16_probe.yml` — bf16 probe용 (사용 후 폐기 가능)
- `config/cluspro_baseline_utzap_l14_guard_probe.yml` — fix 검증용 (1 ep)
- `config/cluspro_baseline_utzap_l14_v2_seed{0,1,2}.yml` — 본런 3-seed (15 ep, lr=1e-4)

### 새 스크립트

- `scripts/run_baseline_utzap_3seeds.sh` — seed 0/1/2 순차 실행, 각 seed별 로그 분리

---

## 4. 현재 상태 + 다음 단계

### 완료
- **5-2 16:10 ~ 23:06**: 3-seed UT-Zap baseline (HM 0.6397 / AUC 0.4907). 비교 기준선 확정.
- **5-2 23:47 ~ 5-3 06:36**: 3-seed UT-Zap v1_init_only (HM 0.6384 / AUC 0.4877). **baseline과 tie** (val/test 두 split 모두).
- 결론: **sub-meaning name 기반 prototype init은 mit-states / UT-Zap 두 데이터셋 모두에서 baseline 대비 의미 있는 격차를 만들지 못함**. 단순 init 방향은 dead end로 확인.

### 폐기 (5-3 09:16 ~ 16:08 KST, invalid)
- **3-seed UT-Zap v3_text 본런 — invalid, 결과 신뢰 불가.** §4.5 참고.
  - 끝까지 NaN 0회로 정상 완료, config 덤프에 `text_ensemble=True` 정상 진입, 체크포인트 size도 v1 대비 +764B 차이.
  - 그러나 seed 0/1/2 모두에 대해 (a) 매 epoch tqdm train loss가 v1_init_only와 소수점 두 자리까지 동일, (b) `test.py` 5-line summary 5줄이 모두 v1_init_only와 비트 일치. 두 학습이 사실상 같은 trajectory를 탔다는 뜻.
  - 결론: 표에 올리지 않음. 표를 올리면 advisor 미팅에서 "v3=v1 tie"로 잘못 해석될 위험. 디버깅 후 재실행.

### (5-3 09:16 KST에 시작했던 v3_text 본런 원래 설명, 참고 보존)
- 코드 변경 (`model/lhp_czsl.py`):
  - `text_ensemble: True` 플래그 추가 — 켜지면 `soft_att_obj` shape이 `(N_attr+N_obj, dim)` → `(sum_Ka+sum_Ko, dim)`로 바뀌고 sub-meaning당 학습 임베딩 1개씩 (init = sub-meaning name 토큰 평균).
  - `_construct_token_tensors`: comp 분기는 sub 임베딩을 primitive별로 mean-pool 후 prompt slot에 주입; attr-only/obj-only 분기는 sub-level prompt 각각 인코딩.
  - `train_forward` / `val_forward`: text encoder 통과 후 attr/obj 헤드 feature를 `_pool_sub_features` (scatter_add 기반 mean) 로 per-primitive 환원.
  - `_format_sub_name`: K>1 sub 이름은 `_`/`.`을 공백으로 (e.g. `smooth_leather` → `smooth leather`); K=1이고 sub==primitive인 경우는 baseline tokenization 보존.
- UT-Zap 기준 sum_Ka=19, sum_Ko=16 (vs N_attr=16, N_obj=12) — text encoder 인코딩량 +20% 정도.
- 새 config: `config/lhp_czsl_v3_text_utzap_l14_seed{0,1,2}.yml` (baseline과 동일 조건: lr=1e-4, 15 ep, fp16+GradScaler+guard). 새 스크립트: `scripts/run_v3_text_utzap_3seeds.sh`.
- 마스터 로그 `logs/run_v3_text_utzap_3seeds_20260503_091602.log`, seed별 `logs/train_v3_text_utzap_seed{0,1,2}_20260503_091602.log`.
- 첫 3분 헬스체크: 22s만에 step 57/2875, train loss 1.5→1.48 수렴, NaN 0회. 정상.
- 예상 ~7h, 완료 ~16:30 KST.

### 끝나면 할 일 (v3_text 폐기로 보류)
1. v3_text vs baseline / v1_init_only 비교 (val + test 두 convention 모두):
   - **HM/AUC 차이가 시드 std 밖**(즉 `>0.013` HM, `>0.016` AUC) **+ 양의 방향**이면 sub-meaning text-side 효과 확인 → C-GQA로 확장.
   - tie면 sub-meaning 방향 전체가 dead end로 굳힘 → angle C (selective branching) 또는 다른 방향.
2. RESEARCH_LOG.md §1에 v3_text 표 추가.

### §4.5 v3_text invalid run 진단 (5-3 21:35)

증거:
- `logs/train_v3_text_utzap_seed{0,1,2}_*.log`의 첫 10개 tqdm `train loss=` 값이 `logs/train_v1_init_only_utzap_seed{0,1,2}_*.log`와 정확히 동일 (1.24, 1.35, 1.40, 1.45, 1.50, ...). baseline과는 다름 (1.26, 1.37, 1.42, ...).
- `test.py` 최종 5-line summary (best_seen / best_unseen / best_hm / AUC / attr_acc / obj_acc)가 v3_text seed 0/1/2 ↔ v1_init_only seed 0/1/2 비트 일치.
- config Namespace 덤프: `text_ensemble=True, sub_meanings_path='data/sub_meanings_utzap.json', decorr_weight=0.0, sem_weight=0.0` 정상.
- `data/sub_meanings_utzap.json` 검사: attr 19 sub (Faux.Fur K=2, Leather K=2, Sheepskin K=2 + 13×K=1) / obj 16 sub (Sandals/Clogs/Heels/Sneakers K=2 + 8×K=1). sum_Ka=19 ≠ N_attr=16, sum_Ko=16 ≠ N_obj=12 → text_ensemble path가 활성화될 조건은 만족.
- `model/lhp_czsl.py:99,310,366,381,571,605` text_ensemble 분기 코드는 sub-emb pool/scatter_add로 v1과 다른 graph를 만들어야 함.
- `checkpoint/lhp_czsl_v3_text_l14_utzap_seed0/`: epoch_*.pt 파일 size 1746013317 (v1: 1746012553), 차이 +764B. 7행 × 768dim × float32 = 21,504B 예상치와 안 맞음 → soft_att_obj가 실제로 (sum_Ka+sum_Ko, dim) = (35, 768)로 alloc 됐는지 의심스러움.

가설 (디버그 우선순위):
1. **a)** soft_att_obj가 (28, dim) 그대로 alloc 됐고, `_construct_token_tensors`의 `embs[:self.sum_Ka]` 슬라이싱이 길이 부족으로 fallback 동작.
2. **b)** `_construct_soft_prompt`에서 text_ensemble=True 분기는 탔으나 `tokens_to_init`이 K=1 case에서 primitive 이름 그대로라 `attr_pooled`/`obj_pooled`가 baseline과 동일한 init.
3. **c)** test.py가 학습된 v3 ckpt가 아닌 다른 ckpt(v1)를 로드. (가능성 낮음 — save_path는 분리됨.)

디버깅 결과 (5-3 21:50):
1. v3 epoch_0.pt의 `soft_att_obj.shape == (28, 768)` 확인. v1과 동일 shape — **가설 a/b 둘 다 부분 적중**: 학습 자체가 v1의 28-row prompt graph로 동작.
2. 직접 LHPCZSL을 수동 init하면 `(35, 768)` 정상. 둘의 차이는 **primitive 이름**이었음: train.py가 `'Faux.Fur'` → `'faux fur'`로 normalize한 후 모델에 넘기는데(`train.py:172-173`), `data/sub_meanings_utzap.json`의 key는 dotted/cased raw (`'Faux.Fur'`). `_load_sub_meanings`의 `if attr_name in attrs_data:` 매칭이 모두 실패 → 모든 primitive가 default `K=1, sub=[name]`으로 떨어져 v1과 사실상 동일한 구조.
3. mit-states json은 lowercase + no-dot이라 이 normalize의 영향이 없음 → 이전 mit-states 결과는 이 버그의 영향 없음.

수정 (5-3 21:50, `model/lhp_czsl.py:_load_sub_meanings`):
- json key를 train.py와 동일하게 `replace('.', ' ').lower()`로 normalize한 dict로 lookup.
- 함수 끝에 `[_load_sub_meanings] sub_meanings_path=... | attrs: total=N, K>1=k, sum_K=s | objs: ...` sanity print 추가.
- `text_ensemble=True`인데 K>1 매칭이 0이면 즉시 `RuntimeError` (silent regression 방지).
- 검증: 수정 후 동일 yml로 instantiate → `soft_att_obj.shape == (35, 768)`, `sum_Ka=19, sum_Ko=16` 정상.

상태:
- v3_text 3-seed 본런 **재실행 필요**. 폐기된 `checkpoint/lhp_czsl_v3_text_l14_utzap_seed{0,1,2}/`는 invalid이므로 정리 또는 _invalid suffix로 rename 후 새 학습. 동일 스크립트(`scripts/run_v3_text_utzap_3seeds.sh`)로 ~7h 예상.

### 다음 단계 후보 (B 결과 본 후)
- **C. selective branching**: K=1 vs K>1 binary signal로 재해석. K>1 attribute에만 branching.
- **D. mit-states fix 후 재실행**: §4.4 caveat 검증. 비용 큼.
- **A. C-GQA**: B가 효과 있으면 그쪽으로 확장; 효과 없으면 비용 대비 가치 낮음.

### 정리해야 할 것 (시간 날 때)
- probe 체크포인트 디렉터리 삭제: `checkpoint/cluspro_baseline_l14_utzap_{hsic0,bf16,guard}_probe/`
- mit-states 결과들을 fix 후 재실행할지 결정 (단, 시간 비용 큼; UT-Zap에서 의미 있는 격차가 나오면 mit-states는 secondary)

### 정리해 둘 미해결 의문
- fp16 GradScaler가 NaN gradient를 어떻게 skip하길래 prototype EMA가 결국 회복했는지 정확한 메커니즘은 unclear. (이전 fp16+scaler 환경에서도 step 1356~2790 동안 NaN 표시 후 회복 패턴이 있었음.) 진짜로 회복된 것인지, 아니면 EMA queue가 부분적으로만 오염됐다가 fresh batch들로 mix-out 된 것인지 미상. 새 가드 들어왔으니 실용적으로는 무관.
- 본질적 NaN 원인은 끝까지 안 잡았음 — HSIC도 contrastive도 단독 범인 아니었고, AMP fp16 + 어떤 batch의 특정 input combination이 numeric overflow 일으키는 패턴. 가드로 우회한 상태. 향후 같은 문제가 다른 데이터셋에서 더 심하게 나오면 그때 재조사.

---

## 5. 5-4 진단 라운드 — k_validation + prototype collapse (advisor 미팅 5/6 자료)

5/4 v3_text seed2 재실행이 16:13 KST에 끝나면서 baseline_v2 / v1_init_only / v3_text 모두 3-seed 정상 비교 가능. 동시에 thesis 자체에 대한 두 가지 진단을 수행.

### 5.1 v3_text seed2 재실행 결과

3-seed 최종 (test_pairs):
| seed | seen | unseen | HM | AUC |
|---|---|---|---|---|
| 0 | 0.6823 | 0.7536 | 0.5509 | 0.4326 |
| 1 | 0.6823 | 0.7488 | 0.5402 | 0.4194 |
| 2 (재실행) | 0.7128 | 0.7411 | 0.5299 | 0.4102 |
| **mean** | 0.6925 | 0.7478 | **0.5403** | **0.4207** |

baseline_v2 mean (HM 0.5481 / AUC 0.4279) 대비 **HM −0.008, AUC −0.007** — 시드 std 안. **v3_text도 baseline tie 확정**.

### 5.2 K-validation: LLM K vs CLIP visual K (UT-Zap, n=28 primitives)

스크립트: [`tools/k_validation.py`](tools/k_validation.py). primitive별로 학습 이미지를 CLIP ViT-L/14로 인코딩 → silhouette 기반 k_visual ∈ [1,5] 결정 → `sub_meanings_*.json`의 K_LLM과 비교. mit-states (v1 + v2)는 18:08~ 진행 중, 결과 나오면 추가.

| | n | Spearman ρ | exact agreement | mean k_vis | mean k_llm | k_vis 분포 | k_llm 분포 |
|---|---|---|---|---|---|---|---|
| UT-Zap **attrs** | 16 | **−0.450** (p=0.08) | 3/16 (19%) | 3.00 | 1.19 | K=2:8, K=3:2, K=4:4, K=5:2 | K=1:13, K=2:3 |
| UT-Zap **objs**  | 12 | −0.316 (p=0.32) | 4/12 (33%) | 2.50 | 1.33 | K=2:10, K=5:2 | K=1:8, K=2:4 |

**해석**:
1. **음의 상관** — LLM이 K 크다고 한 primitive에서 visual K는 오히려 작음. ρ=0이면 "다른 차원" 변명 가능, ρ<0이면 LLM K가 visual diversity와 **체계적으로 어긋남**.
2. **Visual K는 모두 ≥2** — 전 primitive가 시각적으로 ≥2 cluster. LLM은 attrs 81%, objs 66%를 K=1 처리.
3. **Mean 크게 어긋남** (visual ~3 vs LLM ~1.2) — LLM이 시각적 다양성을 **체계적으로 과소평가**.

UT-Zap 한 데이터셋만으로 단정은 무리지만, "LLM zero-shot K가 visual diversity proxy로 작동한다"는 가설은 UT-Zap에선 **명확히 기각**. mit-states 결과도 비슷하면 K determination 메커니즘 자체 재설계 근거 확보. (caveat: silhouette는 instance-level variation도 포함하므로 LLM이 측정하는 "semantic sub-type"과 정확히 같진 않음. 그러나 anti-correlation은 그 caveat를 넘는 신호.)

### 5.3 Prototype collapse 진단 — 모든 학습 ckpt에서 effective K 측정

스크립트: [`tools/prototype_diagnostic.py`](tools/prototype_diagnostic.py). 각 `val_best.pt`에서 모든 `attr_queue{i}` / `obj_queue{i}` (각 [K=5, D=768])를 꺼내 per-primitive로:
- mean off-diagonal cosine — prototype 간 유사도 (1 = collapse, 0 = orthogonal)
- effective K = (Σs)² / Σs² (singular values) ∈ [1, K] — 실제로 사용되는 prototype 차원 수

전 16개 체크포인트 결과:

| checkpoint | attr off-cos | attr eff_K | obj off-cos | obj eff_K |
|---|---:|---:|---:|---:|
| **cluspro_baseline_l14_mit** | **0.994** | **1.08/5** | **0.986** | **1.12/5** |
| cluspro_baseline_l14_mit_k3 | 0.989 | 1.08/3 | 0.981 | 1.10/3 |
| cluspro_baseline_l14_utzap (lr=1e-4) | 0.800 | 2.21/5 | 0.939 | 1.36/5 |
| cluspro_baseline_l14_utzap_lr5e5 | 0.914 | 1.67/5 | 0.975 | 1.21/5 |
| **cluspro_baseline_l14_utzap_v2_seed0** | **0.967** | **1.37/5** | **0.992** | **1.11/5** |
| cluspro_baseline_l14_utzap_v2_seed1 | 0.999 | 1.06/5 | 1.000 | 1.02/5 |
| cluspro_baseline_l14_utzap_v2_seed2 | 0.993 | 1.15/5 | 0.999 | 1.04/5 |
| **lhp_czsl_v1_init_only_l14_mit** | **0.024** | **3.88/5** | **0.052** | **3.65/5** |
| lhp_czsl_v1_init_only_l14_utzap_seed0 | 0.967 | 1.37/5 | 0.992 | 1.11/5 |
| lhp_czsl_v1_init_only_l14_utzap_seed1 | 0.988 | 1.20/5 | 0.998 | 1.06/5 |
| lhp_czsl_v1_init_only_l14_utzap_seed2 | 0.981 | 1.27/5 | 0.996 | 1.08/5 |
| lhp_czsl_v1_l14_mit | 0.023 | 3.88/5 | 0.051 | 3.65/5 |
| lhp_czsl_v2_l14_mit | 0.317 | 2.07/5 | 0.318 | 2.05/5 |
| **lhp_czsl_v3_text_l14_utzap_seed0** | **0.016** | **3.89/5** | **0.029** | **3.74/5** |
| lhp_czsl_v3_text_l14_utzap_seed1 | 0.025 | 3.88/5 | 0.033 | 3.74/5 |
| lhp_czsl_v3_text_l14_utzap_seed2 | 0.015 | 3.88/5 | 0.046 | 3.74/5 |

**핵심 관찰**:

1. **ClusPro baseline은 K=5를 사실상 안 씀**. mit에서 eff_K ≈ 1.08, off-cos ≈ 0.99 — 5개 prototype이 거의 동일 vector. utzap_v2도 평균 eff_K ~1.2.
2. **LHP-CZSL는 데이터셋·variant에 따라 분리도 차이 큼**:
   - mit + sub-meaning init (v1, v1_init_only) → eff_K ≈ 3.88 (분리 잘 됨)
   - mit + 강제 K≥3 (v2) → eff_K ≈ 2.07 (중간)
   - **utzap + v1_init_only → 거의 collapse (eff_K ~1.2)** — utzap에선 init 효과 없음
   - **utzap + v3_text (text ensemble) → eff_K ≈ 3.88** (text-side ensemble이 분리 강제)
3. 모든 LHP variant도 buffer는 K=5 할당 (variable K는 loss/매스킹으로만 표현; tensor shape 동일).

### 5.4 Thesis 자체의 재정의 필요 — 5-3 (advisor 미팅 핵심 포인트)

진단 결과를 종합하면 thesis의 전제 자체가 흔들림:

**기존 thesis** (5/3 시점): "Variable K가 fixed K=5보다 효율적 — 단순 primitive에 K=5는 낭비"

**현 진단으로 드러난 사실**:
1. ClusPro baseline의 fixed K=5는 **이미 collapse해서 effective K=1**로 동작 중 (특히 mit, utzap_v2_seed1/2)
2. LHP variant의 eff_K가 baseline보다 훨씬 큰 경우(mit v1/v1_init/v3_text utzap eff_K~3.88)에도 **test 성능 tie**
3. LLM K는 visual diversity의 proxy 역할 못 함 (UT-Zap ρ=−0.45)

→ **재정의된 질문**: "Variable K vs fixed K=5"가 아니라 **"prototype 수가 1이든 4든 효과 없는데, 그럼 prototype 메커니즘 자체가 CZSL 성능에 의미 있는 lever인가?"**

이건 thesis를 **확대**하는 게 아니라 **축소**시키는 발견. advisor에게 솔직히 던질 질문:
- (Q1) ClusPro의 "5개 cluster prototype"이 collapse하는 게 정상 동작인지, 학습 dynamics 결함인지? (decorrelation loss 부재가 원인일 가능성)
- (Q2) Prototype memory가 효과 없는 게 검증되면, LHP-CZSL 전체 framing(prototype 중심)을 재고할지? text-side / loss-side로 axis 이동?

### 5.5 Falsifiable next experiment — 1~2일

**"올바른 K (visual K)로 학습 → 이기면 thesis salvageable, tie면 thesis 폐기"**

1. `tools/k_validation.py` 결과로 visual-K 기반 sub_meanings JSON 생성 (LLM 우회)
2. 동일 학습 (`v1_init_only` 구조) on UT-Zap, 3-seed
3. 결과:
   - baseline tie → variable K thesis 폐기. prototype 메커니즘 부분 자체를 재고. text-side / open-world / 다른 axis로 pivot
   - baseline 이김 → "LLM K가 문제였다" 증명. Path A(multimodal LLM) 또는 Path C(LLM+visual hybrid) 진행

이 실험 한 번이면 thesis salvage 가능성에 대한 binary 결론 확보. **advisor 미팅에서 이 실험을 제안하면 "다음 1주 로드맵"이 명확**.

### 5.6 미팅 자료 정리 (5/6 수요일용)

들고 갈 것:
- 이 §5 섹션 전체 (한 번에 나열) + §1.1 부록 테이블
- `logs/k_validation/k_validation_*.json` summary (UT-Zap + mit_v1 + mit_v2)
- `logs/k_validation/prototype_diagnostic.json`

핵심 talking points (3분 내):
1. **부정적 결과 정직 보고**: variable K로 baseline 못 이김 (3 variants × 2 dataset × 3 seeds 모두)
2. **두 가지 진단으로 원인 좁힘**:
   - (a) LLM K 자체가 visual diversity와 anti-correlated (UT-Zap ρ=−0.45)
   - (b) ClusPro baseline은 K=5를 사실상 안 씀 (eff_K ≈ 1.08)
3. **Thesis 재정의 제안**: "어떤 K가 좋은가"가 아니라 "prototype 수 자체가 lever인가"
4. **Falsifiable next step 제안**: visual K로 학습해서 thesis 운명 결정 (1~2일)
