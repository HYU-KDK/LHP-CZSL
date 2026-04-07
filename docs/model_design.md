# LHP-CZSL Stage 1 모델 설계서

## 개요

CLUSPRO baseline에 두 가지 변경을 가한 모델:
1. **Variable K**: LLM이 결정한 primitive별 prototype 수
2. **L_sem**: 같은 primitive 내 prototype 간 의미 거리 정렬 손실

---

## Phase 0: LLM Sub-meaning Generation

### 입력
데이터셋의 전체 primitive (attr + obj) 목록

### LLM 프롬프트 (예시)
```
Given the visual attribute "{attr}", list its visually distinct sub-meanings.
For each pair, provide a semantic distance (0-1, where 0=identical, 1=completely different).

Rules:
- Only include sub-meanings that are VISUALLY distinguishable in images
- If the attribute has no meaningful visual sub-types, return just the attribute itself
- Maximum 5 sub-meanings

Output JSON:
{
  "sub_meanings": ["worn", "faded", "aged"],
  "pairwise_distances": {"worn-faded": 0.3, "worn-aged": 0.5, "faded-aged": 0.4}
}
```

### 출력 형식
```json
// sub_meanings.json
{
  "attrs": {
    "old":         {"sub": ["worn", "faded", "aged"], "K": 3, "d_sem": {"worn-faded": 0.3, "worn-aged": 0.5, "faded-aged": 0.4}},
    "broken":      {"sub": ["cracked", "shattered"], "K": 2, "d_sem": {"cracked-shattered": 0.6}},
    "translucent": {"sub": ["translucent"], "K": 1, "d_sem": {}}
  },
  "objs": {
    "car":   {"sub": ["sedan", "suv", "truck"], "K": 3, "d_sem": {"sedan-suv": 0.4, ...}},
    "knife": {"sub": ["kitchen_knife", "pocket_knife"], "K": 2, "d_sem": {"kitchen_knife-pocket_knife": 0.5}}
  }
}
```

---

## Architecture 변경사항

### Variable K Prototype

**기존 CLUSPRO:**
```python
# 모든 primitive에 K=5 고정
self.attr_prototypes = nn.Parameter(torch.randn(num_attrs, 5, d))
self.obj_prototypes = nn.Parameter(torch.randn(num_objs, 5, d))
```

**LHP-CZSL Stage 1:**
```python
# primitive별 K가 다름 → dict 또는 padded tensor
# 옵션 A: Dict 방식
self.attr_prototypes = nn.ParameterDict({
    attr: nn.Parameter(torch.randn(K_attr[attr], d))
    for attr in attr_list
})

# 옵션 B: Padded tensor 방식 (배치 연산 효율)
# K_max=5로 패딩, mask로 유효 prototype 구분
self.attr_prototypes = nn.Parameter(torch.randn(num_attrs, K_max, d))
self.attr_K_mask = {}  # {attr_idx: K} — 유효 prototype 수
```

> 구현 편의상 옵션 B (padded tensor + mask) 권장.
> K < K_max인 경우 나머지 슬롯은 logit 계산에서 -inf masking.

### Momentum Update 변경

```python
# 기존: 모든 K개 prototype에 KMeans
# 변경: K_p개만 사용하여 KMeans
def update_prototypes(self, features, primitive_idx):
    K = self.attr_K_mask[primitive_idx]
    if K == 1:
        # KMeans 불필요, 단순 EMA
        self.prototypes[primitive_idx][0] = momentum * self.prototypes[primitive_idx][0] + (1-momentum) * features.mean(0)
    else:
        # K개 클러스터로 KMeans
        kmeans = KMeans(n_clusters=K)
        ...
```

---

## L_sem 구현

```python
def compute_l_sem(self, prototypes, d_sem_matrix, K_mask):
    """
    prototypes: [num_primitives, K_max, d]
    d_sem_matrix: [num_primitives, K_max, K_max] — LLM 제공 의미 거리
    K_mask: {primitive_idx: K}
    """
    loss = 0.0
    count = 0
    
    for p_idx, K in K_mask.items():
        if K < 2:
            continue  # K=1이면 pair 없음
        
        protos = prototypes[p_idx, :K]  # [K, d]
        protos_norm = F.normalize(protos, dim=-1)
        
        # prototype 간 cosine similarity
        vis_sim = protos_norm @ protos_norm.T  # [K, K]
        
        # LLM 의미 거리 → similarity로 변환 (1 - distance)
        sem_sim = 1.0 - d_sem_matrix[p_idx, :K, :K]
        
        # upper triangle만 (중복 제거)
        mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
        
        loss += ((vis_sim[mask] - sem_sim[mask]) ** 2).sum()
        count += mask.sum()
    
    return loss / max(count, 1)
```

---

## 전체 Loss

```python
L = L_BAS + alpha * L_PCL + beta * L_PDL + gamma * L_sem

# 초기 권장값
alpha = 0.1   # CLUSPRO 기본값
beta = 0.1    # cosine decorrelation
gamma = 0.05  # L_sem (새로 도입, 보수적으로 시작)
```

---

## 구현 체크리스트

1. [ ] sub_meanings.json 생성 (LLM Phase 0)
2. [ ] sub_meanings.json 로드 및 K_mask 구성
3. [ ] Prototype tensor를 variable K 지원으로 변경 (padded + mask)
4. [ ] Momentum update에서 K별 분기 처리
5. [ ] d_sem_matrix 구성 (JSON → tensor)
6. [ ] L_sem 구현
7. [ ] HSIC → Cosine decorrelation 교체
8. [ ] 학습 스크립트 수정 (gamma 하이퍼파라미터 추가)
