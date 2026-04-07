# Likelihood-based Hierarchical Prototypes for Compositional Zero-Shot Learning (LHP-CZSL)

이 저장소는 Compositional Zero-Shot Learning (CZSL)을 위한 LHP-CZSL(Likelihood-based Hierarchical Prototypes) 프레임워크의 공식 구현체입니다.

## 1. 환경 설정 (Environment Setup)

이 프로젝트는 Python 3.8+ 및 PyTorch를 기반으로 합니다.

### 의존성 설치
필요한 패키지들을 설치합니다:

```bash
pip install torch torchvision numpy Pillow tqdm opencv-python scipy PyYAML
```

*참고: `clip_modules/` 폴더에 커스텀 CLIP 구현이 포함되어 있어 별도의 `openai-clip` 설치가 필요하지 않을 수 있습니다.*

## 2. 데이터셋 준비 (Dataset Preparation)

기본적으로 MIT-States, UT-Zappos, C-GQA 데이터셋을 지원합니다.

### 데이터셋 구조
데이터셋 폴더(예: `data/mit-states/`)는 다음과 같은 구조여야 합니다:

```
data/mit-states/
├── images/             # 이미지 파일들이 포함된 폴더
├── compositional-split-natural/
│   ├── train_pairs.txt
│   ├── val_pairs.txt
│   └── test_pairs.txt
└── metadata_compositional-split-natural.t7  # 데이터셋 메타데이터 (PyTorch serialized)
```

### 경로 설정
`config/cluspro_baseline_mit.yml` 파일에서 `dataset_path`를 실제 데이터가 위치한 경로로 수정하십시오.

```yaml
train:
  dataset_path: /path/to/your/data/mit-states
```

## 3. 훈련 (Training)

제공된 설정 파일을 사용하여 모델을 훈련할 수 있습니다.

```bash
python train.py --yml_path config/cluspro_baseline_mit.yml
```

명령줄 인수를 통해 설정을 덮어쓸 수도 있습니다:
```bash
python train.py --yml_path config/cluspro_baseline_mit.yml --lr 0.00005 --epochs 20
```

## 4. 평가 (Evaluation / Testing)

훈련된 체크포인트를 사용하여 모델의 성능을 평가합니다.

```bash
python test.py --yml_path config/cluspro_baseline_mit.yml --load_model checkpoint/cluspro_baseline_b16_mit/best_model.pt
```

## 주요 파라미터 안내 (`parameters.py`)
- `--model_name`: 사용할 모델 이름 (기본값: `cluspro_baseline`)
- `--dataset`: 데이터셋 이름 (`mit-states`, `ut-zappos`, `cgqa`)
- `--lr`: 학습률 (Learning Rate)
- `--epochs`: 총 에폭 수
- `--train_batch_size`: 훈련 배치 크기
- `--save_path`: 모델 체크포인트 저장 경로
- `--open_world`: Open-world 설정에서의 평가 여부 (기본값: `False`)
- `--bias`: 평가 시의 Bias 값

---
더 자세한 모델 설계 내용은 `docs/model_design.md`를 참조하십시오.
