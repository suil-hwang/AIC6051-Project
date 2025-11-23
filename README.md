# AIC6051-Project

AIC6051 Prediction and Planning in Autonomous Driving (2025 Fall, Hanyang University)

## 원본 논문

Ryo Yonetani*, Tatsunori Taniai*, Mohammadamin Barekatain, Mai Nishimura, Asako Kanezaki, "Path Planning using Neural A\* Search", ICML, 2021

- [[논문]](https://arxiv.org/abs/2009.07476)
- [[프로젝트 페이지]](https://omron-sinicx.github.io/neural-astar/)

## 🎯 개요

Neural A*는 학습 가능한 인코더와 미분 가능한 A* 탐색 알고리즘을 결합한 데이터 기반 경로 탐색 플래너입니다.

### 주요 특징

- **세 가지 모델 변형 지원**:

  - 🔵 **Vanilla A\***: 전통적인 A\* 알고리즘 (베이스라인)
  - 🟢 **Neural A\***: 학습 가능한 인코더를 활용한 개선된 A\* (원본)
  - 🟣 **Field-based Neural A\***: Geodesic 및 Obstacle Distance Fields를 활용한 개선

- **Distance Fields 통합**: Heat Method를 사용한 측지 거리장과 장애물 거리장 계산
- **통합된 학습 파이프라인**: 멀티런 지원으로 모든 모델을 한 번에 학습
- **시각화 도구**: 모델 간 성능 비교를 위한 GIF 생성 기능

---

## 🚀 빠른 시작

### 시스템 요구사항

- **OS**: Ubuntu ≥18.04, WSL2 (Ubuntu 20.04), Windows 11
- **Python**: 3.11.9 (환경 파일에서 자동 설치)
- **GPU**: 학습 시 권장 (추론은 CPU로 가능)
- **CUDA**: 11.8 (환경 파일에 포함)
- **Conda**: Anaconda 또는 Miniconda 설치 필요

### 설치 방법

```bash
git clone --recursive https://github.com/omron-sinicx/neural-astar
cd neural-astar
conda env create -f environment.yml
conda activate neural-astar
pip install -e .
```

---

## 🎓 학습하기

### 1. 단일 모델 학습

#### Vanilla A\* (베이스라인)

```bash
python scripts/train.py model_type=vanilla
```

> 학습 없이 검증만 수행하여 베이스라인 성능을 측정합니다.

#### Neural A\* (원본)

```bash
python scripts/train.py model_type=neural
```

> 학습 가능한 인코더를 사용하는 Neural A\*를 학습합니다.

#### Field-based Neural A\* (개선 버전)

```bash
python scripts/train.py model_type=field
```

> Distance Fields를 활용한 개선된 Neural A\*를 학습합니다.

### 2. 모든 모델 한 번에 학습

```bash
python scripts/train.py --multirun model_type=vanilla,neural,field
```

> Hydra의 multirun 기능을 사용하여 세 가지 모델을 순차적으로 학습합니다.

### 3. 학습 결과

학습된 모델은 다음 경로에 저장됩니다:

```
model/
├── vanilla_mazes_032_moore_c8/  # Vanilla A* 결과
├── neural_mazes_032_moore_c8/   # Neural A* 체크포인트
└── field_mazes_032_moore_c8/    # Field-based 체크포인트
```

TensorBoard로 학습 과정을 모니터링할 수 있습니다:

```bash
tensorboard --logdir model/
```

---

## 🎨 시각화

### 1. 단일 모델 GIF 생성

```bash
# Vanilla A*
python scripts/create_gif.py planner=va problem_id=1

# Neural A*
python scripts/create_gif.py planner=na problem_id=1

# Field-based Neural A*
python scripts/create_gif.py planner=field problem_id=1
```

### 2. 모델 비교 GIF 생성

세 가지 모델을 나란히 비교하는 GIF를 생성합니다:

```bash
python scripts/create_comparison_gif.py problem_id=1
```

**출력물:**

- `gif/comparison/vanilla_{dataset}_{id}.gif` - Vanilla 단독
- `gif/comparison/neural_{dataset}_{id}.gif` - Neural 단독
- `gif/comparison/field_{dataset}_{id}.gif` - Field-based 단독
- `gif/comparison/comparison_{dataset}_{id}.gif` - **세 모델 동시 비교**

### 3. 여러 문제에 대해 GIF 생성

```bash
python scripts/create_comparison_gif.py --multirun problem_id=1,2,3,4,5
```

## 🎮 WarCraft 맵 데이터 사용

### 데이터 준비

1. [Blackbox Combinatorial Solvers](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S)에서 `warcraft_maps.tar.gz` 다운로드
2. `12x12` 디렉토리를 추출하여 프로젝트 루트에 배치

### 학습

```bash
python scripts/train_warcraft.py
```

학습 후 [`notebooks/example_warcraft.ipynb`](notebooks/example_warcraft.ipynb)를 참조하세요.

---

## 🔬 Distance Fields 기능

### Heat Method를 사용한 Geodesic Distance Field

Heat Method는 열 확산 방정식을 활용하여 표면 상의 거리장을 계산합니다:

1. **Heat Diffusion**: 목표 지점에서 열을 확산
2. **Gradient Computation**: 열 분포의 기울기 계산
3. **Distance Recovery**: Poisson 방정식을 풀어 측지 거리 복원

---

## 📊 성능 평가 지표

학습 및 평가 시 다음 지표가 계산됩니다:

- **`metrics/val_loss`**: 검증 손실
- **`metrics/p_opt`**: 최적 경로 비율 (1.0이 최선)
- **`metrics/p_exp`**: 탐색한 노드 비율 (낮을수록 효율적)
- **`metrics/h_mean`**: 조화 평균 성능 지표

Field-based 모델은 추가 손실을 포함합니다:

- **`loss/geodesic`**: Geodesic distance field 손실
- **`loss/obstacle`**: Obstacle distance field 손실

---

## 📝 데이터셋 생성

새로운 미로 데이터를 생성하려면 [planning-datasets](https://github.com/omron-sinicx/planning-datasets) 리포지토리를 참조하세요.

---

## 🤝 기여

이 리포지토리는 다음 코드를 포함합니다:

- [RLAgent/gated-path-planning-networks](https://github.com/RLAgent/gated-path-planning-networks) [1] (저자 허가)
- [martius-lab/blackbox-backprop](https://github.com/martius-lab/blackbox-backprop) [2]

---

## 📖 참고 문헌

### 원본 논문

```bibtex
@InProceedings{pmlr-v139-yonetani21a,
  title     = {Path Planning using Neural A* Search},
  author    = {Ryo Yonetani and Tatsunori Taniai and Mohammadamin Barekatain and Mai Nishimura and Asako Kanezaki},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages     = {12029--12039},
  year      = {2021},
  volume    = {139},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {http://proceedings.mlr.press/v139/yonetani21a.html},
}
```

### 관련 연구

- [1] Lisa Lee*, Emilio Parisotto*, Devendra Singh Chaplot, Eric Xing, Ruslan Salakhutdinov, "Gated Path Planning Networks", ICML, 2018.
- [2] Marin Vlastelica Pogančić, Anselm Paulus, Vit Musil, Georg Martius, Michal Rolinek, "Differentiation of Blackbox Combinatorial Solvers", ICLR, 2020.

---

## 📬 문의

- **원본 프로젝트**: [omron-sinicx/neural-astar](https://github.com/omron-sinicx/neural-astar)
- **이슈 리포팅**: GitHub Issues 사용
- **논문 관련 문의**: 원본 논문 저자에게 문의

---

## 📜 라이선스

이 프로젝트는 원본 Neural A\* 프로젝트의 라이선스를 따릅니다.
