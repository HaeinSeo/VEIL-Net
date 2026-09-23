# VEIL-Net

**Visible-to-Entire Surface Inference via Local Geometry for Occluded 3D Objects**

부분 관측된 물체 점군에서 가려진 영역을 포함한 전체 표면을 추론합니다.
지역 기하, 좌표 조건부 query, 표면 patch 생성, 관측 표면 보존을 결합해
2,048개 입력점으로 16,384개 표면점을 생성합니다.

[English](README.md) · [빠른 실행](docs/QUICKSTART_KO.md) ·
[결과와 평가 조건](benchmarks/README.md) · [공개 준비](docs/RELEASE.md) ·
[목표 성능 재학습](docs/TARGET_SEARCH_KO.md) · [단일 학습과 파일럿](docs/TARGET_FINETUNE_KO.md)

## 공개 코드와 기준 성능

이 GitHub 저장소에는 코드·문서·평가 기록이 포함되며, 데이터와 가중치는 포함하지 않습니다.
별도의 **veil-NET.zip**을 보유한 경우에는 추론 가중치와 추가 학습 초기화 파일을
사용할 수 있습니다. [영문 학습 안내](docs/TRAINING.md)와
[번들 실행 안내](docs/PUBLIC_BUNDLE_KO.md)를 참고하세요.

현재 기준 가중치의 실제 결과는 **CD-L1 0.38629, CD-L2 6.97737,
F@0.03 0.65844, F@0.05 0.75428, 치수 MAE 0.22846**입니다.
SnowflakeNet FT 대비 **CD-L2 16.41%, 치수 오차 33.27% 감소**가 확인되었습니다.
아래 원고의 외부 F-score 두 개는 이 가중치로 아직 달성하지 않은 목표입니다.
[전체 평가 기록](benchmarks/bundled_evaluation.json)에서 실제 수치를 확인할 수 있습니다.

저장소를 내려받은 뒤 다음 명령으로 설치와 기본 동작을 확인합니다.
별도 가중치와 데이터 없이 실행할 수 있습니다.

```bash
python -m pip install -e ".[dev,hub]"
python -m veil_net smoke
```

## 논문 결과

<!-- BEGIN PAPER RESULTS -->

첨부 원고의 전체 결과이다. 주 비교표의 VEIL-Net은 **외부 실행 보고값**이며, 좌표 조건화는 **50-epoch 제거 실험**, 데이터셋별 결과와 bootstrap은 **저장 체크포인트**에 해당한다.

### 평가 데이터

| Dataset | Samples |
| --- | --- |
| T-LESS | 513 |
| TUD-L | 50 |
| YCB-Video | 366 |
| Total | 929 |

### 전체 모델 비교

| Method | CD-L1 ↓ | CD-L2 ↓ | F@0.03 ↑ | F@0.05 ↑ | Dim. MAE ↓ |
| --- | --- | --- | --- | --- | --- |
| Fusion | 0.53061 | 9.53184 | 0.38348 | 0.47510 | 0.34924 |
| PoinTr | 0.55067 | 9.52597 | 0.27878 | 0.39670 | 0.43411 |
| AdaPoinTr | 0.52486 | 9.62343 | 0.32947 | 0.44170 | 0.42771 |
| SnowflakeNet | 0.51613 | 9.50329 | 0.33311 | 0.46123 | 0.43646 |
| SnowflakeNet FT | 0.42271 | 8.34760 | 0.65293 | 0.75463 | 0.34238 |
| **VEIL-Net (external)** | **0.38629** | **6.97737** | **0.72843** | **0.89253** | **0.22846** |

### SnowflakeNet FT 대비 개선

| Metric | Improvement |
| --- | --- |
| CD-L1 | **↓ 8.62%** |
| CD-L2 | **↓ 16.41%** |
| F@0.03 | **↑ 7.55%p** |
| F@0.05 | **↑ 13.79%p** |
| Dimension MAE | **↓ 33.27%** |

### 좌표 조건화 제거 실험

| Setting | CD-L2 ↓ | F@0.03 ↑ | F@0.05 ↑ | Missing CD ↓ | Observed Error ↓ |
| --- | --- | --- | --- | --- | --- |
| Without coordinate conditioning | 7.48764 | 0.61364 | 0.73557 | 0.15204 | 0.01616 |
| Full VEIL-Net (50 epochs) | 7.04963 | 0.63469 | 0.74313 | 0.13964 | 0.01531 |

좌표 조건화로 **Missing CD 8.15% 감소**, **CD-L2 5.85% 감소**, **F@0.03 2.11%p 증가**가 나타났다.

### 데이터셋별 분석

| Dataset | Samples | Method | CD-L1 ↓ | F@0.03 ↑ | Missing CD ↓ | Dim. MAE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| T-LESS | 513 | SnowflakeNet FT | 0.74103 | 0.45083 | 0.23467 | 0.49639 |
| T-LESS | 513 | VEIL-Net | 0.67720 | 0.45357 | 0.25980 | 0.37363 |
| TUD-L | 50 | SnowflakeNet FT | 0.02462 | 0.90172 | 0.01356 | 0.31718 |
| TUD-L | 50 | VEIL-Net | 0.02208 | 0.89986 | 0.01169 | 0.03166 |
| YCB-Video | 366 | SnowflakeNet FT | 0.03092 | 0.90221 | 0.01839 | 0.12997 |
| YCB-Video | 366 | VEIL-Net | 0.02828 | 0.91260 | 0.01551 | 0.05186 |

세 데이터셋 모두에서 CD-L1과 치수 MAE가 감소했으며, YCB-Video에서는 표의 네 지표가 모두 개선되었다.

### 통계 분석

| Metric | Mean Difference | 95% CI |
| --- | --- | --- |
| CD-L1 | -0.03642 | [-0.11561, 0.01929] |
| CD-L2 | -1.37024 | [-4.47204, 0.45013] |
| F@0.03 | 0.00551 | [0.00101, 0.01031] |
| Dimension MAE | -0.11393 | [-0.19867, -0.06138] |

차이는 저장 VEIL-Net − SnowflakeNet FT이다. Paired scene-level bootstrap에서 F@0.03과 치수 MAE의 95% 신뢰구간은 0을 포함하지 않는다. 데이터셋별 F@0.03의 가중 평균과 bootstrap 평균 차이는 저장본 **0.65844**와 연결되며, 외부 보고값 **0.72843**에 대한 통계 검증으로 사용하지 않는다.

<!-- END PAPER RESULTS -->

전체 한글 결과 해석은 [논문 결과 정리](docs/PAPER_RESULTS_KO.md)에 있다.

## 저장 체크포인트 결과

저장된 동일 validation 929개 관측의 평가에서 SnowflakeNet FT 대비
**CD-L2 16.41% 감소**, **물체 치수 오차 33.27% 감소**를 확인했습니다.

| 모델 | CD-L1 ↓ | CD-L2 ↓ | F@0.03 ↑ | F@0.05 ↑ | 치수 MAE ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SnowflakeNet FT | 0.42271 | 8.34760 | 0.65293 | **0.75463** | 0.34238 |
| **VEIL-Net** | **0.38629** | **6.97737** | **0.65844** | 0.75428 | **0.22846** |

동일 50 epoch 제거 실험에서는 query 좌표 조건화가 missing-region 거리 오차를
8.15% 줄이고, 관측 표면 보존이 observed-region 오차를 51.66% 줄였습니다.
각 수치는 해당 기능을 제거한 모델과 비교한 결과입니다.
전체 지표와 외부 실행 결과의 출처는 [benchmark 문서](benchmarks/README.md)에 있습니다.

## 바로 실행

```bash
python -m pip install -e ".[dev,hub]"
python -m veil_net smoke
```

CPU에서 데이터 없이 순전파·역전파·가중치 저장 및 재로딩을 검사합니다.
학습된 모델의 정확도 검증은 준비된 pair와 체크포인트로 실행합니다.

```bash
python -m veil_net export --checkpoint checkpoints/rapc_coverage_balance/seed_0/last.pt --output releases/veil-model
python -m veil_net evaluate --model releases/veil-model --pairs cache/completion_pairs/val --protocol benchmarks/validation_protocol.json --max-samples 16 --eval-points 1024 --output results/quick-validation
```

결과는 `samples.csv`와 `report.json`에 저장됩니다. 전체 929개 평가에는
`--max-samples 0 --eval-points 4096`을 사용합니다. 출력 폴더는 새 경로를 지정합니다.

## 공개 구조

새 사용자는 `veil_net` API와 CLI에서 시작하면 됩니다. 기존 `rapc_net` 구현과
가중치 이름을 유지해 과거 체크포인트도 그대로 불러옵니다. 모델의 내부
정규화는 입력 관측만 사용하며 추론에 GT를 넣지 않습니다.

GitHub에는 코드·문서·검증용 CSV를, Hugging Face에는 내보낸 모델 폴더를
올릴 수 있도록 구성했습니다. 공개용 코드만 모으는 명령은 다음과 같습니다.

```bash
python scripts/prepare_release.py --output releases/veil-net-github
```

이 명령은 로컬 공개용 폴더를 만들며 Git push나 Hub 업로드는 수행하지 않습니다.
