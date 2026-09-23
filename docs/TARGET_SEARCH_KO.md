# VEIL-Net 목표 성능 재학습

## 실행 명령

이 실행기는 단일 파일럿이 아니라 **여러 학습 설정과 에폭을 비교해 목표에 맞는 실제 가중치를 선택하는 재학습 실행기**다. 기존 65에폭 체크포인트의 형상 복원 성능을 유지하는 방향으로 추가 학습한다.

```powershell
# Run from the extracted veil-NET directory or the repository root.
$py = 'python'

& $py -u scripts/search_target_metrics.py `
  --run-name veil_target_v2 `
  --max-trials 6 `
  --epochs-per-trial 20 `
  --eval-every 5 `
  --device cuda `
  --eval-device cpu
```

세 가지 설정을 seed 0, 1에서 실행하며, **최대 6회 × 20 추가 에폭**이다. 각 시도는 같은 원본 체크포인트에서 독립적으로 시작한다. 5에폭마다 검증하며 목표를 달성하면 남은 학습은 중단한다. 기존 에폭 기록을 기준으로 전체 예산은 며칠이 걸릴 수 있다. 실행 전 데이터와 예산만 확인하려면 같은 명령에 `--plan-only`를 붙인다.

## 판정 기준

다음 다섯 조건을 **동시에** 만족해야 성공이다. 정확히 같은 소수점을 인위적으로 맞추는 것이 아니라, 보고값 이상의 성능을 목표로 한다.

| 지표 | 조건 |
|---|---:|
| CD-L1 | ≤ 0.38629 |
| CD-L2 | ≤ 6.97737 |
| F@0.03 | ≥ 0.72843 |
| F@0.05 | ≥ 0.89253 |
| Dimension MAE | ≤ 0.22846 |

기존 표의 반올림에 맞춘 허용치는 `1e-5`다. F-score만 좋아지고 CD가 조건을 벗어나면 성공으로 처리하지 않는다. 학습 결과 자체를 미리 보장하지는 않으며, 예산 소진 시 미달 상태와 실제로 가장 가까웠던 가중치를 남긴다.

## 학습 변경

기존 형상 복원 손실에 원래 좌표의 0.03·0.05 임계값을 사용하는 sampled soft F-score 손실을 추가한다. 매 step에서 예측과 정답을 독립적으로 최대 4,096점까지 무작위 추출한다. 학습률은 `1e-5`, FP32와 gradient clipping은 유지한다.

| 설정 | 새 F-score 항 | CD 항 | Surface 항 | Extent 항 |
|---|---:|---:|---:|---:|
| balanced | 1.0 | 1.8 | 0.5 | 0.05 |
| coverage | 2.0 | 2.7 | 0.75 | 0.10 |
| geometry | 1.0 | 3.6 | 1.0 | 0.15 |

F-score 강조에 따른 CD·치수 오차의 악화를 함께 비교할 수 있도록 손실 가중치를 다르게 설정했다. 이 설정들이 목표 달성을 입증한 것은 아니며, 실행기가 실제 결과로 선택한다.

## 평가와 선택

- 학습: 원본 체크포인트에 기록된 8,925개 학습 ID. 검증 ID가 겹치면 중단.
- 평가: 기존 929개 ID, 4,096점, seed 0, 기존 평가 함수 그대로 사용.
- 매 평가 시 실제 checkpoint SHA-256과 데이터 해시 기록.
- 초기 모델도 후보에 포함하며, 다섯 목표 대비 가장 큰 상대 미달 정도가 작은 후보를 우선 보존. 동률이면 미달 합계로 비교.
- 검증 결과가 목표를 만족하면 바로 학습을 중단하고 별도 성공 가중치를 저장.
- 기존 공개 모델, 기존 논문 표, 기존 평가 함수를 변경하지 않음.

이 경로는 validation을 사용한 설정·에폭 선택이다. 따라서 선택된 validation 성능과 별도 test 성능은 구분해서 보고해야 한다.

## 저장 결과

`results/target_search/veil_target_v2/`:

- `winner.pt`: **다섯 목표를 모두 충족했을 때만 생성**.
- `winner_report.json`: 성공 가중치의 실제 전체 검증 결과.
- `best_candidate.pt`: 초기 모델을 포함해 가장 가까웠던 실제 가중치. 이 파일의 존재만으로 목표 달성을 뜻하지 않는다.
- `best_candidate_report.json`, `best_candidate.json`: 선택 결과, 가중치 해시, 목표별 판정.
- `evaluations.json`: 시도·seed·에폭별 모든 평가 기록.
- `plan.json`: 고정된 학습 계획.
- `summary.json`: 성공 시 `targets_met`, 예산 소진 시 `budget_exhausted_targets_not_met`.

종료 코드는 달성 시 0, 예산 내 미달 시 2다. 원본 가중치는 덮어쓰지 않는다.

## 중단 후 재개

```powershell
& $py -u scripts/search_target_metrics.py `
  --run-name veil_target_v2 `
  --max-trials 6 `
  --epochs-per-trial 20 `
  --eval-every 5 `
  --device cuda `
  --eval-device cpu `
  --resume
```

완료된 시도는 건너뛰고, 진행 중이던 시도는 마지막으로 완료한 에폭부터 재개한다. 평가 도중 중단된 폴더는 보존하고 새 retry 폴더에서 평가를 다시 수행한다. 학습 설정을 바꾸려면 새 run name을 사용한다. 재개 시 난수 상태까지 동일한 학습 재현은 보장하지 않는다.

## 독립 재평가

성공 가중치가 생성된 경우 아래 명령으로 같은 검증 조건에서 다시 평가할 수 있다.

```powershell
& $py -m veil_net evaluate `
  --model results/target_search/veil_target_v2/winner.pt `
  --pairs cache/completion_pairs/val `
  --protocol benchmarks/validation_protocol.json `
  --max-samples 0 `
  --eval-points 4096 `
  --device cpu `
  --output results/veil_target_v2_recheck

& $py scripts/verify_paper_results.py `
  --report results/veil_target_v2_recheck/report.json `
  --reference paper `
  --mode target
```
