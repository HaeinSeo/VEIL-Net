# VEIL-Net 목표 지표 추가 학습

## 목표와 현재 기준

이 경로는 **실제 예측 성능 개선**을 위한 추가 학습이다. 평가값을 치환하거나 평가 기준을 완화하지 않는다. 기존 공개 모델과 논문 표는 수정하지 않으며, 새 결과는 별도 후보로 저장한다.

| 지표 | 검증된 기존 가중치 | 새 학습의 목표 |
|---|---:|---:|
| CD-L1 | 0.38629 | ≤ 0.38629 |
| CD-L2 | 6.97737 | ≤ 6.97737 |
| F@0.03 | 0.65844 | ≥ 0.72843 |
| F@0.05 | 0.75428 | ≥ 0.89253 |
| Dimension MAE | 0.22846 | ≤ 0.22846 |

목표는 소수점 다섯 자리까지 똑같은 숫자를 만들라는 뜻이 아니라 **다섯 지표 모두 목표 이상을 달성**하는 것이다. 추가 학습이 이를 보장하지는 않는다. 기존 외부 보고 F-score는 아직 현재 가중치로 재현되지 않았다.

## 변경 내용

기존 정규화 좌표의 형상 손실은 유지하고, 원래 좌표의 0.03·0.05 임계값에 대한 미분 가능한 F-score 손실을 추가했다. 매 학습 단계마다 예측과 정답에서 독립적으로 최대 4,096개 점을 새로 뽑는다. 이는 평가 시 점 개수가 줄어들어도 표면 정밀도와 커버리지를 유지하도록 학습하기 위한 항이다. 학습 손실은 sigmoid 근사이고, **보고하는 평가는 기존의 엄격한 거리 임계값 판정을 그대로 사용한다.**

- 검증된 65에폭 가중치에서 시작, 기본 10에폭 추가 학습.
- 학습률 `1e-5`, 새 F-score 항 가중치 `0.5`, FP32, gradient clipping 유지.
- 원본 체크포인트 SHA-256과 모델 설정 확인.
- 원본 체크포인트의 학습 ID 목록만 사용하며 검증 ID와 겹치면 중단.
- 검증 집합은 기존 929개 ID, 평가 점 수 4,096개, seed 0으로 고정.
- 기존 모델과 새 후보를 같은 기기와 같은 관측 데이터로 평가하고 데이터 해시 확인.
- 검증 결과는 역전파, 학습 데이터 추가, 에폭별 자동 선택에 사용하지 않음.
- 원본 공개 가중치에 덮어쓰지 않음.

검증 집합은 개발 비교용이다. 논문 최종 성능 주장은 설정을 확정한 뒤 별도 test 집합으로도 확인한다.

## 전체 실행

현재 작업 폴더에서 PowerShell로 실행한다. 기존 학습 데이터와 원본 체크포인트가 필요하다.

```powershell
# Run from the extracted veil-NET directory or the repository root.
$py = 'python'

& $py -u scripts/finetune_target_metrics.py `
  --run-name veil_metric_full_v1 `
  --additional-epochs 10 `
  --device cuda `
  --eval-device cpu
```

기존 기록의 전체 1에폭은 약 28~34분이며, 새 손실이 추가되어 실제 시간은 달라질 수 있다. 10에폭은 수 시간이 걸릴 수 있다. 위 명령은 기존 모델 평가, 추가 학습, 후보 평가, 목표 판정까지 수행한다.

이 명령의 epoch 수는 **추가 에폭 수**다. 기존 `veil train --epochs`가 사용하는 총 에폭 수와 혼동하지 않는다.

## 빠른 확인

```powershell
& $py -u scripts/finetune_target_metrics.py `
  --run-name veil_metric_pilot_v1 `
  --additional-epochs 1 `
  --max-train-samples 64 `
  --max-eval-samples 32 `
  --device cuda `
  --eval-device cpu
```

짧은 실행도 실제 가중치를 업데이트하고 전후 성능을 계산한다. 다만 상태는 `pilot_only`이며 **논문 전체 결과로 사용하지 않는다.** 실제 전체 학습은 별도 run name으로 원본 가중치에서 시작한다.

## 결과와 재개

`results/target_finetune/<run-name>/seed_0/`에 다음 파일이 생긴다.

- `run.json`, `training.json`: 실행 설정과 학습 ID 목록.
- `baseline/report.json`, `candidate/report.json`: 평가값, 관측 ID, 데이터 해시, 가중치 해시.
- `baseline/samples.csv`, `candidate/samples.csv`: 모든 관측별 실제 평가값.
- `summary.json`: 지표별 전후 차이와 다섯 목표의 충족 여부.

가중치는 `checkpoints/<run-name>/seed_0/last.pt`에 저장된다. `best.pt`는 학습 손실 기준이며, 비교 보고서는 마지막 에폭의 `last.pt`를 사용한다.

전체 실행에서 `targets_met`일 때만 종료 코드 0, 미달이면 `targets_not_met`와 종료 코드 2를 반환한다. 미달이어도 결과는 보존한다. 일부 지표만 좋아진 경우에도 전체 달성으로 표시하지 않는다. 표의 반올림 오차를 고려하는 허용치는 `1e-5`다.

학습 중단 후에는 **동일한 실행 인자에 `--resume`만 추가**한다. 마지막으로 완료한 에폭부터 이어 간다. 재개 시 난수 상태까지 이어지는 bitwise-identical 학습은 보장하지 않는다. 완료된 실험이나 불완전한 평가 폴더는 덮어쓰지 않으며, 그런 경우 새 run name을 사용한다.

## 목표 판정만 실행

```powershell
& $py scripts/verify_paper_results.py `
  --report results/target_finetune/veil_metric_full_v1/seed_0/candidate/report.json `
  --reference paper `
  --mode target
```

`--mode target`은 목표 이상 여부를 확인한다. 기존 기본 모드인 `--mode reproduce`는 보고된 숫자와의 일치 여부를 확인하며 동작이 바뀌지 않았다.

## 실제 파일럿 확인

`veil_metric_pilot_20260920`에서 학습 64개, 검증 32개, 추가 1에폭을 실행했다. GPU에서 optimizer update 32회가 수행되었으며 누락된 update는 0회였다. 학습 에폭 자체는 약 30.5초였다. 모델 로딩과 전후 CPU 평가는 별도 시간이 필요하다.

| 지표 | 학습 전 | 학습 후 | 변화 |
|---|---:|---:|---|
| CD-L1 | 0.178572 | 0.178980 | 소폭 악화 |
| CD-L2 | 0.091368 | 0.091771 | 소폭 악화 |
| F@0.03 | 0.661502 | 0.662211 | +0.071%p |
| F@0.05 | 0.770950 | 0.771333 | +0.038%p |
| Dimension MAE | 0.149410 | 0.146902 | 감소 |

이는 같은 32개 관측에서 비교한 파일럿이며 929개 전체의 논문 수치와 직접 비교할 수 없다. **목표 달성 결과가 아니며, 전체 10에폭 학습은 아직 실행하지 않았다.** 확인된 사실은 새 손실이 실제 가중치 업데이트에 사용되고 전후 평가와 미달 판정까지 정상 작동한다는 것이다.

실제 상세 결과: `results/target_finetune/veil_metric_pilot_20260920/seed_0/summary.json`.
실제 학습 항 로그: `results/full/training/veil_metric_pilot_20260920/seed_0/loss_components.csv`.
