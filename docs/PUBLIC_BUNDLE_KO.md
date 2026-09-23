# veil-NET 공개 패키지

`veil-NET.zip`은 **코드와 실행 가능한 기준 가중치를 포함한 단일 배포 파일**이다. 압축을 풀면 `veil-NET/` 폴더 하나가 생긴다. 데이터셋 원본과 준비된 point cloud는 포함하지 않는다.

## 포함 내용

```text
veil-NET/
  veil_net/                    공개 API와 CLI
  rapc_net/                    가중치 호환 모델·손실·학습 구현
  model/                       safetensors, config, model card, 실제 평가 기록
  weights/initial.pt           동일 가중치의 추가 학습 초기화 파일
  configs/release/              공개본 내부 상대 경로를 쓰는 학습 설정
  scripts/                     재학습, 평가 판정, 배포 검증
  benchmarks/                  전체 표, frozen protocol, 실제 평가 기록
  tests/, docs/                회귀 테스트와 사용 문서
  PUBLICATION_STATUS.json      포함된 가중치의 목표 달성 여부
  release_manifest.json        배포 파일의 SHA-256
```

`model/`은 추론과 Hugging Face용이다. `weights/initial.pt`는 같은 모델 텐서와 학습 ID를 포함하며 optimizer는 제외한다. 추가 학습은 새 optimizer로 시작한다. 내부 `rapc_net` 이름은 과거 가중치와의 호환을 위해 유지하고 공개 API는 `VEILNet`으로 통일한다.

## 현재 성능

| 지표 | 실제 검증값 | 원고의 목표 |
|---|---:|---:|
| CD-L1 | 0.38629 | ≤ 0.38629 |
| CD-L2 | 6.97737 | ≤ 6.97737 |
| F@0.03 | 0.65844 | ≥ 0.72843 |
| F@0.05 | 0.75428 | ≥ 0.89253 |
| Dimension MAE | 0.22846 | ≤ 0.22846 |

929개 validation 관측의 기준 결과다. **원고의 두 F-score 목표는 아직 이 가중치로 달성하지 않았다.** 원고 전체 표는 `benchmarks/paper_*.csv`와 README의 외부 보고 결과로 보존했다. 현재 확인된 강점은 SnowflakeNet FT 대비 CD-L2 16.41%, 치수 MAE 33.27% 감소다. 추가 학습은 목표 이상을 탐색하지만 결과를 보장하지 않는다.

## 설치와 추론

압축을 푼 폴더에서 Python 3.10 이상으로 실행한다. GPU 학습에는 CUDA 사용이 가능한 PyTorch 환경이 필요하다.

```powershell
python -m pip install -e ".[dev,hub]"
python scripts/verify_bundle.py
python -m veil_net smoke
python -m veil_net infer --model model --input partial.npy --output results/completion.npz
```

`verify_bundle.py`는 파일 해시를 검사한다. `integrity_passed`는 파일 무결성이고, `paper_targets_met`은 별도의 성능 상태다. 소스를 수정하면 해당 파일의 해시 검사는 실패할 수 있다. 입력은 분할된 한 물체의 XYZ 점군이며, RGB-D segmentation은 모델 외부 단계다.

## 동일 조건 평가

기존과 동일하게 준비된 validation pair를 `--pairs` 경로에 둔다. 관측 ID뿐 아니라 점 구성과 좌표 단위도 같아야 한다. [데이터 계약](DATA_FORMAT.md)을 참고한다.

```powershell
python -m veil_net evaluate --model model --pairs cache/completion_pairs/val --protocol benchmarks/validation_protocol.json --max-samples 0 --eval-points 4096 --device cpu --output results/full-validation
python scripts/verify_paper_results.py --report results/full-validation/report.json --reference saved
python scripts/verify_paper_results.py --report results/full-validation/report.json --reference paper --mode target
```

현재 가중치에서는 saved 검사는 통과하고 paper target 검사는 두 F-score 때문에 미달이다. 숫자를 맞추기 위해 평가 점 수·임계값·관측 집합을 바꾸지 않는다.

## 목표 성능 재학습

기존 train/val pair가 필요하다. 포함된 초기 가중치의 학습 ID 목록을 이용해 파일 누락과 검증 ID 중복을 검사한다.

```powershell
python -u scripts/search_target_metrics.py --run-name veil_public_v1 --max-trials 6 --epochs-per-trial 20 --eval-every 5 --device cuda --eval-device cpu
```

최대 6회 × 20 추가 에폭이며 여러 날이 걸릴 수 있다. 먼저 `--plan-only`로 계획을 확인할 수 있다. 중단 후 동일 명령에 `--resume`을 추가한다. `results/target_search/veil_public_v1/winner.pt`는 다섯 목표를 모두 충족했을 때만 생성된다. 미달이면 `best_candidate.pt`와 실제 평가 기록만 보존한다. Validation으로 설정과 에폭을 선택하므로 별도 test 결과와 구분해 보고한다.

## 공개와 재생성

GitHub에는 이 폴더의 코드·문서·테스트를 사용한다. `.gitignore`는 가중치·데이터·실험 출력을 제외한다. Hugging Face에는 `model/`의 가중치·설정·model card·평가 기록을 사용한다. 현재 실제 성능과 외부 보고값 구분을 유지한다. 업로드는 자동 실행하지 않는다. 원본 데이터와 타 연구팀의 구현·가중치는 포함하지 않는다.

연구 작업 폴더에서 새 배포본을 만들려면 다음 명령을 사용한다.

```powershell
python scripts/build_public_bundle.py --checkpoint checkpoints/rapc_coverage_balance/seed_0/last.pt --model releases/veil-model --report results/release_full_reproduction_20260920/report.json --output releases/veil-NET
```

기존 ZIP이나 폴더는 덮어쓰지 않는다. 다섯 목표를 달성한 배포본만 허용하려면 `--require-targets`를 추가한다. 현재 기준 가중치에는 이 옵션이 실패하는 것이 정상이다.
