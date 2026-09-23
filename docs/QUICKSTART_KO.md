# 빠른 검증

가중치가 포함된 `veil-NET.zip` 사용자는 [공개본 실행 안내](PUBLIC_BUNDLE_KO.md)를
먼저 참고하세요. 아래 내보내기 예시는 원래 연구 작업 폴더를 위한 명령입니다.

## 1. 설치와 소프트웨어 검사

프로젝트 루트에서 CPU/GPU에 맞는 PyTorch를 설치한 Python으로 실행합니다.

```bash
python -m pip install -e ".[dev,hub]"
python -m veil_net smoke
```

`status: passed`가 나오면 작은 geometry-query 모델의 순전파, 역전파,
safetensors 저장·재로딩이 통과한 것입니다. 이 단계는 데이터 없이 실행됩니다.

## 2. 실제 학습 가중치 내보내기

```bash
python -m veil_net export --checkpoint checkpoints/rapc_coverage_balance/seed_0/last.pt --output releases/veil-model
```

원본 체크포인트의 config를 그대로 사용하며, 내보낸 모든 텐서의 일치를
검사합니다. 결과 폴더에는 모델 카드와 SHA-256 출처 정보가 함께 들어갑니다.
이미 있는 폴더에는 덮어쓰지 않으므로 재실행 시 새 출력 경로를 지정합니다.

## 3. 실제 validation 일부 검사

```bash
python -m veil_net evaluate --model releases/veil-model --pairs cache/completion_pairs/val --protocol benchmarks/validation_protocol.json --max-samples 16 --eval-points 1024 --output results/quick-validation
```

고정된 validation 목록에서 seed 0으로 16개를 뽑아 재추론합니다. 샘플별
수치는 `samples.csv`, 평균과 평가 조건은 `report.json`에서 확인합니다.
일부 샘플·1,024점 평가이므로 README의 전체 평가 수치와는 조건이 다릅니다.

전체 저장본 조건으로 검증하려면 다음과 같이 실행합니다.

```bash
python -m veil_net evaluate --model releases/veil-model --pairs cache/completion_pairs/val --protocol benchmarks/validation_protocol.json --max-samples 0 --eval-points 4096 --output results/full-validation
```

## 4. 공개 API 회귀 검사

```bash
python -m pytest -q tests/test_public_api.py tests/test_public_cli.py tests/test_public_release.py tests/test_metrics.py tests/test_geometry_queries.py
```

CPU 검증이 기본입니다. `--device cuda`를 지정하면 사용 가능한 CUDA 장치에서
실행하며, 사용할 수 없을 때는 명확한 오류를 반환합니다.
