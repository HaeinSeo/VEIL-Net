# Active VEIL: Building-Rome-in-a-Day-Inspired Multi-View 3D Completion

## 1. Research Goal

기존 **VEIL-Net**의 single-view RGB-D completion 구조를 유지하면서,  
**Building Rome in a Day**의 핵심 아이디어인 **다수의 서로 다른 시점에서 얻은 관측을 하나의 3D 구조로 연결·정렬·통합하는 방식**을 object-level point cloud completion에 적용한다.

최종 목표는 다음과 같다.

```text
Single-view RGB-D
    ↓
VEIL-Net Completion
    ↓
Cross-View Evidence Aggregation
    ↓
Uncertainty Estimation
    ↓
Next-Best-View Selection
    ↓
NVIDIA Isaac Sim Robot Camera Motion
    ↓
Additional RGB-D Observation
    ↓
Completion Refinement
    ↓
Robotic Grasping
```

핵심 방향:

```text
VEIL-Net
    ↓
Rome-style Cross-View Evidence
    ↓
Active View Selection
    ↓
Robot Simulation
```

---

# 2. Building Rome in a Day에서 가져오는 핵심 개념

Building Rome in a Day의 전체 SfM pipeline을 그대로 사용하는 것이 아니다.

본 연구에서는 다음 개념만 object completion 문제에 맞게 가져온다.

## 2.1 Multi-View Evidence

하나의 장면을 한 시점만 보는 것이 아니라,  
서로 다른 camera view가 제공하는 부분적인 geometry를 함께 사용한다.

```text
View 1
View 2
View 3
View 4
   ↓
Shared 3D Structure
```

본 연구에서는 이를 object-level로 바꾼다.

```text
Partial Object View 1
Partial Object View 2
Partial Object View 3
Partial Object View 4
        ↓
Common Object Coordinate
        ↓
Complete Object Geometry
```

---

## 2.2 Cross-View Correspondence

서로 다른 view에서 관측된 local surface가  
동일한 3D surface를 나타내는지 matching한다.

```text
View A local patch
        ↕
3D correspondence
        ↕
View B local patch
```

즉 단순히 point cloud를 합치는 것이 아니라,

```text
"이 두 관측이 같은 surface를 보고 있는가?"
```

를 판단한다.

---

## 2.3 Shared 3D Coordinate

서로 다른 camera coordinate의 point cloud를  
하나의 object canonical coordinate로 정렬한다.

\[
\tilde{P}_v
=
T_{o \leftarrow c}^{(v)} P_v
\]

- \(P_v\): view \(v\)의 partial point cloud
- \(T_{o \leftarrow c}^{(v)}\): camera → object coordinate transform
- \(\tilde{P}_v\): canonical coordinate로 정렬된 point cloud

---

## 2.4 Incremental Reconstruction

여러 view를 한 번에만 사용하는 것이 아니라,  
새로운 관측이 들어올 때마다 geometry를 점진적으로 개선한다.

```text
View 1
 ↓
Initial Completion

View 1 + View 2
 ↓
Refined Completion

View 1 + View 2 + View 3
 ↓
More Complete Geometry
```

이 부분을 active vision과 연결한다.

---

# 3. Base Model: VEIL-Net

기존 VEIL-Net 구조는 유지한다.

```text
Partial Point Cloud
        ↓
Point Encoder
        ↓
FPS + kNN
        ↓
Local Geometry Encoding
        ↓
Observation Quality Estimation
        ↓
Quality-Aware Representation
        ↓
Surface Center Prediction
        ↓
Coordinate-Conditioned Query
        ↓
Transformer Decoder
        ↓
Local Surface Generation
        ↓
Observed + Generated Surface
        ↓
Complete Point Cloud
```

기존 query:

\[
q_i
=
f(\tilde{g}_i,c_i)
\]

의미:

```text
g_i  → What to Generate
c_i  → Where to Generate
```

기존 VEIL-Net의 핵심 구조는 제거하지 않는다.

---

# 4. Proposed Architecture

최종 확장 구조:

```text
                    Query View
                        │
                        ▼
                  VEIL Encoder
                        │
                        ▼
               Local Geometry Tokens
                        │
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 Current View Tokens             Reference Views
                                        │
                                        ▼
                                 VEIL Encoder
                                        │
                                        ▼
                               Reference Tokens
                                        │
        ┌───────────────────────────────┘
        ▼
Canonical Coordinate Alignment
        ↓
Cross-View Correspondence
        ↓
Cross-View Evidence Memory
        ↓
Evidence-Gated VEIL Query
        ↓
Existing VEIL Decoder
        ↓
Completed Surface
        ↓
Surface Uncertainty
        ↓
Next-Best-View
        ↓
NVIDIA Isaac Sim
        ↓
New RGB-D Observation
        ↓
Repeat
```

---

# 5. Module 1 — View-Wise VEIL Encoding

각 view는 동일한 VEIL encoder를 사용한다.

\[
P_v
\rightarrow
\{g_{v,i},r_{v,i},a_{v,i}\}
\]

각 token:

\[
z_{v,i}
=
[g_{v,i},r_{v,i},a_{v,i}]
\]

구성:

- \(g_{v,i}\): local geometry feature
- \(r_{v,i}\): observation reliability
- \(a_{v,i}\): local anchor coordinate

모든 view는 weight를 공유한다.

```text
View 1 ─┐
View 2 ─┼→ Shared VEIL Encoder
View 3 ─┤
View 4 ─┘
```

---

# 6. Module 2 — Canonical View Alignment

각 view의 point와 anchor를 object coordinate로 변환한다.

\[
\tilde{P}_v
=
T_{o \leftarrow c}^{(v)}P_v
\]

\[
\tilde{a}_{v,i}
=
T_{o \leftarrow c}^{(v)}a_{v,i}
\]

이후 모든 view는 동일한 3D object coordinate 안에서 비교한다.

---

# 7. Module 3 — Cross-View Correspondence

현재 query view의 local region과  
다른 view의 local region 사이 correspondence를 계산한다.

spatial distance:

\[
d_{ij}^{(t,v)}
=
\|
\tilde{a}_{t,i}
-
\tilde{a}_{v,j}
\|_2
\]

feature similarity:

\[
s_{ij}^{feat}
=
cos(
g_{t,i},
g_{v,j}
)
\]

reliability-aware matching:

\[
m_{ij}
=
\sigma
\left(
MLP[
d_{ij},
s_{ij}^{feat},
r_{t,i},
r_{v,j}
]
\right)
\]

최종 matching은 다음 정보를 사용한다.

```text
3D Spatial Distance
+
Feature Similarity
+
Observation Reliability
+
Optional Surface Normal Similarity
```

---

# 8. Module 4 — Cross-View Evidence Memory

matching된 다른 view의 local features를 attention으로 통합한다.

\[
M_i
=
\sum_{v,j}
\alpha_{ij}^{(v)}
W z_{v,j}
\]

attention:

\[
\alpha_{ij}^{(v)}
=
softmax
\left(
\frac{
Q(z_{t,i})^T K(z_{v,j})
}{
\sqrt{d}
}
+
\beta m_{ij}
\right)
\]

의미:

```text
Current Local Geometry
        +
Matched Geometry from Other Views
        +
Observation Reliability
        ↓
Cross-View Evidence Memory
```

단순 point union보다 각 view의 geometry 관계를 명시적으로 사용한다.

---

# 9. Module 5 — Evidence-Gated VEIL Query

기존 VEIL query는 그대로 유지한다.

\[
q_i
=
f(\tilde{g}_i,c_i)
\]

cross-view evidence를 residual 형태로 추가한다.

\[
q_i^*
=
q_i
+
\gamma_i M_i
\]

gate:

\[
\gamma_i
=
\sigma
\left(
MLP[
r_i,
M_i
]
\right)
\]

동작:

```text
Evidence 부족
→ 기존 VEIL query 유지

Reliable cross-view evidence 존재
→ 다른 view geometry를 query에 반영
```

따라서 기존 VEIL-Net의 구조를 최대한 보존한다.

---

# 10. Module 6 — Incremental View Memory

한 번 사용한 reference view의 evidence를 저장한다.

\[
\mathcal{M}_t
=
\mathcal{M}_{t-1}
\cup
M_t
\]

전체 과정:

```text
View 1
 ↓
Memory M1

View 2
 ↓
M1 + M2

View 3
 ↓
M1 + M2 + M3
```

단, 모든 token을 그대로 저장하지 않고 다음 기준으로 유지한다.

```text
High Reliability
+
High Surface Novelty
+
Non-Redundant Geometry
```

중복 token은 merge한다.

---

# 11. Module 7 — Surface Uncertainty

각 generated surface patch에 uncertainty를 예측한다.

\[
u_i
=
f_{unc}(q_i^*)
\]

두 종류로 구분한다.

## Geometry Uncertainty

\[
u_i^{geo}
\]

생성된 shape 자체의 불확실성.

## Evidence Uncertainty

\[
u_i^{evi}
\]

해당 영역을 지지하는 실제 관측 evidence가 부족한 정도.

최종:

\[
u_i
=
\lambda_g u_i^{geo}
+
\lambda_e u_i^{evi}
\]

---

# 12. Module 8 — Next-Best-View

현재 uncertainty가 높은 surface를 실제로 확인할 수 있는 camera view를 선택한다.

candidate view:

\[
\mathcal{V}
=
\{v_1,\dots,v_K\}
\]

각 candidate에 대해 계산한다.

## Uncertainty Coverage

\[
G_{unc}(v)
=
\sum_i
u_i V_i(v)
\]

## Surface Novelty

\[
G_{nov}(v)
\]

현재까지 보지 못한 surface가 얼마나 많이 보이는지 측정한다.

## Visibility Gain

\[
G_{vis}(v)
\]

## Motion Cost

\[
C_{motion}(v)
\]

## Collision Cost

\[
C_{col}(v)
\]

최종 score:

\[
S(v)
=
\lambda_1G_{unc}(v)
+
\lambda_2G_{nov}(v)
+
\lambda_3G_{vis}(v)
-
\lambda_4C_{motion}(v)
-
\lambda_5C_{col}(v)
\]

선택:

\[
v^*
=
\arg\max_{v \in \mathcal V}S(v)
\]

---

# 13. Active Reconstruction Loop

```text
Initial RGB-D
    ↓
VEIL-Net Completion
    ↓
Cross-View Memory
    ↓
Surface Uncertainty
    ↓
Generate Candidate Views
    ↓
NBV Selection
    ↓
Robot Camera Movement
    ↓
New RGB-D Observation
    ↓
Canonical Alignment
    ↓
Cross-View Evidence Update
    ↓
Completion Refinement
    ↓
Stop or Repeat
```

수식:

\[
Q_t
=
F(
P_{1:t},
\mathcal{M}_t
)
\]

\[
v_{t+1}
=
\arg\max_v
S(v|Q_t,U_t)
\]

\[
P_{t+1}
=
Observe(v_{t+1})
\]

\[
Q_{t+1}
=
F(
P_{1:t+1},
\mathcal{M}_{t+1}
)
\]

---

# 14. Stop Condition

다음 중 하나를 만족하면 추가 관측을 종료한다.

## Low Uncertainty

\[
\frac{1}{N}
\sum_i u_i
<
\tau_u
\]

## Sufficient Coverage

\[
Coverage(Q_t)
>
\tau_c
\]

## Maximum View Budget

```text
Maximum additional views = 2~3
```

---

# 15. NVIDIA Isaac Sim

로봇 active perception은 **NVIDIA Isaac Sim**에서 구현한다.

## Simulation Components

```text
NVIDIA Isaac Sim
├── Franka Panda
├── RGB-D Camera
├── PhysX
├── Object Assets
├── Tabletop Scene
├── Camera Pose Controller
├── Collision Checker
├── VEIL-Net Inference
├── NBV Module
└── Grasp Planner
```

---

# 16. Robot Setup

기본 robot:

```text
Franka Panda
```

camera:

```text
Wrist-mounted RGB-D Camera
```

환경:

```text
Table
+
Target Object
+
Distractor Objects
+
Occlusion
```

전체 구조:

```text
Robot Arm
   │
RGB-D Camera
   │
   ▼
Target Object
   │
Partial Point Cloud
   │
   ▼
Active VEIL
```

---

# 17. Isaac Sim Scene Generation

scene마다 다음을 randomize한다.

```text
Target Object Pose
Distractor Positions
Camera Initial Pose
Lighting
Object Occlusion
Object Distance
Robot Configuration
```

목표 visibility:

```text
20% ~ 70%
```

초기 관측에서 object가 완전히 보이지 않도록 설정한다.

---

# 18. Candidate Camera Views

target object 중심 주변에 hemisphere candidate를 생성한다.

azimuth:

\[
\phi
=
0^\circ,30^\circ,\dots,330^\circ
\]

elevation:

\[
\theta
=
15^\circ,30^\circ,45^\circ,60^\circ
\]

distance:

\[
r
\in
[r_{min},r_{max}]
\]

초기 candidate 수:

```text
32 ~ 48
```

candidate filtering:

```text
IK Reachable
+
No Collision
+
Target in Camera FOV
+
Minimum Visibility
```

---

# 19. Robot Active View Experiment

각 episode:

```text
1. Random initial camera pose
2. RGB-D capture
3. Partial point cloud extraction
4. VEIL completion
5. Surface uncertainty
6. NBV selection
7. Robot movement
8. New RGB-D capture
9. Cross-view evidence update
10. Completion refinement
11. Repeat maximum 2~3 times
12. Grasp planning
13. Grasp execution
```

---

# 20. Dataset Usage

## GraspNet

Main dataset.

사용:

```text
Multi-view training
Cross-view correspondence
View diversity learning
Offline NBV
Grasp evaluation
```

특히 여러 camera view를 이용해 Rome-style multi-view geometry를 학습한다.

---

## T-LESS

사용:

```text
Canonical geometry
View diversity
Symmetric object evaluation
Texture-less objects
```

single-view ambiguity와 multi-view evidence 효과를 분석한다.

---

## YCB-Video

사용:

```text
Temporal multi-view
Incremental evidence accumulation
Real RGB-D validation
```

연속 frame을 활용한다.

```text
Frame t
Frame t+Δ
Frame t+2Δ
```

---

## HOPE

사용:

```text
Robot-camera sequence evaluation
Real-world multi-view validation
Active observation evaluation
```

training보다는 test 중심으로 사용한다.

---

## LM-O

사용:

```text
Heavy occlusion stress test
Hidden surface uncertainty
External generalization
```

---

## UWIS_Occluded

사용:

```text
Occlusion-level generalization
```

평가:

```text
Low Occlusion
Medium Occlusion
Heavy Occlusion
```

---

## OCID

사용:

```text
Clutter
Multiple objects
Occlusion boundary
Robot-like tabletop environment
```

---

## TUD-L

사용:

```text
Lighting robustness
Small-scale external validation
```

---

## HB / HBTLESS

optional pretraining.

사용:

```text
Industrial objects
Domain diversity
Cross-view pretraining
```

---

# 21. Training Loss

전체 loss:

\[
\mathcal L
=
\lambda_{cd}\mathcal L_{CD}
+
\lambda_{miss}\mathcal L_{miss}
+
\lambda_{obs}\mathcal L_{obs}
+
\lambda_{dim}\mathcal L_{dim}
+
\lambda_{match}\mathcal L_{match}
+
\lambda_{evi}\mathcal L_{evidence}
+
\lambda_{unc}\mathcal L_{unc}
+
\lambda_{cons}\mathcal L_{cons}
\]

---

## Completion Loss

\[
\mathcal L_{CD}
=
CD(Q,G)
\]

---

## Missing Surface Loss

\[
\mathcal L_{miss}
=
CD(
Q_{missing},
G_{missing}
)
\]

---

## Observed Surface Preservation

\[
\mathcal L_{obs}
=
CD(
Q_{observed},
P
)
\]

---

## Dimension Loss

\[
\mathcal L_{dim}
=
\|
D(Q)-D(G)
\|_1
\]

---

## Cross-View Matching Loss

\[
y_{ij}
=
\mathbf{1}
[
\|
\tilde a_i-\tilde a_j
\|
<
\tau
]
\]

\[
\mathcal L_{match}
=
BCE(
m_{ij},
y_{ij}
)
\]

---

## Cross-View Evidence Consistency

\[
\mathcal L_{evidence}
=
\|
z_{v,i}
-
z_{u,j}
\|_2
\]

동일 surface에 대응되는 token 사이에 적용한다.

---

## Multi-View Completion Consistency

\[
\mathcal L_{cons}
=
CD(
Q^{(v_1)},
Q^{(v_2)}
)
\]

---

# 22. Evaluation Metrics

## Completion

```text
CD-L1 ↓
CD-L2 ↓
F@0.03 ↑
F@0.05 ↑
Missing CD ↓
Observed Error ↓
Dimension MAE ↓
```

## Active View

```text
Completion Gain / View
Missing-CD Gain / View
Coverage Gain
Number of Views
Camera Travel Distance
NBV Selection Time
```

## Uncertainty

```text
Error-Uncertainty Correlation
Calibration Error
High-Uncertainty Region Recall
```

## Robotics

```text
Grasp Success Rate
Planning Success Rate
Collision Rate
Execution Success Rate
Views Before Grasp
Total Camera Motion
```

---

# 23. Main Experiments

## E1. VEIL vs Rome-Style Cross-View VEIL

```text
VEIL-Net
vs
VEIL + Cross-View Evidence
```

---

## E2. Number of Views

```text
1 View
2 Views
3 Views
4 Views
```

평가:

```text
CD-L1
Missing CD
F@0.03
Coverage
```

---

## E3. Fusion Strategy

```text
Point Union
Feature Average
Cross-Attention
Spatial Cross-Attention
Proposed Evidence Memory
```

---

## E4. Correspondence Ablation

```text
Feature only
Spatial only
Spatial + Feature
Spatial + Feature + Reliability
Full
```

---

## E5. Incremental Reconstruction

```text
Initial View
+
Additional View 1
+
Additional View 2
```

각 단계에서 completion improvement 측정.

---

## E6. NBV Baselines

```text
Random
Maximum View Distance
Maximum Visibility
Maximum Uncertainty
Proposed
Oracle
```

---

## E7. Robot Simulation

```text
Raw Partial
VEIL
Cross-View VEIL
Active VEIL
```

비교:

```text
Completion Quality
Grasp Success
Collision Rate
Required Views
Camera Motion
```

---

# 24. Ablation

| Setting | Local Geometry | Quality | Coordinate Query | Cross-View | Memory | Uncertainty | NBV |
|---|---:|---:|---:|---:|---:|---:|---:|
| VEIL | ✓ | ✓ | ✓ |  |  |  |  |
| + Cross-View | ✓ | ✓ | ✓ | ✓ |  |  |  |
| + Memory | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |
| + Uncertainty | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| Active VEIL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

# 25. Implementation Order

## Phase 1 — Freeze Current VEIL

```text
Current VEIL checkpoint
Current evaluation protocol
Current metrics
```

기존 결과를 baseline으로 고정한다.

---

## Phase 2 — Multi-View Data Loader

출력 형식:

```python
{
    "query_points": ...,
    "query_pose": ...,
    "reference_points": [...],
    "reference_poses": [...],
    "gt_complete": ...
}
```

---

## Phase 3 — Canonical Alignment

```text
camera frame
→ object frame
```

pose transform 검증.

---

## Phase 4 — Cross-View Correspondence

순서:

```text
radius matching
→ spatial + feature
→ learned matching
```

---

## Phase 5 — Evidence Memory

```text
Cross-Attention
+
Reliability Weighting
+
Spatial Matching
```

---

## Phase 6 — Evidence-Gated Query

\[
q_i^*
=
q_i+\gamma_iM_i
\]

기존 decoder는 변경하지 않는다.

---

## Phase 7 — Incremental Memory

새로운 view가 들어올 때 memory update.

---

## Phase 8 — Uncertainty

surface-level uncertainty head 추가.

---

## Phase 9 — Offline NBV

먼저 실제 robot 없이 dataset frame 안에서 평가한다.

예:

```text
GraspNet View 1
    ↓
NBV predicts View 87
    ↓
실제 View 87 point cloud 추가
    ↓
Completion improvement 측정
```

---

## Phase 10 — NVIDIA Isaac Sim

```text
Franka Panda
+
RGB-D Camera
+
Target / Distractor Objects
+
NBV Controller
```

---

## Phase 11 — Robotic Grasp

```text
Completion
→ Grasp Planning
→ Simulation Execution
```

---

# 26. Recommended Repository Structure

```text
active-veil/
│
├── configs/
│   ├── datasets/
│   ├── model/
│   ├── nbv/
│   └── isaac/
│
├── datasets/
│   ├── graspnet/
│   ├── tless/
│   ├── ycb_video/
│   ├── hope/
│   ├── lmo/
│   ├── ocid/
│   └── uwis/
│
├── models/
│   ├── veil/
│   ├── cross_view/
│   │   ├── matcher.py
│   │   ├── memory.py
│   │   └── evidence_gate.py
│   ├── uncertainty/
│   │   └── uncertainty_head.py
│   └── active_veil.py
│
├── geometry/
│   ├── transforms.py
│   ├── correspondence.py
│   └── visibility.py
│
├── nbv/
│   ├── candidates.py
│   ├── scoring.py
│   └── selector.py
│
├── simulation/
│   └── isaac_sim/
│       ├── scene.py
│       ├── robot.py
│       ├── camera.py
│       ├── controller.py
│       └── grasp_eval.py
│
├── train/
│   ├── train_veil.py
│   └── train_active_veil.py
│
├── eval/
│   ├── completion.py
│   ├── multiview.py
│   ├── nbv.py
│   └── grasp.py
│
├── scripts/
│
└── plan.md
```

---

# 27. Final Research Story

기존 VEIL-Net:

```text
Visible Surface
    ↓
Infer Hidden Surface
```

확장 모델:

```text
Visible Surface
    ↓
Infer Hidden Surface
    ↓
Connect Evidence Across Views
    ↓
Estimate What Is Still Uncertain
    ↓
Move the Camera to Where Evidence Is Missing
    ↓
Observe
    ↓
Refine Geometry
    ↓
Grasp
```

최종적으로 본 연구는

\[
\boxed{
\text{Visible-to-Entire}
\rightarrow
\text{Cross-View Evidence}
\rightarrow
\text{Active Observation}
\rightarrow
\text{Robot Action}
}
\]

으로 VEIL-Net을 확장한다.

---

# 28. Core Contribution

### 1. Rome-Style Cross-View Geometry

다른 시점의 local geometry를 canonical 3D coordinate에서 matching하고  
VEIL query에 직접 연결한다.

### 2. Incremental Evidence Memory

새로운 RGB-D observation이 들어올 때마다  
기존 completion을 버리지 않고 surface evidence를 누적한다.

### 3. Evidence-Gated Completion

불확실하거나 잘못 정렬된 view가 기존 VEIL prediction을 망치지 않도록  
cross-view evidence를 gated residual 형태로 사용한다.

### 4. Uncertainty-Guided Active Observation

completion uncertainty가 높은 영역을 실제 camera observation으로 확인한다.

### 5. Robotic Validation

NVIDIA Isaac Sim에서 active RGB-D perception과 grasp까지 연결하여  
completion improvement가 실제 robot action에도 유효한지 평가한다.

---

# 29. One-Line Summary

> **Active VEIL extends single-view Visible-to-Entire Surface Inference into a Building-Rome-in-a-Day-inspired incremental 3D reconstruction framework that aligns and aggregates cross-view geometric evidence, estimates hidden-surface uncertainty, and actively acquires new RGB-D observations for robotic completion and grasping.**
