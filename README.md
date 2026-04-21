# 🚗 YOLOv5 Pothole Detector: From Scratch to Real-world Detection

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-red.svg)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple.svg)

---

## 📌 프로젝트 요약 (Executive Summary)
본 프로젝트는 객체 탐지(Object Detection)를 **핵심 원리 구현 → 단일 객체 탐지 모델 설계 → 실전 SOTA 모델 적용**의 3단계로 심화 학습한 포트폴리오입니다. BBox 연산과 NMS 알고리즘을 밑바닥부터 직접 구현하며 탐지의 수학적 원리를 체득하고, ResNet18 기반 멀티태스크 탐지 모델을 직접 설계한 뒤, 실무 수준의 YOLOv5로 실제 도로 포트홀 영상에서 실시간 탐지를 수행하였습니다. **"라이브러리 뒤에 숨겨진 원리를 이해하고 쓰는 것"** 과 단순 사용의 차이를 스스로 증명하려 노력했습니다.

---

## 🎯 핵심 목표 (Motivation)
| 핵심 역량 | 적용 단계 | 상세 목표 및 엔지니어링 포인트 |
| :--- | :---: | :--- |
| **수학적 원리 구현<br>(Core Algorithm)** | **Stage 1. BBox & Ops** | BBox 포맷 변환, IoU 계산, NMS 알고리즘을 외부 라이브러리 없이 순수 NumPy로 직접 구현 |
| **모델 설계 능력<br>(Model Architecture)** | **Stage 2. Single Detector** | ResNet18 Backbone + Regression/Classification Head 분리 설계, 멀티태스크 Loss 밸런싱 |
| **실전 적용 능력<br>(Real-world Application)** | **Stage 3. YOLOv5** | Roboflow 포트홀 데이터셋 + YOLOv5 파인튜닝으로 실제 도로 영상 실시간 탐지 수행 |

---

## 1. 실험 환경 및 단계별 도전 과제 (3 Stages)
| 단계 | 주제 | 데이터셋 | 도전 과제 |
| :---: | :--- | :--- | :--- |
| **Stage 1 (🥉)** | **BBox 연산 & NMS 직접 구현** | 임의 이미지 (고양이/강아지) | xyxy / xywh / normalized cxywh / Polygon 4가지 포맷 구현, cv2 없이 순수 NumPy NMS 구현 |
| **Stage 2 (🥈)** | **Single Object Detector** | Image Localization (오이/가지/버섯 3 classes) | XML 파싱 기반 Custom Dataset, ResNet18 멀티태스크 탐지 모델, alpha·beta 가중 Loss 설계 |
| **Stage 3 (🥇)** | **YOLOv5 Pothole Detection** | Roboflow Pothole (665장) | YAML 기반 커스텀 학습, 실제 도로 영상 실시간 추론, mAP50 기반 성능 검증 |

---

## 2. 프로젝트 구조
    ├── src/
    │   └── train_yolo.py        # YOLOv5 전체 학습 파이프라인 (clone → download → train → val → test)
    ├── results/
    │   └── yolo_results.png     # YOLOv5 학습 결과 그래프 (Loss, mAP 곡선)
    ├── .gitignore
    ├── LICENSE
    ├── requirements.txt
    └── README.md

> **Note**: BBox 연산(IoU, NMS) 및 Single Object Detector 구현 코드는
> 선행 프로젝트 [object-detection-fundamentals](https://github.com/AD-Styles/object-detection-fundamentals)를 참고하세요.

---

## 3. 핵심 구현 상세 (Implementation Details)

### 🔹 Stage 1. BBox 연산 & NMS — 핵심 알고리즘 직접 구현

객체 탐지 파이프라인의 가장 근본이 되는 연산들을 외부 라이브러리 없이 직접 구현했습니다.

**BBox 포맷 4종 구현**
| 포맷 | 표현 방식 | 사용처 |
| :--- | :--- | :--- |
| `xyxy` | 좌상단(x1,y1) + 우하단(x2,y2) | IoU 계산 입력 |
| `xywh` | 좌상단(x,y) + 너비·높이(w,h) | COCO 스타일 |
| `cxywh normalized` | 중심점(cx,cy) + 비율(w,h) 0~1 | **YOLO 표준 포맷** |
| `Polygon` | N개 꼭짓점 좌표 | 불규칙 객체 경계 |

**IoU 계산** (`ops.py` → `calculate_iou`)
```
교집합 면적 / 합집합 면적
= intersection / (area1 + area2 - intersection + 1e-16)
```
- ZeroDivisionError 방지를 위한 **Epsilon(1e-16)** 처리
- Tensor / NumPy / List **범용 인터페이스** 설계

**NMS 구현** (`ops.py` → `nms`) — `cv2.dnn.NMSBoxes` 의존 제거
```
1. Confidence Score 기준 내림차순 정렬
2. 가장 높은 박스 선택 → keep 리스트에 추가
3. 나머지 박스와 IoU 계산 → threshold 이상이면 제거
4. 남은 박스로 반복
```

---

### 🔹 Stage 2. Single Object Detector — ResNet18 기반 모델 설계

**노트북(scratch CNN) 대비 `detector.py`의 개선 사항:**

| 항목 | 노트북 (scratch) | detector.py (개선) |
| :--- | :--- | :--- |
| Backbone | Conv2d × 3 직접 구현 | ResNet18 **ImageNet Pretrained** |
| BBox 출력 | raw 픽셀 좌표 | **Sigmoid 정규화 (0~1)** |
| 가중치 초기화 | 기본값 | **He Initialization** |
| 학습 안정화 | 없음 | **BatchNorm1d + Dropout** |
| Loss 구조 | bbox + class 단순 합산 | **alpha·beta 가중합** |

```
Input (B,3,224,224)
    → ResNet18 Backbone (ImageNet Pretrained)
    → Flatten → [B, 512]
    → Regression Head  → Sigmoid → [B, 4]  (cx, cy, w, h, 0~1)
    → Classification Head         → [B, num_classes]  (logits)

Total Loss = alpha × MSELoss(bbox) + beta × CrossEntropyLoss(class)
```

---

### 🔹 Stage 3. YOLOv5 Pothole Detection — 실전 파이프라인

**데이터셋**: Roboflow Public Pothole Dataset — 665장, Train/Val/Test = 7:2:1, 단일 클래스(pothole)

**학습 설정**:
- Model: YOLOv5s (커스텀 YAML — nc=1로 수정)
- Epochs: 100 / Batch: 32 / Image size: 640
- Weights: scratch (사전학습 없이 처음부터 학습)

**평가 지표**:
| 지표 | 설명 |
| :--- | :--- |
| `box_loss` | 바운딩 박스 위치 오차 |
| `obj_loss` | 객체 존재 여부 오차 |
| `cls_loss` | 클래스 분류 오차 |
| `mAP50` | IoU 50% 기준 평균 정밀도 |
| `mAP50-95` | IoU 50~95% 구간 평균 정밀도 |

---

## 4. 실험 결과 (Results)

### 최종 성능 요약

| Stage | 모델 | 최종 지표 | 비고 |
| :---: | :--- | :--- | :--- |
| **Stage 2** | Single Object Detector | Total Loss `0.024` (Epoch 100) | BBox MSE `0.004` / Class CE `0.02` |
| **Stage 3** | YOLOv5s Pothole | mAP50 기반 수렴 확인 | 실제 도로 영상 실시간 탐지 성공 |

---

### 📈 Stage 2. Single Object Detector — Train/Val Loss
| 학습 손실 곡선 (Train/Val Loss) |
| :---: |
| ![loss](results/detector_loss.png) |

- **엔지니어링 인사이트**: Epoch 1 기준 Total Loss 1.96에서 Epoch 100 기준 0.024까지 **98% 이상 감소**하며 안정적으로 수렴했습니다. BBox MSE Loss(0.004)와 Class CE Loss(0.02)가 서로 충돌하지 않고 함께 하강한 것은 `alpha·beta` 가중합 구조가 멀티태스크 학습의 균형을 효과적으로 잡았음을 의미합니다. He Initialization + BatchNorm1d 적용으로 학습 초기 수렴 속도가 크게 개선되었습니다.

---

### 📈 Stage 3. YOLOv5 Pothole Detection — 학습 결과
| YOLOv5 학습 결과 (results.png) |
| :---: |
| ![results](results/yolo_results.png) |

- **엔지니어링 인사이트**: YOLOv5의 3개 손실(box / obj / cls)이 모두 안정적으로 수렴하였으며, mAP50이 빠르게 상승하는 패턴을 확인했습니다. 특히 scratch 학습(사전학습 없음)임에도 665장의 소규모 데이터셋에서 포트홀을 성공적으로 탐지한 것은 YOLO 아키텍처의 강력한 특징 추출 능력을 증명합니다. 실제 도로 영상(mp4)에 대한 프레임별 실시간 추론에서도 포트홀이 일관되게 검출되었습니다.

---

## 5. 향후 과제 (Future Work)
현재 YOLOv5s 모델은 **scratch 학습 + 단일 클래스** 환경에서 검증되었습니다. 실무 적용을 위해 다음 방향으로 확장할 계획입니다.

- **Transfer Learning 적용**: COCO Pretrained 가중치로 파인튜닝하여 소규모 데이터셋에서의 성능 상한선 탐색
- **모델 경량화**: YOLOv5n(nano) 또는 YOLOv8로 전환하여 엣지 디바이스 실시간 추론 최적화
- **다중 도로 결함 탐지**: 포트홀 외 크랙, 노면 파손 등 다중 클래스로 확장
- **ONNX 변환**: TorchScript/ONNX Export를 통한 서비스 배포 파이프라인 구축

---

## 6. 💡 회고록 (Retrospective)
이번 프로젝트에서는 객체 탐지를 단순히 YOLOv5 명령어 한 줄로 끝내는 것이 아니라, 그 내부에서 작동하는 핵심 로직을 손으로 직접 짜보는 것에서 출발했습니다.

- **Stage 1 (BBox & Ops)**: IoU를 처음 구현했을 때 분모가 0이 되어 프로그램이 멈추는 순간을 경험했습니다. `1e-16` Epsilon 하나를 추가하는 과정에서 "왜 실무 코드에는 항상 이런 방어 로직이 있는가"를 처음으로 이해했습니다. NMS 역시 `cv2.dnn.NMSBoxes` 한 줄 뒤에 Confidence Score 기반 정렬 → 중복 제거 → 반복이라는 명확한 알고리즘이 있음을 직접 구현하며 체득했습니다.

- **Stage 2 (Single Detector)**: 하나의 신경망이 좌표(회귀)와 클래스(분류)를 동시에 맞추는 멀티태스크 학습의 핵심은 **두 손실의 균형**이었습니다. alpha, beta 가중치를 조절하며 BBox Loss와 Class Loss가 서로를 방해하지 않고 함께 줄어드는 최적의 균형점을 찾는 과정에서, 딥러닝 모델 설계가 단순한 레이어 쌓기가 아니라 목적 함수 설계임을 깨달았습니다.

- **Stage 3 (YOLOv5)**: Stage 1, 2를 통해 BBox 포맷 변환, IoU, NMS, 멀티태스크 Loss의 원리를 이미 알고 있는 상태에서 YOLOv5를 사용하니 YAML 설정 하나하나가 다르게 읽혔습니다. `nc: 1`, `anchors`, `mAP50-95`가 단순한 설정 값이 아니라 Stage 1~2에서 직접 구현했던 개념들의 실무적 추상화임을 이해할 수 있었습니다. 실제 도로 영상에서 포트홀이 실시간으로 검출되는 순간은, 3단계 학습 여정의 가장 강력한 동기부여가 되었습니다.

세 단계를 거치며 얻은 가장 큰 수확은 **"YOLO가 왜 빠르고 강력한가"를 원리 수준에서 설명할 수 있는 능력**입니다. 라이브러리를 쓰는 사람과 라이브러리 내부를 이해하고 쓰는 사람의 차이는, 문제가 발생했을 때 어디서 무엇을 봐야 하는지를 아는 것에서 나타납니다. 이번 프로젝트가 그 차이를 만들어준 결정적인 경험이었습니다.
