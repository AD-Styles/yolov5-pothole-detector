"""
YOLOv5 Pothole Detector — Training Pipeline
=============================================
실행 순서:
  1. python train_yolo.py --mode clone     # YOLOv5 클론 및 설치
  2. python train_yolo.py --mode demo      # 사전학습 모델 데모 확인
  3. python train_yolo.py --mode download  # 포트홀 데이터셋 다운로드
  4. python train_yolo.py --mode train     # 모델 학습
  5. python train_yolo.py --mode val       # 모델 검증
  6. python train_yolo.py --mode test      # 모델 테스트 (실제 도로 영상)
  7. python train_yolo.py --mode results   # results/ 폴더에 그래프 저장

또는 전체 파이프라인 한 번에 실행:
  python train_yolo.py --mode all
"""

import os
import sys
import shutil
import argparse
import subprocess
import yaml

# ─────────────────────────────────────────────
# 0. 경로 설정
# ─────────────────────────────────────────────
BASE_DIR      = "/kaggle/working"
YOLO_DIR      = os.path.join(BASE_DIR, "yolov5")
POTHOLE_DIR   = os.path.join(YOLO_DIR, "pothole")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
DATA_YAML     = os.path.join(POTHOLE_DIR, "data.yaml")
MODEL_YAML    = os.path.join(YOLO_DIR, "models", "custom_yolo5s.yaml")
BEST_WEIGHTS  = os.path.join(YOLO_DIR, "runs", "train", "pothole_results", "weights", "best.pt")
DEMO_VIDEO    = "https://assets.mixkit.co/videos/25208/25208-720.mp4"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. YOLOv5 클론 및 설치
# ─────────────────────────────────────────────
def clone_and_install():
    """
    notebook 3:
        !git clone https://github.com/ultralytics/yolov5
        %cd yolov5
        %pip install -qr requirements.txt
    """
    print("\n" + "="*60)
    print("Step 1 | YOLOv5 Clone & Install")
    print("="*60)

    if not os.path.exists(YOLO_DIR):
        print("  YOLOv5 클론 중...")
        subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5", YOLO_DIR],
            check=True
        )
    else:
        print("  이미 클론되어 있습니다. 스킵합니다.")

    print("  requirements.txt 설치 중...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-qr",
         os.path.join(YOLO_DIR, "requirements.txt")],
        check=True
    )
    print("  ✅ 설치 완료")


# ─────────────────────────────────────────────
# 2. 사전학습 모델 데모
# ─────────────────────────────────────────────
def run_demo():
    """
    notebook 3:
        !python ./detect.py --weights ./yolov5s.pt
                            --img 640
                            --source ./data/images/
                            --exist-ok
    """
    print("\n" + "="*60)
    print("Step 2 | Pretrained YOLOv5s Demo")
    print("="*60)

    subprocess.run(
        [sys.executable, os.path.join(YOLO_DIR, "detect.py"),
         "--weights", os.path.join(YOLO_DIR, "yolov5s.pt"),
         "--img", "640",
         "--source", os.path.join(YOLO_DIR, "data", "images"),
         "--exist-ok"],
        cwd=YOLO_DIR,
        check=True
    )
    print("  ✅ 데모 완료 — runs/detect/exp/ 확인")


# ─────────────────────────────────────────────
# 3. 포트홀 데이터셋 다운로드
# ─────────────────────────────────────────────
def download_dataset():
    """
    notebook 3:
        %mkdir /kaggle/working/yolov5/pothole
        !curl -L "https://public.roboflow.com/ds/..." > roboflow.zip
        unzip roboflow.zip; rm roboflow.zip
    """
    print("\n" + "="*60)
    print("Step 3 | Pothole Dataset Download (Roboflow)")
    print("="*60)

    os.makedirs(POTHOLE_DIR, exist_ok=True)
    zip_path = os.path.join(POTHOLE_DIR, "roboflow.zip")

    print("  데이터셋 다운로드 중...")
    subprocess.run(
        ["curl", "-L",
         "https://public.roboflow.com/ds/554OsOsfOv?key=iVs10sN1Ht",
         "-o", zip_path],
        check=True
    )

    print("  압축 해제 중...")
    subprocess.run(
        ["unzip", "-q", zip_path, "-d", POTHOLE_DIR],
        check=True
    )
    os.remove(zip_path)

    # 이미지 경로 txt 파일 저장
    from glob import glob
    train_list = glob(os.path.join(POTHOLE_DIR, "train", "images", "*.jpg"))
    valid_list = glob(os.path.join(POTHOLE_DIR, "valid", "images", "*.jpg"))
    test_list  = glob(os.path.join(POTHOLE_DIR, "test",  "images", "*.jpg"))

    for split, paths in [("train", train_list), ("valid", valid_list), ("test", test_list)]:
        txt_path = os.path.join(POTHOLE_DIR, f"{split}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(paths) + "\n")

    print(f"  Train: {len(train_list)}장 | Valid: {len(valid_list)}장 | Test: {len(test_list)}장")
    print("  ✅ 데이터셋 준비 완료")


# ─────────────────────────────────────────────
# 4. YAML 설정
# ─────────────────────────────────────────────
def setup_yaml():
    """
    notebook 3:
        %%writetemplate /kaggle/working/yolov5/pothole/data.yaml
            train: /kaggle/working/yolov5/pothole/train/images
            ...
            nc: 1
            names: ['pothole']

        %%writetemplate /kaggle/working/yolov5/models/custom_yolo5s.yaml
            nc: 1
            ...
    """
    print("\n" + "="*60)
    print("Step 4 | YAML 설정")
    print("="*60)

    # data.yaml
    data_yaml_content = {
        "train": os.path.join(POTHOLE_DIR, "train", "images"),
        "val":   os.path.join(POTHOLE_DIR, "valid", "images"),
        "test":  os.path.join(POTHOLE_DIR, "test",  "images"),
        "nc":    1,
        "names": ["pothole"]
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    print(f"  data.yaml 저장 완료: {DATA_YAML}")

    # custom_yolo5s.yaml
    custom_yaml_content = """\
# 파라미터
nc: 1          # 클래스 수 (pothole만 탐지)
depth_multiple: 0.33
width_multiple: 0.50
anchors:
  - [10,13, 16,30, 33,23]
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]

# YOLOv5 v6.0 backbone
backbone:
  [[-1, 1, Conv, [64, 6, 2, 2]],
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],
   [-1, 3, C3, [512, False]],
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],
   [-1, 3, C3, [256, False]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],
   [-1, 3, C3, [512, False]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],
   [-1, 3, C3, [1024, False]],
   [[17, 20, 23], 1, Detect, [nc, anchors]],
  ]
"""
    with open(MODEL_YAML, "w") as f:
        f.write(custom_yaml_content)
    print(f"  custom_yolo5s.yaml 저장 완료: {MODEL_YAML}")
    print("  ✅ YAML 설정 완료")


# ─────────────────────────────────────────────
# 5. 모델 학습
# ─────────────────────────────────────────────
def train():
    """
    notebook 3:
        !python train.py --img 640 --batch 32 --epochs 100
                         --data ./pothole/data.yaml
                         --cfg ./models/custom_yolo5s.yaml
                         --weights ''
                         --name pothole_results
                         --cache --exist-ok
    """
    print("\n" + "="*60)
    print("Step 5 | YOLOv5 모델 학습")
    print("="*60)

    subprocess.run(
        [sys.executable, os.path.join(YOLO_DIR, "train.py"),
         "--img",     "640",
         "--batch",   "32",
         "--epochs",  "100",
         "--data",    DATA_YAML,
         "--cfg",     MODEL_YAML,
         "--weights", "",
         "--name",    "pothole_results",
         "--cache",
         "--exist-ok"],
        cwd=YOLO_DIR,
        check=True
    )
    print("  ✅ 학습 완료 — runs/train/pothole_results/ 확인")


# ─────────────────────────────────────────────
# 6. 모델 검증
# ─────────────────────────────────────────────
def validate():
    """
    notebook 3:
        !python val.py --weights runs/train/pothole_results/weights/best.pt
                       --data ./pothole/data.yaml
                       --img 640 --iou 0.75 --exist-ok
    """
    print("\n" + "="*60)
    print("Step 6 | 모델 검증 (Validation)")
    print("="*60)

    subprocess.run(
        [sys.executable, os.path.join(YOLO_DIR, "val.py"),
         "--weights", BEST_WEIGHTS,
         "--data",    DATA_YAML,
         "--img",     "640",
         "--iou",     "0.75",
         "--exist-ok"],
        cwd=YOLO_DIR,
        check=True
    )
    print("  ✅ 검증 완료")


# ─────────────────────────────────────────────
# 7. 실제 도로 영상 테스트
# ─────────────────────────────────────────────
def test():
    """
    notebook 3:
        !python detect.py
                --weights ./runs/train/pothole_results/weights/best.pt
                --source https://assets.mixkit.co/videos/25208/25208-720.mp4
                --exist-ok
    """
    print("\n" + "="*60)
    print("Step 7 | 실제 도로 영상 테스트")
    print("="*60)

    subprocess.run(
        [sys.executable, os.path.join(YOLO_DIR, "detect.py"),
         "--weights", BEST_WEIGHTS,
         "--source",  DEMO_VIDEO,
         "--exist-ok"],
        cwd=YOLO_DIR,
        check=True
    )
    print("  ✅ 테스트 완료 — runs/detect/exp/ 에서 결과 영상 확인")


# ─────────────────────────────────────────────
# 8. results/ 폴더에 그래프 저장
# ─────────────────────────────────────────────
def save_results():
    """
    YOLOv5 학습 완료 후 생성되는 results.png를
    results/yolo_results.png 로 복사.
    """
    print("\n" + "="*60)
    print("Step 8 | results/ 폴더에 그래프 저장")
    print("="*60)

    src = os.path.join(
        YOLO_DIR, "runs", "train", "pothole_results", "results.png"
    )
    dst = os.path.join(RESULTS_DIR, "yolo_results.png")

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  → saved: {dst}")
    else:
        print(f"  ⚠️  results.png 없음 — 학습(--mode train)을 먼저 실행하세요.")

    print("\n✅ 완료! /kaggle/working/results/ 에서 다운로드하세요.")
    print("   - yolo_results.png")


# ─────────────────────────────────────────────
# 9. 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Pothole Detector Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["clone", "demo", "download", "train", "val", "test", "results", "all"],
        help="실행할 단계 선택 (default: all)"
    )
    args = parser.parse_args()

    mode_map = {
        "clone":    clone_and_install,
        "demo":     run_demo,
        "download": download_dataset,
        "train":    train,
        "val":      validate,
        "test":     test,
        "results":  save_results,
    }

    if args.mode == "all":
        clone_and_install()
        download_dataset()
        setup_yaml()
        train()
        validate()
        test()
        save_results()
    else:
        # download 이후에는 yaml 설정 필요
        if args.mode == "download":
            download_dataset()
            setup_yaml()
        else:
            mode_map[args.mode]()


if __name__ == "__main__":
    import sys
    if any("ipykernel" in arg or "jupyter" in arg for arg in sys.argv):
        # 캐글/주피터 환경 → argparse 우회, 직접 mode 지정
        MODE = "all"   # ← 원하는 mode로 변경하세요
        
        mode_map = {
            "clone":    clone_and_install,
            "demo":     run_demo,
            "download": download_dataset,
            "train":    train,
            "val":      validate,
            "test":     test,
            "results":  save_results,
        }
        
        if MODE == "all":
            clone_and_install()
            download_dataset()
            setup_yaml()
            train()
            validate()
            test()
            save_results()
        elif MODE == "download":
            download_dataset()
            setup_yaml()
        else:
            mode_map[MODE]()
    else:
        # 로컬 환경 → argparse 정상 작동
        main()
