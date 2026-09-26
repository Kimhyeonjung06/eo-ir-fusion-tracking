"""KAIST로 YOLO 미세조정.

COCO 사전학습 가중치는 가시광 주간 영상의 분포를 학습했다.
야간 열화상은 그 분포에서 크게 벗어나며, 그대로 쓰면 신뢰도 0.15에서 탐지가 0이다.
모달리티별로 따로 미세조정해 그 격차를 줄인다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml 경로")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/finetune")
    ap.add_argument("--name", required=True, help="예: lwir, visible")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0,
                    help="Windows에서 워커를 늘리면 pinned memory 스레드가 "
                         "CUDA error: resource already mapped 로 죽는다. 기본 0")
    args = ap.parse_args()

    from ultralytics import YOLO

    # project가 상대경로면 ultralytics의 runs_dir 아래로 한 번 더 들어가 경로가 중첩된다
    project = Path(args.project).resolve()

    print("미세조정 시작: %s  (data=%s, imgsz=%d, batch=%d, device=%s, workers=%d)"
          % (args.name, args.data, args.imgsz, args.batch, args.device, args.workers))
    t0 = time.perf_counter()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,
        # 열화상은 색 정보가 없으므로 색 증강은 의미가 없고 오히려 해롭다
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        # 보행자는 좌우 대칭이 성립하지만 상하 반전은 성립하지 않는다
        fliplr=0.5, flipud=0.0,
        mosaic=1.0,
        verbose=True,
    )

    best = project / args.name / "weights" / "best.pt"
    print("\n완료 %.1f분 -> %s" % ((time.perf_counter() - t0) / 60.0, best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
