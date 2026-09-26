"""정성 비교 그림 생성.

정답이 있는 프레임을 골라 EO / IR / 융합의 탐지 결과를 나란히 보여준다.
숫자 표만으로는 야간에 무엇이 보이고 무엇이 안 보이는지 전달되지 않는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dataset as ds                       # noqa: E402
from detect import DetectorConfig, detect, preprocess  # noqa: E402
from fusion import late_fuse               # noqa: E402

GT_COLOR = (80, 220, 80)
DET_COLOR = (40, 140, 255)


def draw(img: np.ndarray, boxes, color, label: str = "", thick: int = 2) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        p1 = (int(b[0]), int(b[1]))
        p2 = (int(b[2]), int(b[3]))
        cv2.rectangle(out, p1, p2, color, thick)
    if label:
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1,
                    cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kaist_preview")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="results/night")
    ap.add_argument("--seq", default=None,
                    help="시퀀스 이름 (예: set03/V000). 생략하면 정답이 가장 많은 시퀀스")
    ap.add_argument("--name", default="demo_eo_ir_fusion.jpg")
    args = ap.parse_args()

    seqs = ds.discover(args.data)
    if args.seq:
        match = [s for s in seqs if s.name == args.seq]
        if not match:
            print("[!] 시퀀스를 찾지 못했습니다:", args.seq)
            return 1
        seq = match[0]
    else:
        # 첫 시퀀스는 사람이 한 명도 없을 수 있다(set00/V000). 정답이 많은 쪽을 고른다.
        def density(s):
            probe = s.frames[:150]
            return sum(len(ds.load_annotation(s.ann_dir, Path(f).stem)[0]) for f in probe)
        seq = max(seqs, key=density)
    print("시퀀스:", seq.name, "(%s)" % ds.time_of_day(seq.name))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg_eo = DetectorConfig(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                            preprocess="none")
    cfg_ir = DetectorConfig(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                            preprocess="clahe")

    picked, panels = 0, []
    for fname in seq.frames:
        gt, _ = ds.load_annotation(seq.ann_dir, Path(fname).stem)
        if len(gt) < 2:
            continue
        eo = cv2.imread(str(seq.visible_dir / fname))
        ir = cv2.imread(str(seq.lwir_dir / fname))
        if eo is None or ir is None:
            continue

        d_eo = detect(eo, cfg_eo)
        d_ir = detect(ir, cfg_ir)
        if len(d_ir) == 0:
            continue
        fused = late_fuse(d_eo, d_ir, iou_thresh=0.4)

        a = draw(draw(eo, gt, GT_COLOR, thick=1), d_eo, DET_COLOR,
                 "EO  det=%d" % len(d_eo))
        b = draw(draw(preprocess(ir, "clahe"), gt, GT_COLOR, thick=1), d_ir, DET_COLOR,
                 "IR (CLAHE)  det=%d" % len(d_ir))
        c = draw(draw(eo, gt, GT_COLOR, thick=1), fused[:, :4], DET_COLOR,
                 "Fusion  det=%d" % len(fused))
        panels.append(np.hstack([a, b, c]))
        picked += 1
        if picked >= args.n:
            break

    if not panels:
        print("[!] 조건에 맞는 프레임을 찾지 못했습니다")
        return 1

    grid = np.vstack(panels)
    path = out / args.name
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("저장:", path, grid.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
