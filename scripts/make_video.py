"""EO / IR / 융합 추적 영상 생성.

세 화면을 나란히 붙인다.
    왼쪽   EO 탐지
    가운데 IR 탐지
    오른쪽 융합 + 추적 ID + 최근 궤적

중간 구간에서 IR을 의도적으로 끊어(--cut) 추적이 유지되는지 눈으로 보이게 한다.
숫자 표로는 전달되지 않는 부분이다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import dataset as ds                                   # noqa: E402
from detect import DetectorConfig, detect, preprocess   # noqa: E402
from fusion import late_fuse                            # noqa: E402
from track import ByteLikeTracker                       # noqa: E402

# BGR
GT = (90, 200, 90)
EO_C = (180, 140, 90)      # 차가운 청회색
IR_C = (40, 110, 200)      # 따뜻한 주황
FUS_C = (140, 160, 40)     # 청록
LOST = (60, 60, 220)
BG = (24, 24, 24)
FG = (238, 238, 238)
DIM = (150, 150, 150)

PALETTE = [(96, 176, 96), (200, 150, 60), (80, 140, 220), (190, 110, 200),
           (70, 190, 200), (150, 150, 240), (110, 200, 150), (210, 170, 90)]


def band(img, text, color=FG, h=26):
    out = np.vstack([np.full((h, img.shape[1], 3), BG, np.uint8), img])
    cv2.putText(out, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def draw_boxes(img, boxes, color, thick=2):
    for b in boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thick)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kaist_full")
    ap.add_argument("--seq", default="set03/V000")
    ap.add_argument("--weights-eo", required=True)
    ap.add_argument("--weights-ir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--ir-preprocess", default="none", choices=["none", "gray", "clahe"])
    ap.add_argument("--cut", default="180:300", help="IR을 끊을 구간 start:end (비우면 끊지 않음)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="results/video/tracking.mp4")
    args = ap.parse_args()

    cut_a = cut_b = -1
    if args.cut:
        cut_a, cut_b = (int(v) for v in args.cut.split(":"))

    seqs = ds.discover(args.data)
    match = [s for s in seqs if s.name == args.seq]
    if not match:
        print("[!] 시퀀스를 찾지 못했습니다:", args.seq)
        return 1
    seq = match[0]
    print("시퀀스 %s (%s) 프레임 %d" % (seq.name, ds.time_of_day(seq.name), len(seq)))

    cfg_eo = DetectorConfig(weights=args.weights_eo, imgsz=args.imgsz, conf=args.conf,
                            preprocess="none", device=args.device)
    cfg_ir = DetectorConfig(weights=args.weights_ir, imgsz=args.imgsz, conf=args.conf,
                            preprocess=args.ir_preprocess, device=args.device)

    tracker = ByteLikeTracker(high_thresh=max(args.conf, 0.2), low_thresh=args.conf * 0.4,
                              max_age=30, min_hits=2)
    trails = defaultdict(lambda: deque(maxlen=28))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    names = seq.frames[args.start: args.start + args.frames]
    for i, fname in enumerate(names):
        eo = cv2.imread(str(seq.visible_dir / fname))
        ir_raw = cv2.imread(str(seq.lwir_dir / fname))
        if eo is None or ir_raw is None:
            continue
        gt, _ = ds.load_annotation(seq.ann_dir, Path(fname).stem)

        ir_lost = cut_a <= i < cut_b
        d_eo = detect(eo, cfg_eo)
        d_ir = np.zeros((0, 5), np.float32) if ir_lost else detect(ir_raw, cfg_ir)

        if ir_lost:
            fused = (np.hstack([d_eo, np.zeros((len(d_eo), 1), np.float32)])
                     if len(d_eo) else np.zeros((0, 6), np.float32))
        else:
            fused = late_fuse(d_eo, d_ir, iou_thresh=0.4)

        tracks = tracker.update(fused)

        # 1) EO
        p_eo = draw_boxes(eo.copy(), gt, GT, 1)
        draw_boxes(p_eo, d_eo[:, :4], EO_C)
        p_eo = band(p_eo, "EO  visible   det=%d" % len(d_eo), EO_C)

        # 2) IR
        ir_view = preprocess(ir_raw, "clahe" if args.ir_preprocess == "none" else args.ir_preprocess)
        p_ir = draw_boxes(ir_view.copy(), gt, GT, 1)
        if ir_lost:
            overlay = p_ir.copy()
            cv2.rectangle(overlay, (0, 0), (p_ir.shape[1], p_ir.shape[0]), (0, 0, 0), -1)
            p_ir = cv2.addWeighted(overlay, 0.55, p_ir, 0.45, 0)
            cv2.putText(p_ir, "SENSOR LOST", (p_ir.shape[1] // 2 - 110, p_ir.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, LOST, 2, cv2.LINE_AA)
            p_ir = band(p_ir, "IR  thermal   SIGNAL CUT", LOST)
        else:
            draw_boxes(p_ir, d_ir[:, :4], IR_C)
            p_ir = band(p_ir, "IR  thermal   det=%d" % len(d_ir), IR_C)

        # 3) 융합 + 추적
        p_fu = draw_boxes(eo.copy(), gt, GT, 1)
        for t in tracks:
            c = PALETTE[t.track_id % len(PALETTE)]
            x1, y1, x2, y2 = (int(v) for v in t.box)
            cv2.rectangle(p_fu, (x1, y1), (x2, y2), c, 2)
            cv2.putText(p_fu, "#%d" % t.track_id, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            trails[t.track_id].append((int((x1 + x2) / 2), y2))

        # 살아 있는 트랙의 궤적만 남긴다. 그러지 않으면 사라진 트랙의 선이 화면에 계속 쌓인다.
        alive = {t.track_id for t in tracker.tracks}
        for tid in [k for k in trails if k not in alive]:
            del trails[tid]
        for t in tracks:
            pts = trails[t.track_id]
            c = PALETTE[t.track_id % len(PALETTE)]
            for k in range(1, len(pts)):
                cv2.line(p_fu, pts[k - 1], pts[k], c, 1, cv2.LINE_AA)
        p_fu = band(p_fu, "FUSION + TRACK   tracks=%d" % len(tracks), FUS_C)

        grid = np.hstack([p_eo, p_ir, p_fu])

        # 하단 상태줄
        hud = np.full((30, grid.shape[1], 3), BG, np.uint8)
        state = "IR LOST" if ir_lost else "IR OK"
        cv2.putText(hud, "frame %04d / %04d    %s    GT %d    tracks %d"
                    % (i + 1, len(names), state, len(gt), len(tracks)),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    LOST if ir_lost else DIM, 1, cv2.LINE_AA)
        frame = np.vstack([grid, hud])

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
            print("출력 %dx%d @ %dfps" % (w, h, args.fps))
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(names)))

    if writer is not None:
        writer.release()
    print("저장:", out_path)

    # 브라우저에서 바로 재생되도록 H.264로 한 번 더 변환한다(ffmpeg가 있을 때만)
    h264 = out_path.with_name(out_path.stem + "_h264.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(h264)],
                       check=True)
        print("H.264:", h264)
    except Exception as e:
        print("ffmpeg 변환 생략:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
