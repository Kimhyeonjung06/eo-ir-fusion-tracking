"""조건별 실험 실행.

설계 요지 — 탐지는 비싸고 조건은 많다.
  1단계: 시퀀스마다 EO/IR 탐지를 한 번만 계산해 캐시에 저장한다.
  2단계: 결손·지연처럼 결과 수준의 열화는 캐시 위에서 적용해 조건을 늘린다.
  잡음·흐림처럼 이미지를 바꾸는 열화만 재탐지가 필요하다(별도 실행).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dataset as ds                                   # noqa: E402
from degrade import DegradeConfig, SensorDegrader      # noqa: E402
from detect import DetectorConfig, detect              # noqa: E402
from fusion import late_fuse                           # noqa: E402
from metrics import detection_scores, track_stability  # noqa: E402
from track import ByteLikeTracker                      # noqa: E402


@dataclass
class Condition:
    name: str
    use_eo: bool = True
    use_ir: bool = True
    eo_degrade: Optional[Dict] = None
    ir_degrade: Optional[Dict] = None


def default_conditions() -> List[Condition]:
    conds = [
        Condition("EO_only", use_ir=False),
        Condition("IR_only", use_eo=False),
        Condition("Fusion"),
    ]
    for r in (0.1, 0.3, 0.5, 0.7):
        pct = int(r * 100)
        conds.append(Condition("Fusion_IRdrop%d" % pct, ir_degrade={"dropout": r}))
        conds.append(Condition("Fusion_EOdrop%d" % pct, eo_degrade={"dropout": r}))
    for d in (1, 2, 5):
        conds.append(Condition("Fusion_IRdelay%d" % d, ir_degrade={"delay_frames": d}))
    return conds


def build_cache(seq, cfg_eo, cfg_ir, stride, limit, cache_path: Path,
                ir_noise: float = 0.0, ir_blur: int = 0) -> Dict:
    if cache_path.is_file():
        blob = np.load(cache_path, allow_pickle=True)
        return {k: blob[k] for k in blob.files}

    # 잡음·흐림은 이미지 자체를 바꾸므로 결과 수준에서 흉내 낼 수 없고 재탐지가 필요하다.
    # 그래서 캐시 단계에서 적용하고, 캐시 파일 이름으로 조건을 구분한다.
    deg = None
    if ir_noise > 0 or ir_blur >= 3:
        deg = SensorDegrader(DegradeConfig(noise_sigma=ir_noise, blur_ksize=ir_blur))

    eo_dets, ir_dets, gts, lat_eo, lat_ir = [], [], [], [], []
    for item in ds.iterate(seq, stride=stride, limit=limit):
        ir_img = item["ir"]
        if deg is not None:
            ir_img = deg(ir_img)
            if ir_img is None:
                ir_img = item["ir"]
        t0 = time.perf_counter()
        d_eo = detect(item["eo"], cfg_eo)
        t1 = time.perf_counter()
        d_ir = detect(ir_img, cfg_ir)
        t2 = time.perf_counter()
        eo_dets.append(d_eo)
        ir_dets.append(d_ir)
        gts.append(item["gt"])
        lat_eo.append((t1 - t0) * 1000.0)
        lat_ir.append((t2 - t1) * 1000.0)

    cache = {
        "eo": np.array(eo_dets, dtype=object),
        "ir": np.array(ir_dets, dtype=object),
        "gt": np.array(gts, dtype=object),
        "lat_eo": np.array(lat_eo),
        "lat_ir": np.array(lat_ir),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    return cache


def apply_result_degrade(dets: List[np.ndarray], spec: Optional[Dict]):
    """결과 수준 열화(결손/지연)를 탐지 결과 열에 적용한다. None = 해당 프레임 관측 없음."""
    if not spec:
        return list(dets)
    cfg = DegradeConfig(**spec)
    rng = np.random.default_rng(cfg.seed)
    out: List[Optional[np.ndarray]] = []
    buf: List[np.ndarray] = []
    for d in dets:
        buf.append(d)
        if cfg.delay_frames:
            if len(buf) <= cfg.delay_frames:
                out.append(None)
                continue
            cur = buf[-1 - cfg.delay_frames]
        else:
            cur = d
        if cfg.dropout > 0 and rng.random() < cfg.dropout:
            out.append(None)
            continue
        out.append(cur)
    return out


def as_xyxyc(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return np.zeros((0, 5), np.float32)
    return a.reshape(-1, a.shape[-1])[:, :5]


def run_condition(cache: Dict, cond: Condition, fuse_iou: float, w_rgb: float,
                  trk_high: float = 0.5, trk_low: float = 0.1,
                  op_conf: float = 0.10) -> Dict:
    eo = [as_xyxyc(x) for x in cache["eo"]]
    ir = [as_xyxyc(x) for x in cache["ir"]]
    gt = list(cache["gt"])

    eo_s = apply_result_degrade(eo, cond.eo_degrade) if cond.use_eo else [None] * len(eo)
    ir_s = apply_result_degrade(ir, cond.ir_degrade) if cond.use_ir else [None] * len(ir)

    # 추적기의 고신뢰 기준은 탐지 임계값과 함께 움직여야 한다.
    # 탐지 conf를 낮춰 놓고 추적기 기준을 그대로 두면 새 트랙이 하나도 생기지 않는다.
    tracker = ByteLikeTracker(high_thresh=trk_high, low_thresh=trk_low)
    preds, log = [], []
    n_eo_lost = n_ir_lost = 0

    for i in range(len(gt)):
        a, b = eo_s[i], ir_s[i]
        if a is None:
            n_eo_lost += 1
        if b is None:
            n_ir_lost += 1

        if a is not None and b is not None:
            fused = late_fuse(a, b, iou_thresh=fuse_iou, w_rgb=w_rgb)
        elif a is not None:
            fused = (np.hstack([a, np.zeros((len(a), 1), np.float32)])
                     if len(a) else np.zeros((0, 6), np.float32))
        elif b is not None:
            fused = (np.hstack([b, np.ones((len(b), 1), np.float32)])
                     if len(b) else np.zeros((0, 6), np.float32))
        else:
            fused = np.zeros((0, 6), np.float32)

        preds.append(fused[:, :5] if len(fused) else np.zeros((0, 5), np.float32))
        tracks = tracker.update(fused)
        log.append([(t.track_id, t.box) for t in tracks])

    det = detection_scores(gt, preds)
    # AP 곡선을 그리려면 낮은 신뢰도까지 남겨야 하지만, 그 설정의 정밀도는 운용 값이 아니다.
    # 실제로 쓸 임계값에서의 정밀도·재현율을 따로 보고한다.
    op = [p[p[:, 4] >= op_conf] if len(p) else p for p in preds]
    det_op = detection_scores(gt, op)
    trk = track_stability(log)
    lat = np.asarray(cache["lat_eo"], dtype=float) + np.asarray(cache["lat_ir"], dtype=float)

    row = {"condition": cond.name}
    for k, v in det.items():
        row[k] = round(v, 4) if isinstance(v, float) else v
    for k in ("precision", "recall", "f1"):
        row["op_" + k] = round(det_op[k], 4)
    row["op_conf"] = op_conf
    for k, v in trk.items():
        row[k] = round(v, 3) if isinstance(v, float) else v
    row["eo_lost_frames"] = n_eo_lost
    row["ir_lost_frames"] = n_ir_lost
    row["lat_p50_ms"] = round(float(np.percentile(lat, 50)), 2)
    row["lat_p95_ms"] = round(float(np.percentile(lat, 95)), 2)
    row["lat_max_ms"] = round(float(np.max(lat)), 2)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kaist_preview")
    ap.add_argument("--limit", type=int, default=300, help="시퀀스당 사용할 프레임 수")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=4)
    ap.add_argument("--sets", default=None,
                    help="쓸 세트를 콤마로 지정 (예: set00,set04). 생략하면 전체")
    ap.add_argument("--balanced", action="store_true",
                    help="주간·야간에서 번갈아 뽑는다. 앞에서부터 자르면 한쪽만 들어간다")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--ir-preprocess", default="clahe", choices=["none", "gray", "clahe"])
    ap.add_argument("--fuse-iou", type=float, default=0.4)
    ap.add_argument("--w-rgb", type=float, default=0.5)
    ap.add_argument("--trk-high", type=float, default=None,
                    help="추적 고신뢰 기준. 생략하면 탐지 conf에서 자동 산정")
    ap.add_argument("--trk-low", type=float, default=None)
    ap.add_argument("--op-conf", type=float, default=0.10,
                    help="운용 임계값. AP는 낮은 conf까지 쓰지만 정밀도는 이 값에서 본다")
    ap.add_argument("--ir-noise", type=float, default=0.0,
                    help="IR에 가우시안 잡음 주입(표준편차 0~255). 재탐지가 필요하다")
    ap.add_argument("--ir-blur", type=int, default=0,
                    help="IR에 가우시안 흐림 커널(홀수). 재탐지가 필요하다")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # 탐지 임계값을 낮추면 추적 기준도 함께 내려야 한다.
    # 그러지 않으면 고신뢰 탐지가 하나도 없어 새 트랙이 생성되지 않는다.
    trk_high = args.trk_high if args.trk_high is not None else min(max(args.conf * 3, 0.1), 0.5)
    trk_low = args.trk_low if args.trk_low is not None else max(args.conf, 0.01)
    print("추적 기준: high=%.3f low=%.3f (탐지 conf=%.3f)" % (trk_high, trk_low, args.conf))

    seqs = ds.discover(args.data)
    if not seqs:
        print("[!] visible/lwir 쌍을 찾지 못했습니다: %s" % args.data)
        return 1
    if args.sets:
        want = {s.strip() for s in args.sets.split(",")}
        seqs = [s for s in seqs if s.name.split("/")[0] in want]

    if args.balanced:
        # 앞에서부터 자르면 set00~ 만 들어가 주간만 남는다. 주·야를 번갈아 뽑는다.
        day = [s for s in seqs if ds.time_of_day(s.name) == "day"]
        night = [s for s in seqs if ds.time_of_day(s.name) == "night"]
        mixed = []
        while (day or night) and len(mixed) < args.max_seq:
            if day:
                mixed.append(day.pop(0))
            if night and len(mixed) < args.max_seq:
                mixed.append(night.pop(0))
        seqs = mixed
    else:
        seqs = seqs[: args.max_seq]

    if not seqs:
        print("[!] 조건에 맞는 시퀀스가 없습니다")
        return 1
    print("시퀀스 %d개: %s" % (len(seqs),
          ", ".join("%s[%s](%d)" % (s.name, ds.time_of_day(s.name), len(s)) for s in seqs)))

    cfg_eo = DetectorConfig(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                            preprocess="none", device=args.device)
    cfg_ir = DetectorConfig(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                            preprocess=args.ir_preprocess, device=args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for seq in seqs:
        key = seq.name.replace("/", "_")
        t0 = time.perf_counter()
        suffix = ""
        if args.ir_noise > 0:
            suffix += "_n%g" % args.ir_noise
        if args.ir_blur >= 3:
            suffix += "_b%d" % args.ir_blur
        cache = build_cache(seq, cfg_eo, cfg_ir, args.stride, args.limit,
                            out_dir / "cache" / (key + suffix + ".npz"),
                            args.ir_noise, args.ir_blur)
        print("  [%s] 프레임 %d개, 캐시 %.1fs" % (seq.name, len(cache["gt"]), time.perf_counter() - t0))
        for cond in default_conditions():
            r = run_condition(cache, cond, args.fuse_iou, args.w_rgb,
                              trk_high, trk_low, args.op_conf)
            r["sequence"] = seq.name
            r["time_of_day"] = ds.time_of_day(seq.name)
            rows.append(r)
            print("    %-22s AP50=%.3f R=%.3f tracks=%d frag/track=%.2f"
                  % (r["condition"], r["ap50"], r["recall"], r["n_tracks"], r["frag_per_track"]))

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = out_dir / "results.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장: %s" % csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
