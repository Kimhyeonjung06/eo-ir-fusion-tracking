"""KAIST -> YOLO 학습 형식 변환.

COCO 사전학습 가중치는 야간 보행자에서 사실상 작동하지 않는다(README 참고).
절대 성능을 올리려면 이 데이터로 미세조정해야 하고, 그러려면 YOLO 형식이 필요하다.

만드는 구조:
    <out>/images/train/<seq>_<frame>.jpg
    <out>/labels/train/<seq>_<frame>.txt      # class cx cy w h  (모두 0~1 정규화)
    <out>/images/val/...
    <out>/data.yaml

이미지는 복사하지 않고 하드링크를 건다(36GB를 두 벌 두지 않기 위해).
같은 볼륨이 아니면 복사로 넘어간다.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dataset as ds  # noqa: E402

# KAIST 표준 분할: set00~05 학습, set06~11 평가
TRAIN_SETS = {"set00", "set01", "set02", "set03", "set04", "set05"}
VAL_SETS = {"set06", "set07", "set08", "set09", "set10", "set11"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def to_yolo(box, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
            min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0))


def split_of(seq_name: str) -> str:
    s = seq_name.split("/")[0]
    if s in TRAIN_SETS:
        return "train"
    if s in VAL_SETS:
        return "val"
    return "train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kaist_full")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--modality", default="lwir", choices=["lwir", "visible"])
    ap.add_argument("--stride", type=int, default=2, help="연속 프레임은 거의 같으므로 솎아낸다")
    ap.add_argument("--empty-ratio", type=float, default=0.1,
                    help="사람이 없는 프레임을 배경으로 섞는 비율")
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--img-h", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out) / args.modality
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    seqs = ds.discover(args.data)
    if not seqs:
        print("[!] 시퀀스를 찾지 못했습니다:", args.data)
        return 1
    print("시퀀스 %d개" % len(seqs))

    counts = {"train": 0, "val": 0}
    empty = {"train": 0, "val": 0}
    boxes_total = 0

    for seq in seqs:
        sp = split_of(seq.name)
        src_dir = seq.lwir_dir if args.modality == "lwir" else seq.visible_dir
        tag = seq.name.replace("/", "_")
        for fname in seq.frames[::args.stride]:
            stem = Path(fname).stem
            gt, ign = ds.load_annotation(seq.ann_dir, stem)
            has = len(gt) > 0
            if not has and rng.random() > args.empty_ratio:
                continue

            img_src = src_dir / fname
            if not img_src.is_file():
                continue
            base = "%s_%s" % (tag, stem)
            link_or_copy(img_src, out / "images" / sp / (base + ".jpg"))

            lines = []
            for b in gt:
                cx, cy, bw, bh = to_yolo(b, args.img_w, args.img_h)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append("0 %.6f %.6f %.6f %.6f" % (cx, cy, bw, bh))
            (out / "labels" / sp / (base + ".txt")).write_text("\n".join(lines), encoding="utf-8")

            counts[sp] += 1
            boxes_total += len(lines)
            if not has:
                empty[sp] += 1

        print("  %-14s %-5s 누적 train=%d val=%d" % (seq.name, sp, counts["train"], counts["val"]))

    yaml = out / "data.yaml"
    yaml.write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnc: 1\nnames: [person]\n"
        % out.resolve().as_posix(),
        encoding="utf-8")

    print("\n학습 %d장(배경 %d) / 평가 %d장(배경 %d), 박스 %d개"
          % (counts["train"], empty["train"], counts["val"], empty["val"], boxes_total))
    print("data.yaml:", yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
