"""KAIST Multispectral Pedestrian 로더.

프리뷰 배포본 구조:
    kaist-cvpr15-preview/images/setXX/VYYY/visible/I00000.jpg
    kaist-cvpr15-preview/images/setXX/VYYY/lwir/I00000.jpg
    kaist-cvpr15-preview/annotations-xml-new-sanitized/setXX/VYYY/I00000.xml

배포본마다 상위 폴더 이름이 달라 visible/lwir 쌍을 재귀 탐색으로 찾는다.
주석은 XML(sanitized)과 Caltech bbGt 텍스트 두 형식을 모두 읽는다.
macOS 메타파일(._*)은 제외한다.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

POSITIVE_LABELS = {"person"}
IGNORE_LABELS = {"people", "person?", "person?a", "cyclist", "persons"}

_EMPTY = np.zeros((0, 4), np.float32)


def _clean(names) -> List[str]:
    return [n for n in names if not n.startswith("._")]


@dataclass
class Sequence:
    name: str                 # 예: set04/V001
    visible_dir: Path
    lwir_dir: Path
    ann_dir: Optional[Path]
    frames: List[str]

    def __len__(self) -> int:
        return len(self.frames)


def _find_ann_dir(seq_dir: Path, root: Path) -> Optional[Path]:
    """images/setXX/VYYY -> annotations*/setXX/VYYY"""
    rel = seq_dir.parts[-2:]
    for cand in root.rglob("*"):
        if not cand.is_dir() or cand.parts[-2:] != rel:
            continue
        low = str(cand).lower()
        if "annotation" in low and "image" not in cand.parts[-3].lower():
            return cand
    return None


def discover(root: str | Path) -> List[Sequence]:
    root = Path(root)
    seqs: List[Sequence] = []
    for vis in sorted(root.rglob("visible")):
        if not vis.is_dir():
            continue
        lwir = vis.parent / "lwir"
        if not lwir.is_dir():
            continue
        frames = sorted(_clean(p.name for p in vis.iterdir()
                               if p.suffix.lower() in {".jpg", ".png"}))
        if not frames:
            continue
        seq_dir = vis.parent
        name = "/".join(seq_dir.parts[-2:])
        seqs.append(Sequence(name, vis, lwir, _find_ann_dir(seq_dir, root), frames))
    return seqs


def _box_from_xml(obj: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    bb = obj.find("bndbox")
    if bb is None:
        return None

    def g(tag: str) -> Optional[float]:
        e = bb.find(tag)
        if e is None or e.text is None:
            return None
        try:
            return float(e.text)
        except ValueError:
            return None

    xmin, ymin, xmax, ymax = g("xmin"), g("ymin"), g("xmax"), g("ymax")
    if None not in (xmin, ymin, xmax, ymax):
        return xmin, ymin, xmax, ymax
    x, y, w, h = g("x"), g("y"), g("w"), g("h")
    if None not in (x, y, w, h):
        return x, y, x + w, y + h
    return None


def parse_xml(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pos, ign = [], []
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return _EMPTY.copy(), _EMPTY.copy()
    for obj in root.iter("object"):
        name_el = obj.find("name")
        lbl = (name_el.text or "").strip().lower() if name_el is not None else ""
        box = _box_from_xml(obj)
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            continue
        (pos if lbl in POSITIVE_LABELS else ign if lbl in IGNORE_LABELS else ign).append(list(box))
    return (np.asarray(pos, np.float32).reshape(-1, 4),
            np.asarray(ign, np.float32).reshape(-1, 4))


def parse_bbgt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pos, ign = [], []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        lbl = parts[0].lower()
        try:
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        box = [x, y, x + w, y + h]
        (pos if lbl in POSITIVE_LABELS else ign).append(box)
    return (np.asarray(pos, np.float32).reshape(-1, 4),
            np.asarray(ign, np.float32).reshape(-1, 4))


def load_annotation(ann_dir: Optional[Path], stem: str) -> Tuple[np.ndarray, np.ndarray]:
    if ann_dir is None:
        return _EMPTY.copy(), _EMPTY.copy()
    xml = ann_dir / (stem + ".xml")
    if xml.is_file():
        return parse_xml(xml)
    txt = ann_dir / (stem + ".txt")
    if txt.is_file():
        return parse_bbgt(txt)
    return _EMPTY.copy(), _EMPTY.copy()


def iterate(seq: Sequence, stride: int = 1, limit: Optional[int] = None) -> Iterator[Dict]:
    names = seq.frames[::stride]
    if limit:
        names = names[:limit]
    for i, fname in enumerate(names):
        eo = cv2.imread(str(seq.visible_dir / fname))
        ir = cv2.imread(str(seq.lwir_dir / fname))
        if eo is None or ir is None:
            continue
        gt, ignore = load_annotation(seq.ann_dir, Path(fname).stem)
        yield {"idx": i, "name": fname, "eo": eo, "ir": ir, "gt": gt, "ignore": ignore}


# KAIST 공식 구성 — set00~02 주간(캠퍼스/도로/시내), set03~05 야간
DAY_SETS = {"set00", "set01", "set02", "set06", "set07", "set08"}
NIGHT_SETS = {"set03", "set04", "set05", "set09", "set10", "set11"}


def time_of_day(seq_name: str) -> str:
    s = seq_name.split("/")[0]
    if s in DAY_SETS:
        return "day"
    if s in NIGHT_SETS:
        return "night"
    return "unknown"
