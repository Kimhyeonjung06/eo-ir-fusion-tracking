"""EO(가시광)/IR(열화상) 단일 모달리티 탐지기.

IR 전처리는 기존 프로젝트(realtime-object-detection-tracking-app)에서 확인한 결과를 그대로 쓴다.
COCO 사전학습 가중치는 가시광의 색 분포를 학습했으므로, 온도를 색으로 칠한 열화상은
학습 분포에서 크게 벗어난다. 팔레트를 걷어내 휘도만 남기면 탐지가 살아난다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_MODEL_CACHE: Dict[str, object] = {}

# KAIST는 보행자 데이터 → COCO 0번(person)만 사용
PERSON_CLASS = 0


def preprocess(frame: np.ndarray, mode: str) -> np.ndarray:
    """mode: none | gray | clahe"""
    if mode == "none":
        return frame
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "clahe":
        gray = _CLAHE.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_model(weights: str):
    if weights not in _MODEL_CACHE:
        from ultralytics import YOLO
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


@dataclass
class DetectorConfig:
    weights: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    preprocess: str = "none"
    classes: Optional[List[int]] = None
    device: str = "cpu"


def detect(frame: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """(N,5) = [x1,y1,x2,y2,conf]"""
    img = preprocess(frame, cfg.preprocess)
    model = load_model(cfg.weights)
    res = model.predict(
        img,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        classes=cfg.classes if cfg.classes is not None else [PERSON_CLASS],
        device=cfg.device,
        verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = res.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
    return np.hstack([xyxy, conf])
