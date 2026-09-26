"""센서 열화 주입.

한 모달리티의 품질이 떨어질 때 융합 시스템이 얼마나 버티는지를 재기 위한 장치다.
프레임 단위로 결정론적(시드 고정)으로 동작해야 조건 간 비교가 성립한다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class DegradeConfig:
    dropout: float = 0.0        # 프레임 결측 비율 0~1
    delay_frames: int = 0       # 지연 프레임 수
    noise_sigma: float = 0.0    # 가우시안 잡음 표준편차(0~255)
    blur_ksize: int = 0         # 홀수 커널. 0이면 미적용
    brightness: float = 1.0     # 1.0 기준 배율. 어둡게 하려면 <1
    seed: int = 42


class SensorDegrader:
    """한 모달리티 스트림에 열화를 주입한다.

    반환이 None이면 그 프레임에서 해당 센서는 관측 없음(LOST)을 뜻한다.
    """

    def __init__(self, cfg: DegradeConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._buf: deque = deque(maxlen=max(cfg.delay_frames + 1, 1))
        self.n_dropped = 0
        self.n_total = 0

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self.n_total += 1
        c = self.cfg

        # 1) 지연: 버퍼에 넣고 delay만큼 지난 프레임을 꺼낸다
        self._buf.append(frame)
        if c.delay_frames > 0:
            if len(self._buf) <= c.delay_frames:
                return None  # 아직 출력할 과거 프레임이 없다
            out = self._buf[0]
        else:
            out = frame

        # 2) 결측
        if c.dropout > 0 and self._rng.random() < c.dropout:
            self.n_dropped += 1
            return None

        out = out.copy()

        # 3) 밝기
        if c.brightness != 1.0:
            out = np.clip(out.astype(np.float32) * c.brightness, 0, 255).astype(np.uint8)

        # 4) 흐림
        if c.blur_ksize and c.blur_ksize >= 3:
            k = c.blur_ksize | 1
            out = cv2.GaussianBlur(out, (k, k), 0)

        # 5) 잡음
        if c.noise_sigma > 0:
            noise = self._rng.normal(0, c.noise_sigma, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return out

    @property
    def drop_rate(self) -> float:
        return self.n_dropped / self.n_total if self.n_total else 0.0
