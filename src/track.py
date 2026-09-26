"""다중 객체 추적기.

ByteTrack의 2단계 연관(고신뢰 먼저, 남은 트랙을 저신뢰와 다시 연관)을 재구현했다.
외부 추적기를 쓰지 않은 이유는 이 프로젝트의 핵심이 센서가 끊겼을 때 트랙을 어떻게
유지하느냐이고, 그 정책을 추적기 내부에서 제어해야 하기 때문이다.
칼만 필터 대신 직전 변위를 이용한 등속 예측을 쓴다(프레임 간격이 일정한 데이터 전제).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from fusion import iou_matrix, _assign


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    conf: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    confirmed: bool = False
    n_gaps: int = 0
    _was_lost: bool = False

    def predict(self) -> np.ndarray:
        return self.box + self.velocity

    def update(self, box: np.ndarray, conf: float) -> None:
        self.velocity = 0.5 * self.velocity + 0.5 * (box - self.box)
        self.box = box.astype(np.float32)
        self.conf = float(conf)
        self.hits += 1
        self.time_since_update = 0
        if self._was_lost:
            self.n_gaps += 1
            self._was_lost = False

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.box = self.predict()
        self._was_lost = True


class ByteLikeTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, match_iou=0.3, max_age=30, min_hits=3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_idx = 0

    def _match(self, tracks, dets):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        pred = np.stack([t.predict() for t in tracks])
        cost = 1.0 - iou_matrix(pred, dets[:, :4])
        return _assign(cost, thresh=1.0 - self.match_iou)

    def update(self, dets: np.ndarray) -> List[Track]:
        self.frame_idx += 1
        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim != 2 or dets.shape[0] == 0:
            dets = np.zeros((0, 5), dtype=np.float32)

        if len(dets):
            high = dets[dets[:, 4] >= self.high_thresh]
            low = dets[(dets[:, 4] < self.high_thresh) & (dets[:, 4] >= self.low_thresh)]
        else:
            high = np.zeros((0, 5), np.float32)
            low = np.zeros((0, 5), np.float32)

        m, u_trk, u_det = self._match(self.tracks, high)
        for ti, di in m:
            self.tracks[ti].update(high[di, :4], high[di, 4])

        rest = [self.tracks[i] for i in u_trk]
        m2, u_trk2, _ = self._match(rest, low)
        for ri, di in m2:
            rest[ri].update(low[di, :4], low[di, 4])
        for ri in u_trk2:
            rest[ri].mark_missed()

        for di in u_det:
            self.tracks.append(Track(track_id=self._next_id,
                                     box=high[di, :4].astype(np.float32),
                                     conf=float(high[di, 4])))
            self._next_id += 1

        alive = []
        for t in self.tracks:
            t.age += 1
            if t.hits >= self.min_hits:
                t.confirmed = True
            if t.time_since_update <= self.max_age:
                alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]
