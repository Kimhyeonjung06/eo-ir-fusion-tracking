"""센서 상태 감시와 융합 전략 판단.

고정 가중 융합은 센서가 정상이라는 전제 위에서만 옳다.
한쪽이 끊기거나 시간이 어긋나면 그 전제가 깨지는데, 두 상황은 관측에 남는 흔적이 다르다.

    결손 - 관측 자체가 들어오지 않는다.        가용성으로 잡힌다.
    지연 - 관측은 계속 들어오지만 위치가 틀리다. 가용성으로는 잡히지 않고,
           두 모달의 박스가 서로 겹치지 않는 것으로만 드러난다.

그래서 가용성, 품질, 정합성 세 신호를 따로 본다.
밝기나 시각 같은 외부 정보는 쓰지 않는다. 탐지 결과만으로 판단한다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

NORMAL = "NORMAL"
DEGRADED = "DEGRADED"
LOST = "LOST"


@dataclass
class HealthConfig:
    window: int = 30            # 이동 평균 구간(프레임)
    lost_after: int = 5         # 연속 무관측이 이만큼이면 LOST
    degraded_obs: float = 0.7   # 최근 관측률이 이 아래면 DEGRADED
    min_conf: float = 0.05      # 품질 하한. 0이면 가중치 계산이 무너진다


class SensorHealth:
    """한 모달의 상태를 프레임 단위로 추정한다."""

    def __init__(self, cfg: Optional[HealthConfig] = None):
        self.cfg = cfg or HealthConfig()
        self._obs = deque(maxlen=self.cfg.window)    # 1=관측 있음
        self._conf = deque(maxlen=self.cfg.window)   # 프레임 최고 신뢰도
        self._miss_streak = 0
        self.state = NORMAL

    def update(self, dets: Optional[np.ndarray]) -> str:
        present = dets is not None
        self._obs.append(1.0 if present else 0.0)
        if present:
            self._miss_streak = 0
            self._conf.append(float(dets[:, 4].max()) if len(dets) else 0.0)
        else:
            self._miss_streak += 1

        if self._miss_streak >= self.cfg.lost_after:
            self.state = LOST
        elif self.obs_rate < self.cfg.degraded_obs:
            self.state = DEGRADED
        else:
            self.state = NORMAL
        return self.state

    @property
    def obs_rate(self) -> float:
        return float(np.mean(self._obs)) if self._obs else 1.0

    @property
    def quality(self) -> float:
        """최근 탐지 신뢰도의 평균. 모달 간 우열을 재는 데 쓴다."""
        if not self._conf:
            return self.cfg.min_conf
        return max(float(np.mean(self._conf)), self.cfg.min_conf)


class ConsistencyMonitor:
    """두 모달이 같은 표적을 같은 자리에서 보고 있는지 본다.

    정합성이 떨어지는 원인은 둘이다. 시간이 어긋났거나, 한쪽이 엉뚱한 것을 보고 있거나.
    어느 쪽이든 결과 수준에서 박스를 합치면 안 되는 상황이다.
    """

    def __init__(self, window: int = 20):
        self._hist = deque(maxlen=window)

    def update(self, n_match: int, mean_iou: float, n_eo: int, n_ir: int) -> None:
        # 양쪽 모두 탐지가 있을 때만 정합성을 따진다.
        # 한쪽이 비어 있는 것은 정합성 문제가 아니라 가용성 문제다.
        if n_eo == 0 or n_ir == 0:
            return
        pairable = min(n_eo, n_ir)
        match_rate = n_match / pairable if pairable else 0.0
        self._hist.append(match_rate * mean_iou)

    @property
    def score(self) -> float:
        """0에 가까우면 두 모달이 서로 다른 것을 보고 있다는 뜻이다."""
        return float(np.mean(self._hist)) if self._hist else 1.0

    @property
    def warmed(self) -> bool:
        return len(self._hist) >= 5


class LagMonitor:
    """각 모달이 몇 프레임 뒤처져 들어오는지 추정한다.

    단순히 예측과 얼마나 맞는지만 보면 지연과 성능 차이를 구분할 수 없다.
    야간에는 IR이 원래 더 잘 맞으므로 격차가 항상 벌어져 있다.

    지연은 방향을 가진다. 늦게 도착한 관측은 표적이 지나온 자리,
    즉 진행 방향의 반대쪽에 찍힌다. 예측 위치에서 관측까지의 변위를
    트랙 속도 방향으로 투영하고 속도 크기로 나누면 뒤처진 프레임 수가 나온다.
    성능이 나쁜 센서의 오차는 방향이 무작위라 평균에서 상쇄되지만,
    지연은 한 방향으로만 쌓이므로 남는다.
    """

    def __init__(self, window: int = 15, min_speed: float = 0.6):
        self._rel = deque(maxlen=window)
        self._agree_eo = deque(maxlen=window * 2)
        self._agree_ir = deque(maxlen=window * 2)
        self.min_speed = min_speed

    @staticmethod
    def _centers(boxes: np.ndarray) -> np.ndarray:
        return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                         (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)

    def update(self, tracks, det_eo, det_ir) -> None:
        """두 모달을 서로 직접 비교한다.

        각 모달을 트랙 예측과 견주면 지연을 놓친다. 지연된 관측이 이미 트랙에 반영되어
        트랙 자체가 뒤처져 있기 때문이다. 두 모달의 상대 변위를 보면 그 순환에서 벗어난다.
        """
        from fusion import iou_matrix

        # 관측 가운데 확립된 트랙과 맞는 비율. 오탐이 많은 모달일수록 낮게 나온다.
        if tracks:
            pred = np.stack([t.predict() for t in tracks])
            for det, buf in ((det_eo, self._agree_eo), (det_ir, self._agree_ir)):
                if det is None or len(det) == 0:
                    continue
                m = iou_matrix(det[:, :4], pred)
                buf.append(float(np.mean(m.max(axis=1) > 0.3)))

        if det_eo is None or det_ir is None or len(det_eo) == 0 or len(det_ir) == 0:
            return
        if not tracks:
            return

        vel = np.stack([t.velocity for t in tracks])
        v = np.stack([(vel[:, 0] + vel[:, 2]) / 2.0, (vel[:, 1] + vel[:, 3]) / 2.0], axis=1)
        speed = np.linalg.norm(v, axis=1)
        moving = speed >= self.min_speed
        if not moving.any():
            return
        v_mean = v[moving].mean(axis=0)
        s2 = float(np.dot(v_mean, v_mean))
        if s2 <= 0:
            return

        m = iou_matrix(det_eo[:, :4], det_ir[:, :4])
        best = m.argmax(axis=1)
        hit = m.max(axis=1) > 0.2
        if not hit.any():
            return

        eo_c = self._centers(det_eo[:, :4])[hit]
        ir_c = self._centers(det_ir[:, :4])[best[hit]]
        # IR 박스가 진행 방향 뒤쪽에 밀려 있으면 음수가 나온다. 그 값이 곧 뒤처진 프레임 수다.
        rel = float(np.median((ir_c - eo_c) @ v_mean / s2))
        self._rel.append(rel)

    @property
    def relative(self) -> float:
        """음수면 IR이, 양수면 EO가 그만큼 뒤처져 있다."""
        return float(np.median(self._rel)) if self._rel else 0.0

    @property
    def gap(self) -> float:
        """decide가 쓰는 부호 규약: 양수면 IR이 뒤처짐."""
        return -self.relative

    @property
    def warmed(self) -> bool:
        return len(self._rel) >= 5

    @property
    def agree_eo(self) -> float:
        return float(np.mean(self._agree_eo)) if self._agree_eo else 0.5

    @property
    def agree_ir(self) -> float:
        return float(np.mean(self._agree_ir)) if self._agree_ir else 0.5

    @property
    def agree_warmed(self) -> bool:
        return len(self._agree_eo) >= 10 and len(self._agree_ir) >= 10


@dataclass
class Decision:
    mode: str          # fuse | eo_only | ir_only | none
    w_eo: float
    w_ir: float
    reason: str


@dataclass
class PolicyConfig:
    agreement_power: float = 2.0      # 클수록 오탐 많은 센서를 세게 깎는다
    lag_scale: float = 2.0            # 이만큼 뒤처지면 그 센서의 비중을 0까지 깎는다
    drop_to_single: float = 0.06      # 가중치가 이보다 작아지면 사실상 단독으로 본다
    quality_ratio_cap: float = 0.97   # 한쪽으로 쏠리는 정도의 상한
    degraded_penalty: float = 0.6     # DEGRADED 모달의 품질을 깎는 비율


def decide(h_eo: SensorHealth, h_ir: SensorHealth,
           lag: LagMonitor,
           cfg: Optional[PolicyConfig] = None) -> Decision:
    """센서 상태를 보고 이번 프레임에 어떤 전략을 쓸지 정한다.

    판단 순서
      1. 관측이 없는 센서는 뺀다 (가용성)
      2. 추적 예측과 어긋나는 센서는 뺀다 (지연·오정렬)
      3. 둘 다 멀쩡하면 최근 탐지 품질 비율로 가중 융합
    """
    cfg = cfg or PolicyConfig()
    eo_up = h_eo.state != LOST
    ir_up = h_ir.state != LOST

    if not eo_up and not ir_up:
        return Decision("none", 0.0, 0.0, "both sensors lost")
    if not ir_up:
        return Decision("eo_only", 1.0, 0.0, "IR lost")
    if not eo_up:
        return Decision("ir_only", 0.0, 1.0, "EO lost")

    q_eo = h_eo.quality * (cfg.degraded_penalty if h_eo.state == DEGRADED else 1.0)
    q_ir = h_ir.quality * (cfg.degraded_penalty if h_ir.state == DEGRADED else 1.0)

    # 확립된 트랙과 맞지 않는 관측이 많은 센서는 오탐을 쏟아내고 있다는 뜻이다.
    # 그 비중을 줄이는 것이 융합의 가장 큰 손해인 정밀도 하락을 막는 길이다.
    if lag.agree_warmed:
        q_eo *= max(lag.agree_eo, 0.05) ** cfg.agreement_power
        q_ir *= max(lag.agree_ir, 0.05) ** cfg.agreement_power

    # 뒤처진 센서의 비중을 뒤처진 만큼 깎는다.
    # 끄고 켜는 대신 연속으로 줄이는 이유는, 추정값이 흔들려도 결과가 요동치지 않게 하기 위해서다.
    reason = "quality EO %.2f / IR %.2f" % (q_eo, q_ir)
    if lag.warmed:
        gap = lag.gap                      # 양수면 IR이 더 뒤처져 있다
        penalty = max(0.0, 1.0 - abs(gap) / cfg.lag_scale)
        if abs(gap) > 0.2:
            if gap > 0:
                q_ir *= penalty
            else:
                q_eo *= penalty
            reason = "%s lags %.1f frames, weight x%.2f" % ("IR" if gap > 0 else "EO",
                                                            abs(gap), penalty)

    total = q_eo + q_ir
    w_eo = q_eo / total if total > 0 else 0.5
    w_eo = min(max(w_eo, 1.0 - cfg.quality_ratio_cap), cfg.quality_ratio_cap)
    w_ir = 1.0 - w_eo

    # 한쪽 비중이 사실상 사라지면 굳이 합치지 않는다.
    if w_ir < cfg.drop_to_single:
        return Decision("eo_only", 1.0, 0.0, reason)
    if w_eo < cfg.drop_to_single:
        return Decision("ir_only", 0.0, 1.0, reason)
    return Decision("fuse", w_eo, w_ir, reason)
