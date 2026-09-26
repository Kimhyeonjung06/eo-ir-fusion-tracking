"""EO(가시광)/IR(열화상) 탐지 결과의 late fusion.

두 모달리티에서 각각 얻은 박스를 IoU로 짝지어 하나의 박스로 합친다.
전제: 두 영상이 공간적으로 정합(registered)되어 있어야 한다.
"""
from __future__ import annotations

import numpy as np

try:
    import lap  # ultralytics 의존성으로 이미 설치돼 있음
    _HAS_LAP = True
except Exception:  # pragma: no cover
    _HAS_LAP = False


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = np.split(a[:, :4], 4, axis=1)
    bx1, by1, bx2, by2 = np.split(b[:, :4], 4, axis=1)
    ix1 = np.maximum(ax1, bx1.T)
    iy1 = np.maximum(ay1, by1.T)
    ix2 = np.minimum(ax2, bx2.T)
    iy2 = np.minimum(ay2, by2.T)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b.T - inter + 1e-9
    return (inter / union).astype(np.float32)


def _assign(cost: np.ndarray, thresh: float):
    """헝가리안 할당. lap이 있으면 lapjv, 없으면 탐욕적 매칭."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if _HAS_LAP:
        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
        matches = [(i, int(x[i])) for i in range(len(x)) if x[i] >= 0]
        ua = [i for i in range(len(x)) if x[i] < 0]
        ub = [j for j in range(len(y)) if y[j] < 0]
        return matches, ua, ub
    # 탐욕적 대체
    matches, ua, ub = [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    used_a, used_b = set(), set()
    for i, j in order:
        if cost[i, j] > thresh:
            break
        if i in used_a or j in used_b:
            continue
        matches.append((int(i), int(j)))
        used_a.add(i)
        used_b.add(j)
    ua = [i for i in ua if i not in used_a]
    ub = [j for j in ub if j not in used_b]
    return matches, ua, ub


def late_fuse(
    det_rgb: np.ndarray,
    det_ir: np.ndarray,
    iou_thresh: float = 0.5,
    w_rgb: float = 0.5,
    keep_unmatched: bool = True,
    unmatched_penalty: float = 0.9,
) -> np.ndarray:
    """두 모달리티 탐지 결과를 합친다.

    det_*: (N,5) = [x1, y1, x2, y2, conf]
    반환:  (K,6) = [x1, y1, x2, y2, conf, src]
           src 0=RGB단독, 1=IR단독, 2=융합
    """
    det_rgb = np.asarray(det_rgb, dtype=np.float32).reshape(-1, 5)
    det_ir = np.asarray(det_ir, dtype=np.float32).reshape(-1, 5)

    ious = iou_matrix(det_rgb, det_ir)
    cost = 1.0 - ious
    matches, u_rgb, u_ir = _assign(cost, thresh=1.0 - iou_thresh)

    out = []
    for i, j in matches:
        cr, ci = det_rgb[i, 4], det_ir[j, 4]
        # 신뢰도 가중 평균으로 좌표를 합친다
        wr = w_rgb * cr
        wi = (1.0 - w_rgb) * ci
        s = wr + wi + 1e-9
        box = (det_rgb[i, :4] * wr + det_ir[j, :4] * wi) / s
        # 두 센서가 모두 본 표적은 신뢰도를 높인다 (noisy-or)
        conf = 1.0 - (1.0 - cr) * (1.0 - ci)
        out.append([*box, conf, 2])
    if keep_unmatched:
        for i in u_rgb:
            out.append([*det_rgb[i, :4], det_rgb[i, 4] * unmatched_penalty, 0])
        for j in u_ir:
            out.append([*det_ir[j, :4], det_ir[j, 4] * unmatched_penalty, 1])
    if not out:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)
