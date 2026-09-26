"""평가 지표.

탐지: greedy IoU 매칭 기반 precision / recall / F1 / AP(all-point)
추적: 정답 ID가 없으므로 조건 간 상대 비교용 안정성 지표를 쓴다.
     (MOTA/HOTA는 정답 ID가 필요하므로 산출하지 않는다)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from fusion import iou_matrix


def match_frame(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5):
    if len(pred) == 0:
        return [], [], list(range(len(gt)))
    if len(gt) == 0:
        return [], list(range(len(pred))), []
    order = np.argsort(-pred[:, 4])
    ious = iou_matrix(pred[:, :4], gt[:, :4])
    used_gt, pairs, fp = set(), [], []
    for pi in order:
        j = int(np.argmax(ious[pi]))
        if ious[pi, j] >= iou_thr and j not in used_gt:
            used_gt.add(j)
            pairs.append((int(pi), j))
        else:
            fp.append(int(pi))
    fn = [j for j in range(len(gt)) if j not in used_gt]
    return pairs, fp, fn


def detection_scores(gts: Sequence[np.ndarray], preds: Sequence[np.ndarray],
                     iou_thr: float = 0.5) -> Dict[str, float]:
    tp = fp = fn = 0
    scored: List[Tuple[float, int]] = []
    n_gt = 0
    for gt, pred in zip(gts, preds):
        gt = np.asarray(gt, np.float32)
        gt = gt.reshape(-1, 4) if gt.size else np.zeros((0, 4), np.float32)
        pred = np.asarray(pred, np.float32)
        pred = pred.reshape(-1, pred.shape[-1]) if pred.size else np.zeros((0, 5), np.float32)
        n_gt += len(gt)
        pairs, fps, fns = match_frame(gt, pred, iou_thr)
        tp += len(pairs)
        fp += len(fps)
        fn += len(fns)
        tp_idx = {p for p, _ in pairs}
        for i in range(len(pred)):
            scored.append((float(pred[i, 4]), 1 if i in tp_idx else 0))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    ap = 0.0
    if scored and n_gt:
        scored.sort(key=lambda x: -x[0])
        hits = np.array([s[1] for s in scored], dtype=np.float32)
        ctp = np.cumsum(hits)
        cfp = np.cumsum(1.0 - hits)
        recalls = ctp / n_gt
        precisions = ctp / np.maximum(ctp + cfp, 1e-9)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        prev = np.concatenate([[0.0], recalls[:-1]])
        ap = float(np.sum((recalls - prev) * precisions))

    return {"precision": prec, "recall": rec, "f1": f1, "ap50": ap,
            "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt}


def track_stability(track_log: Sequence[Sequence[Tuple[int, np.ndarray]]]) -> Dict[str, float]:
    """track_log: 프레임별 [(track_id, box), ...]"""
    lifespan: Dict[int, int] = defaultdict(int)
    last_seen: Dict[int, int] = {}
    gaps: Dict[int, int] = defaultdict(int)
    for f, frame in enumerate(track_log):
        for tid, _ in frame:
            lifespan[tid] += 1
            if tid in last_seen and f - last_seen[tid] > 1:
                gaps[tid] += 1
            last_seen[tid] = f
    n = len(lifespan)
    return {"n_tracks": n,
            "mean_track_len": float(np.mean(list(lifespan.values()))) if n else 0.0,
            "frag_per_track": float(sum(gaps.values()) / n) if n else 0.0,
            "total_frag": int(sum(gaps.values()))}
