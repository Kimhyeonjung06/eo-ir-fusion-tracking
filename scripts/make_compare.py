"""COCO 사전학습 vs 미세조정 비교 그림.

이 저장소의 핵심 결론을 보여주는 그림이다.
융합의 값어치는 단일 센서 성능에 반비례한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

INK = "#141414"
ACC = "#0F4C4A"
WARM = "#B4551F"
GRAY = "#8A8A8A"
BASE = ["EO_only", "IR_only", "Fusion"]
LABEL = {"EO_only": "EO 단독", "IR_only": "IR 단독", "Fusion": "융합"}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_effect(co: pd.DataFrame, ft: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), dpi=150, sharey=True)
    for ax, (df, title) in zip(axes, [(co, "COCO 사전학습"), (ft, "KAIST 미세조정")]):
        piv = (df[df.condition.isin(BASE)]
               .pivot_table(index="time_of_day", columns="condition", values="ap50")
               .reindex(["day", "night"])[BASE])
        x = np.arange(2)
        w = 0.26
        for i, c in enumerate(BASE):
            ax.bar(x + (i - 1) * w, piv[c].values, w,
                   color=[GRAY, WARM, ACC][i], label=LABEL[c], edgecolor="none")
        style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(["주간", "야간"])
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("AP@0.5", color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("단일 센서가 강해지면 융합의 이득이 사라진다", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig4_finetune_effect.png")
    plt.close(fig)


def fig_tradeoff(co: pd.DataFrame, ft: pd.DataFrame, out: Path) -> None:
    rows = []
    for df, name in [(co, "COCO"), (ft, "미세조정")]:
        s = df[df.condition.isin(BASE)].groupby("condition")[["op_precision", "op_recall"]].mean()
        rows.append((name,
                     s.loc["Fusion", "op_precision"] - s.loc["IR_only", "op_precision"],
                     s.loc["Fusion", "op_recall"] - s.loc["IR_only", "op_recall"]))

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    x = np.arange(len(rows))
    w = 0.32
    ax.bar(x - w / 2, [r[1] for r in rows], w, color=WARM, label="정밀도 변화", edgecolor="none")
    ax.bar(x + w / 2, [r[2] for r in rows], w, color=ACC, label="재현율 변화", edgecolor="none")
    ax.axhline(0, color=INK, linewidth=0.9)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("IR 단독 대비 변화", color=INK)
    ax.set_title("융합의 손익 — 모델이 강할수록 오탐만 합쳐진다", color=INK, fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig5_fusion_tradeoff.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="results/coco640/results.csv")
    ap.add_argument("--finetuned", default="results/finetuned/results.csv")
    ap.add_argument("--out", default="results/compare")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    co = pd.read_csv(args.coco)
    ft = pd.read_csv(args.finetuned)
    fig_effect(co, ft, out)
    fig_tradeoff(co, ft, out)

    # 표도 함께 남긴다
    lines = ["### COCO 사전학습 vs KAIST 미세조정 (둘 다 640 입력)\n"]
    piv = pd.concat([
        co[co.condition.isin(BASE)].assign(model="coco"),
        ft[ft.condition.isin(BASE)].assign(model="finetuned"),
    ]).pivot_table(index=["time_of_day", "condition"], columns="model", values="ap50").round(3)
    lines.append(piv.to_markdown())
    (out / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print(piv.to_string())
    print("\n저장:", ", ".join(p.name for p in sorted(out.glob("*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
