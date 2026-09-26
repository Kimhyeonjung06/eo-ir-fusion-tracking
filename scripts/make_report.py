"""results.csv -> 표(마크다운)와 그래프.

만드는 그림은 셋이다.
  1) 주간/야간별 EO vs IR vs Fusion  — "IR은 언제 이기는가"
  2) 결손률에 따른 저하 곡선          — "얼마나 버티는가"
  3) EO 결손 vs IR 결손 비대칭 비교   — "어느 쪽 결손에 더 강한가"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

BASE = ["EO_only", "IR_only", "Fusion"]
INK = "#141414"
ACC = "#0F4C4A"
WARM = "#B4551F"
GRAY = "#8A8A8A"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_day_night(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["condition"].isin(BASE)]
    piv = sub.groupby(["time_of_day", "condition"])["ap50"].mean().unstack()
    piv = piv.reindex(columns=[c for c in BASE if c in piv.columns])
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
    piv.plot(kind="bar", ax=ax, color=[GRAY, WARM, ACC], width=0.72, edgecolor="none")
    _style(ax)
    ax.set_ylabel("AP@0.5", color=INK)
    ax.set_xlabel("")
    ax.set_title("주간·야간별 단일 센서와 융합 비교", color=INK, fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9)
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out / "fig1_day_night.png")
    plt.close(fig)


def _drop_curve(df: pd.DataFrame, prefix: str):
    rows = [(0, df[df["condition"] == "Fusion"]["ap50"].mean(),
             df[df["condition"] == "Fusion"]["recall"].mean(),
             df[df["condition"] == "Fusion"]["frag_per_track"].mean())]
    for r in (10, 30, 50, 70):
        name = "%s%d" % (prefix, r)
        s = df[df["condition"] == name]
        if len(s):
            rows.append((r, s["ap50"].mean(), s["recall"].mean(), s["frag_per_track"].mean()))
    return pd.DataFrame(rows, columns=["drop", "ap50", "recall", "frag"])


def fig_degradation(df: pd.DataFrame, out: Path) -> None:
    ir = _drop_curve(df, "Fusion_IRdrop")
    eo = _drop_curve(df, "Fusion_EOdrop")
    if len(ir) < 2 and len(eo) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150)
    for ax, col, label in zip(axes, ["ap50", "frag_per_track"], ["AP@0.5", "트랙당 단편화"]):
        key = "ap50" if col == "ap50" else "frag"
        ax.plot(ir["drop"], ir[key], marker="o", color=WARM, label="IR 결손")
        ax.plot(eo["drop"], eo[key], marker="s", color=ACC, label="EO 결손")
        _style(ax)
        ax.set_xlabel("결손률 (%)", color=INK)
        ax.set_ylabel(label, color=INK)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("한쪽 센서 결손에 따른 저하 — 어느 쪽을 잃는 것이 더 아픈가",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_degradation.png")
    plt.close(fig)


def fig_delay(df: pd.DataFrame, out: Path) -> None:
    rows = [(0, df[df["condition"] == "Fusion"]["ap50"].mean())]
    for d in (1, 2, 5):
        s = df[df["condition"] == "Fusion_IRdelay%d" % d]
        if len(s):
            rows.append((d, s["ap50"].mean()))
    if len(rows) < 2:
        return
    t = pd.DataFrame(rows, columns=["delay", "ap50"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    ax.plot(t["delay"], t["ap50"], marker="o", color=ACC)
    _style(ax)
    ax.set_xlabel("IR 지연 (프레임)", color=INK)
    ax.set_ylabel("AP@0.5", color=INK)
    ax.set_title("시간 어긋남이 융합에 미치는 영향", color=INK, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out / "fig3_delay.png")
    plt.close(fig)


def markdown_tables(df: pd.DataFrame) -> str:
    cols = ["condition", "ap50", "precision", "recall", "f1",
            "n_tracks", "mean_track_len", "frag_per_track"]
    out = ["### 전체 평균\n"]
    g = df.groupby("condition")[["ap50", "precision", "recall", "f1",
                                 "n_tracks", "mean_track_len", "frag_per_track"]].mean()
    order = [c for c in BASE if c in g.index] + [c for c in g.index if c not in BASE]
    out.append(g.reindex(order).round(3).to_markdown())

    for tod in sorted(df["time_of_day"].unique()):
        sub = df[(df["time_of_day"] == tod) & (df["condition"].isin(BASE))]
        if sub.empty:
            continue
        out.append("\n### %s\n" % ("주간" if tod == "day" else "야간" if tod == "night" else tod))
        gg = sub.groupby("condition")[["ap50", "precision", "recall", "f1", "frag_per_track"]].mean()
        out.append(gg.reindex([c for c in BASE if c in gg.index]).round(3).to_markdown())

    lat = df[["lat_p50_ms", "lat_p95_ms", "lat_max_ms"]].mean().round(2)
    out.append("\n### 지연 (EO+IR 탐지 합, 프레임당 ms)\n")
    out.append("| p50 | p95 | 최대 |\n|---:|---:|---:|\n| %s | %s | %s |"
               % (lat["lat_p50_ms"], lat["lat_p95_ms"], lat["lat_max_ms"]))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    fig_day_night(df, out)
    fig_degradation(df, out)
    fig_delay(df, out)

    md = markdown_tables(df)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print("\n그림 저장:", ", ".join(p.name for p in sorted(out.glob("fig*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
