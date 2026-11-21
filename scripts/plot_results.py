"""Maak de grafieken van docs/experimenten.md uit de cijfers in results/ -> docs/img/.

uv run python scripts/plot_results.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT

RESULTS = ROOT / "results"
OUT = ROOT / "docs/img"
ORANGE, LIGHT, GRAY, INK, GRID = "#eb6834", "#f5b89c", "#8a8a85", "#3d3d3a", "#e4e3df"


def read(name: str) -> list[dict]:
    """Lees een csv uit results/. Getallen worden float."""
    with (RESULTS / f"{name}.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [{key: to_number(value) for key, value in row.items()} for row in rows]


def to_number(value: str) -> float | str:
    """Tekst -> getal als dat kan."""
    try:
        return float(value)
    except ValueError:
        return value


def style(ax, xlabel: str = "", ylabel: str = "") -> None:
    """Rustige assen: geen kader, lichte hulplijnen."""
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, length=0)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    """Bewaar de grafiek als png."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(OUT / f"{name}.png")


def plot_what_matters() -> None:
    """mAP50 op de val-set als je 1 ding verandert ten opzichte van de basis."""
    rows = read("wat_telt")[::-1]  # bovenste rij van de csv bovenaan
    ys = range(len(rows))
    values = [row["map50"] for row in rows]
    colors = [GRAY if row["instelling"].startswith("basis") else ORANGE for row in rows]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(ys, values, height=0.6, color=colors)
    for y, value in zip(ys, values, strict=True):
        ax.text(value + 1, y, f"{value:.1f}", va="center", color=INK)
    ax.set_yticks(ys, [row["instelling"] for row in rows])
    ax.set_xlim(0, 108)
    ax.xaxis.grid(True, color=GRID)
    style(ax, xlabel="mAP50 op de val-set (7 foto's)")
    ax.set_title("Wat telt", loc="left", color=INK)
    save(fig, "wat_telt")


def plot_resolution() -> None:
    """mAP50 tegen de invoerresolutie van YOLO11m."""
    rows = read("resolutie")
    xs = [row["pixels"] for row in rows]
    ys = [row["map50"] for row in rows]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(xs, ys, color=ORANGE, marker="o", markersize=7, linewidth=2)
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                    color=INK)  # fmt: skip
    ax.set_xticks(xs)
    ax.set_ylim(90, 100)
    ax.yaxis.grid(True, color=GRID)
    style(ax, xlabel="invoerresolutie (pixels; de foto's zelf zijn 2048 x 1536)", ylabel="mAP50")
    ax.set_title("Meer pixels helpt niet", loc="left", color=INK)
    save(fig, "resolutie")


def plot_trap() -> None:
    """De beste op de val-set tegen de basis, op de val-set en op de testset."""
    rows = read("valkuil")
    groups = [("keuzeset", "val-set\n(7 foto's)"), ("testset_eigen", "testset\negale ondergrond"),
              ("testset_dambord", "testset\ndambord")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for offset, row, color in ((-0.2, rows[0], GRAY), (0.2, rows[1], ORANGE)):
        for i, (key, _) in enumerate(groups):
            ax.bar(i + offset, row[key], width=0.36, color=color)
            ax.text(i + offset, row[key] + 1, f"{row[key]:.1f}", ha="center", color=INK)
    ax.set_xticks(range(len(groups)), [label for _, label in groups])
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color=GRID)
    style(ax, ylabel="mAP50")
    ax.set_title("Grijs = basis, oranje = beste op de val-set", loc="left", color=INK)
    save(fig, "valkuil")


def main() -> None:
    """Maak alle grafieken."""
    plot_what_matters()
    plot_resolution()
    plot_trap()


if __name__ == "__main__":
    main()
