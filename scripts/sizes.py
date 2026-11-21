"""Maten: binnen een klasse klein en groot (of kort en lang) uit elkaar houden.

    uv run python scripts/sizes.py      # zoek de maten in de labels -> data/maten.yaml

YOLO kent alleen de klasse. De maat halen we uit de lengte van de box. Pixels zeggen niks, want
de camera hangt niet steeds even hoog. Daarom delen we de lengte door die van de moer op dezelfde
foto: de moeren zijn allemaal even groot. evaluate.py zet daarna bij elk onderdeel de maat.
"""

import numpy as np
import yaml
from common import ROOT, load_classes, load_label, load_split

SIZES = ROOT / "data/maten.yaml"
REFERENCE = "moer"  # deze klasse is de meetlat
MIN_EXAMPLES = 8  # met minder voorbeelden zoeken we geen maten
MIN_RATIO = 1.4  # een en dezelfde maat schommelt tot ongeveer 1.25x door perspectief
SURE = 0.5  # alleen moeren waar het model zeker genoeg van is tellen als meetlat
NAMES = {"zeskantbout": ("kort", "lang"), "slotbout": ("kort", "lang")}  # de rest: klein, groot


def length(obj: dict) -> float:
    """Langste kant van de box."""
    return max(obj["size"])


def reference(objects: list[dict]) -> float | None:
    """De meetlat van een foto: de mediaan van de moeren. None als er geen moer ligt."""
    nuts = [length(o) for o in objects
            if o["class"] == REFERENCE and o.get("confidence", 1.0) >= SURE]  # fmt: skip
    if not nuts:
        return None
    return float(np.median(nuts))


def relative_lengths(stems: list[str]) -> dict[str, list[float]]:
    """Per klasse de lengtes gedeeld door de moer, van onderdelen die helemaal zichtbaar zijn."""
    lengths = {cls: [] for cls in load_classes() if cls != REFERENCE}
    for stem in stems:
        objects = load_label(stem)["objects"]
        ref = reference(objects)
        if ref is None:
            continue
        for obj in objects:
            if obj["class"] in lengths and not obj.get("occluded"):
                lengths[obj["class"]].append(length(obj) / ref)
    return lengths


def two_groups(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Deel de waarden in een kleine en een grote groep, met zo min mogelijk spreiding per groep.

    Dit is hetzelfde idee als Otsu bij een drempel voor grijswaarden.
    """
    logs = np.sort(np.log(values))  # log: 2 keer zo groot telt overal even zwaar
    costs = [logs[:i].var() * i + logs[i:].var() * (len(logs) - i) for i in range(1, len(logs))]
    cut = 1 + int(np.argmin(costs))
    return np.exp(logs[:cut]), np.exp(logs[cut:])


def find_sizes(stems: list[str]) -> dict[str, float]:
    """Per klasse met 2 maten de grens tussen klein en groot (als deel van de moer)."""
    borders = {}
    for cls, values in relative_lengths(stems).items():
        if len(values) < MIN_EXAMPLES:
            print(f"  {cls:17s} {len(values):3d} voorbeelden, te weinig")
            continue
        small, large = two_groups(values)
        ratio = large.mean() / small.mean()
        verdict = "2 maten" if ratio >= MIN_RATIO else "1 maat"
        groups = f"{small.mean():.2f} en {large.mean():.2f} keer de moer ({ratio:.2f}x)"
        print(f"  {cls:17s} {len(values):3d} voorbeelden, {groups} -> {verdict}")
        if ratio >= MIN_RATIO:
            borders[cls] = round(float(np.sqrt(small.max() * large.min())), 2)  # midden ertussen
    return borders


def load_sizes() -> dict[str, float]:
    """De grenzen uit data/maten.yaml."""
    return yaml.safe_load(SIZES.read_text())


def add_sizes(objects: list[dict], borders: dict[str, float]) -> None:
    """Zet bij elk onderdeel van een klasse met 2 maten de maat erbij (sleutel `maat`)."""
    ref = reference(objects)
    if ref is None:
        return
    for obj in objects:
        if obj["class"] not in borders:
            continue
        small, large = NAMES.get(obj["class"], ("klein", "groot"))
        obj["maat"] = large if length(obj) / ref > borders[obj["class"]] else small


def main() -> None:
    """Zoek de maten in de train- en val-foto's van de egale ondergrond en bewaar de grenzen."""
    split = load_split()
    stems = [s for s in split["train"] + split["val"] if s.startswith("ps_")]
    print(f"Lengte gedeeld door de moer, {len(stems)} foto's:")
    borders = find_sizes(stems)
    SIZES.write_text(yaml.safe_dump(borders, sort_keys=False))
    print(f"-> {SIZES}")


if __name__ == "__main__":
    main()
