"""Labels van en naar CVAT, zodat je ze daar kunt maken en verbeteren.

In CVAT is elk onderdeel een gedraaide box (rectangle met rotatie) en de kop een losse punt met
label `kop`. Je hoeft de kop niet precies te zetten en ook niet te groeperen: één klik ergens bij
het kop-uiteinde is genoeg. Bij het terugzetten zoekt dit script de bijbehorende box en kiest de
zijde die het dichtst bij je klik ligt als kop: de hoek wijst dan van kop naar punt, size wordt
[lengte, breedte] in die richting en `head` komt precies op het midden van die zijde, zoals in de
bestaande labels. Hoe je de box tekent (liggend, staand, hoek) maakt dus niet uit.

Via de server (makkelijkst):

    uv run python scripts/cvat_labels.py push --split test     # taak maken met foto's + labels
    uv run python scripts/cvat_labels.py pull                  # labels van de laatste taak terug
    uv run python scripts/cvat_labels.py push IMG_7001 IMG_7002 # nieuwe foto's (nog zonder label)

Via bestanden (CVAT for images 1.1, de standaard export van CVAT):

    uv run python scripts/cvat_labels.py export --out runs/cvat/labels.zip [namen...]
    uv run python scripts/cvat_labels.py import export.zip [--dry-run]
    uv run python scripts/cvat_labels.py labels                # labels voor de Raw-editor van CVAT

Inloggen: CVAT_URL (bv. https://app.cvat.ai of je eigen CVAT), CVAT_USER en CVAT_PASSWORD
(wat ontbreekt wordt gevraagd).
"""

import argparse
import getpass
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from common import COLORS, IGNORE, IMAGES, LABELS, ROOT, load_classes, load_split
from scipy.optimize import linear_sum_assignment

HEAD = "kop"  # puntlabel in CVAT voor de kop
HEADLESS = {"moer", "ring", "koppelbus"}  # hebben geen kop (README)
PROJECT = "CV-EIND onderdelen"


# ---------------------------------------------------------------- omrekenen


def r2(v: float) -> float:
    return round(float(v), 2)


def r3(v: float) -> float:  # hoeken: de bestaande labels hebben tot 3 decimalen
    return round(float(v), 3)


def norm_angle(a: float) -> float:
    """Hoek naar (-180, 180], het bereik van de bestaande labels."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def to_shapes(ann: dict) -> list[dict]:
    """Eigen label -> CVAT-shapes (zelfde velden als de CVAT REST API)."""
    shapes = []
    for i, o in enumerate(ann["objects"], start=1):
        (cx, cy), (w, h) = o["center"], o["size"]
        group = i if "head" in o else 0
        shapes.append({"type": "rectangle", "label": o["class"], "points": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                       "rotation": o["angle"] % 360.0, "occluded": bool(o.get("occluded")), "group": group})  # fmt: skip
        if "head" in o:
            shapes.append({"type": "points", "label": HEAD, "points": list(o["head"]), "rotation": 0.0,
                           "occluded": False, "group": group})  # fmt: skip
    return shapes


def _ends(o: dict) -> list[tuple]:
    """De 4 zijmiddens als (punt, hoek van dat uiteinde naar de overkant, lengte, breedte)."""
    (cx, cy), (w, h), a = o["center"], o["size"], o["angle"]
    out = []
    for da, length, width in ((0, w, h), (90, h, w), (180, w, h), (270, h, w)):
        t = math.radians(a + da)
        out.append(((cx - length / 2 * math.cos(t), cy - length / 2 * math.sin(t)), a + da, length, width))
    return out


def _head_end(o: dict, p) -> tuple[tuple, float, float]:
    """Welke zijde is de kop als je op p klikt? Geeft (zijde, diepte, buiten).

    diepte: 0 = op de zijde, 1 = in het midden van de box (relatief, dus gelijk voor kleine en
    grote onderdelen). Langwerpige boxen (verhouding > 2) hebben alleen hun twee korte zijden als
    kop; bij een bijna vierkante box telt de driehoek (tussen de diagonalen) waarin je klikt.
    buiten: hoeveel pixels p buiten de box ligt."""
    (cx, cy), (w, h) = o["center"], o["size"]
    long_only = max(w, h) > 2 * min(w, h)
    best = None
    for e in _ends(o):
        _, a, length, width = e
        if long_only and length < width:
            continue
        t = math.radians(a)
        proj = -((p[0] - cx) * math.cos(t) + (p[1] - cy) * math.sin(t))  # richting kop-uiteinde
        depth = 1.0 - proj / max(length / 2, 1e-6)
        if best is None or depth < best[1]:
            best = (e, depth)
    t = math.radians(o["angle"])
    dx, dy = p[0] - cx, p[1] - cy
    along, across = dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)
    outside = math.hypot(max(0.0, abs(along) - w / 2), max(0.0, abs(across) - h / 2))
    return best[0], max(best[1], 0.0), outside


NO_MATCH = 1.0  # kosten van "deze klik hoort bij geen box"


def _head_cost(o: dict, p) -> float:
    """Hoe goed p als kop bij deze box past (lager = beter). Een klik binnen een box, dicht bij
    een kop-zijde, is goedkoop; elke pixel erbuiten kost veel."""
    _, depth, outside = _head_end(o, p)
    if outside > max(15.0, 0.15 * max(o["size"])):
        return math.inf
    return depth / 2 + outside / 10.0


def orient(o: dict, p) -> None:
    """De kop-zijde bij p: size = [lengte, breedte] in die richting, hoek van kop naar punt,
    head precies op het midden van die zijde (zoals in de bestaande labels)."""
    (head, a, length, width), _, _ = _head_end(o, p)
    o["size"], o["angle"] = [r2(length), r2(width)], r3(norm_angle(a))
    if math.dist(p, head) < 0.05:  # kop ligt al op het uiteinde (onveranderd): laat hem zo
        head = p
    o["head"] = [r2(head[0]), r2(head[1])]


def from_shapes(image: str, width: int, height: int, shapes: list[dict], classes: list[str]):
    """CVAT-shapes -> eigen label. Geeft (label, waarschuwingen)."""
    warn: list[str] = []
    objects, groups, occluded = [], [], []
    for s in shapes:
        if s["type"] != "rectangle":
            continue
        if s["label"] not in classes and s["label"] != IGNORE:
            warn.append(f"onbekend label {s['label']!r} overgeslagen")
            continue
        x0, y0, x1, y1 = s["points"]
        o = {"class": s["label"], "shape": "obb", "center": [r2((x0 + x1) / 2), r2((y0 + y1) / 2)],
             "size": [r2(x1 - x0), r2(y1 - y0)], "angle": r3(norm_angle(s.get("rotation", 0.0)))}  # fmt: skip
        objects.append(o)
        groups.append(s.get("group") or 0)
        occluded.append(bool(s.get("occluded")))
    for s in shapes:
        if s["type"] not in ("rectangle", "points") or (s["type"] == "points" and s["label"] != HEAD):
            warn.append(f"{s['type']} {s['label']!r} overgeslagen (alleen boxen en '{HEAD}'-punten tellen)")

    # koppunten (één points-shape kan meerdere klikken bevatten)
    kops = [((s["points"][i], s["points"][i + 1]), s.get("group") or 0)
            for s in shapes if s["type"] == "points" and s["label"] == HEAD
            for i in range(0, len(s["points"]) - 1, 2)]  # fmt: skip
    can_have = [o["class"] not in HEADLESS and o["class"] != IGNORE for o in objects]
    taken: dict[int, tuple] = {}
    loose = []
    for p, g in kops:  # eerst gegroepeerd (zoals dit script exporteert)
        hit = [i for i, gi in enumerate(groups) if g and gi == g and can_have[i] and i not in taken]
        if hit:
            taken[hit[0]] = p
        else:
            loose.append(p)
    # losse klikken: de beste één-op-één toewijzing van klikken aan boxen (Hongaarse methode),
    # zodat een lange bout niet de kop van een klein onderdeel ernaast "steelt"
    free = [i for i in range(len(objects)) if can_have[i] and i not in taken]
    used = set()
    if loose and free:
        cost = np.full((len(loose), len(free) + len(loose)), NO_MATCH)
        for k, p in enumerate(loose):
            for j, i in enumerate(free):
                cost[k, j] = min(_head_cost(objects[i], p), 1e6)
        for k, j in zip(*linear_sum_assignment(cost)):
            if j < len(free) and cost[k, j] < NO_MATCH:
                i = free[j]
                taken[i] = loose[k]
                used.add(k)
                second = sorted(cost[k, : len(free)])[1] if len(free) > 1 else math.inf
                if second < cost[k, j] + 0.1:
                    p = loose[k]
                    warn.append(f"kop op ({p[0]:.0f}, {p[1]:.0f}) twijfel tussen boxen, gekozen: "
                                f"{objects[i]['class']} (check met show.py, of groepeer box+kop in CVAT met G)")  # fmt: skip
    for k, p in enumerate(loose):
        if k not in used:
            warn.append(f"kop op ({p[0]:.0f}, {p[1]:.0f}) hoort bij geen bout/koppeling, overgeslagen")

    for i, o in enumerate(objects):
        if i in taken:
            orient(o, taken[i])
        elif can_have[i]:
            warn.append(f"{o['class']} op ({o['center'][0]:.0f}, {o['center'][1]:.0f}) heeft geen kop")
    for o, occ in zip(objects, occluded):
        if occ:
            o["occluded"] = True  # na head, zoals in de bestaande labels
    return {"image": image, "width": int(width), "height": int(height), "objects": objects}, warn


# ---------------------------------------------------------------- CVAT for images 1.1 (XML)


def label_spec(classes: list[str]) -> list[dict]:
    def hexcolor(bgr):
        return "#{:02x}{:02x}{:02x}".format(*bgr[::-1])

    spec = [{"name": c, "type": "rectangle", "color": hexcolor(COLORS[i % len(COLORS)]), "attributes": []}
            for i, c in enumerate(classes)]  # fmt: skip
    spec.append({"name": IGNORE, "type": "rectangle", "color": "#000000", "attributes": []})
    spec.append({"name": HEAD, "type": "points", "color": "#ffff00", "attributes": []})
    return spec


def fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def write_xml(anns: list[dict], classes: list[str]) -> bytes:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    labels = ET.SubElement(ET.SubElement(ET.SubElement(root, "meta"), "task"), "labels")
    for spec in label_spec(classes):
        lab = ET.SubElement(labels, "label")
        for key in ("name", "color", "type"):
            ET.SubElement(lab, key).text = spec[key]
        ET.SubElement(lab, "attributes")
    for idx, ann in enumerate(anns):
        img = ET.SubElement(root, "image", id=str(idx), name=ann["image"], width=str(ann["width"]), height=str(ann["height"]))
        for s in to_shapes(ann):
            common = {"label": s["label"], "source": "manual", "occluded": str(int(s["occluded"])), "z_order": "0"}
            if s["group"]:
                common["group_id"] = str(s["group"])
            if s["type"] == "rectangle":
                x0, y0, x1, y1 = s["points"]
                ET.SubElement(img, "box", **common, xtl=fmt(x0), ytl=fmt(y0), xbr=fmt(x1), ybr=fmt(y1), rotation=fmt(s["rotation"]))
            else:
                ET.SubElement(img, "points", **common, points=f"{fmt(s['points'][0])},{fmt(s['points'][1])}")
    ET.indent(root)
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")


def read_xml(data: bytes):
    """XML -> [(naam, breedte, hoogte, shapes)]."""
    out = []
    for img in ET.fromstring(data).iter("image"):
        shapes = []
        for el in img:
            a = el.attrib
            base = {"label": a.get("label", ""), "occluded": a.get("occluded") == "1",
                    "group": int(a.get("group_id", 0) or 0), "rotation": float(a.get("rotation", 0) or 0)}  # fmt: skip
            if el.tag == "box":
                pts = [float(a[k]) for k in ("xtl", "ytl", "xbr", "ybr")]
                shapes.append({"type": "rectangle", "points": pts, **base})
            elif el.tag == "points":
                pts = [float(v) for xy in a["points"].split(";") for v in xy.split(",")]
                shapes.append({"type": "points", "points": pts, **base})
            else:
                shapes.append({"type": el.tag, "points": [], **base})
        out.append((Path(img.get("name")).name, int(img.get("width")), int(img.get("height")), shapes))
    return out


# ---------------------------------------------------------------- opslaan


def keep_order(new: list[dict], old: list[dict]) -> list[dict]:
    """Zet de objecten in de volgorde van het bestaande bestand (CVAT husselt ze na opslaan),
    zodat git diff alleen echte wijzigingen laat zien. Nieuwe objecten komen achteraan."""
    pairs = sorted((math.dist(n["center"], o["center"]), i, j) for i, n in enumerate(new)
                   for j, o in enumerate(old) if n["class"] == o["class"])  # fmt: skip
    rank, used = {}, set()
    for d, i, j in pairs:
        if d <= 30 and i not in rank and j not in used:
            rank[i] = j
            used.add(j)
    for i, j in rank.items():  # alleen afrondingsverschil (CVAT's XML heeft 2 decimalen): oude waarden
        if same_within(new[i], old[j]):
            new[i] = old[j]
    return [n for _, n in sorted(enumerate(new), key=lambda t: (rank.get(t[0], len(old) + t[0]),))]


def same_within(a: dict, b: dict, tol: float = 0.02) -> bool:
    if a.keys() != b.keys() or a["class"] != b["class"] or a.get("occluded") != b.get("occluded"):
        return False
    nums = [(x, y) for k in ("center", "size", "head") if k in a for x, y in zip(a[k], b[k])]
    da = abs((a["angle"] - b["angle"] + 180) % 360 - 180)
    return da <= tol and all(abs(x - y) <= tol for x, y in nums)


def save_labels(items, classes: list[str], dry_run: bool, images: dict[str, bytes] | None = None) -> None:
    """items: [(naam, breedte, hoogte, shapes)] -> data/labels/*.json, met een overzicht."""
    split = load_split()
    in_split = {s for stems in split.values() for s in stems}
    counts = {"nieuw": 0, "gewijzigd": 0, "gelijk": 0}
    for name, width, height, shapes in items:
        stem = Path(name).stem
        if not (IMAGES / name).exists():
            if images and name in images:
                if not dry_run:
                    (IMAGES / name).write_bytes(images[name])
                print(f"{stem}: foto toegevoegd aan data/images")
            else:
                print(f"{stem}: LET OP, foto staat niet in data/images")
        ann, warn = from_shapes(name, width, height, shapes, classes)
        path = LABELS / f"{stem}.json"
        old = path.read_text() if path.exists() else None
        if old is not None:
            ann["objects"] = keep_order(ann["objects"], json.loads(old)["objects"])
        text = json.dumps(ann, indent=1, ensure_ascii=False)
        state = "nieuw" if old is None else ("gelijk" if old == text else "gewijzigd")
        counts[state] += 1
        if state != "gelijk":
            print(f"{stem}: {state}, {len(ann['objects'])} objecten")
            if not dry_run:
                path.write_text(text)
        for w in warn:
            print(f"  {stem}: {w}")
        if stem not in in_split:
            print(f"  {stem}: staat nog niet in data/split.yaml")
    print(("(dry-run, niets geschreven) " if dry_run else "") + ", ".join(f"{v} {k}" for k, v in counts.items()))
    if counts["nieuw"] + counts["gewijzigd"] and not dry_run:
        print("Draai daarna scripts/prepare_data.py opnieuw.")


def load_anns(stems: list[str]) -> list[dict]:
    if not stems:
        stems = sorted(p.stem for p in LABELS.glob("*.json"))
    return [json.loads((LABELS / f"{s}.json").read_text()) for s in stems]


# ---------------------------------------------------------------- CVAT-server (REST API)


class Cvat:
    def __init__(self):
        import requests

        self.url = (os.environ.get("CVAT_URL") or input("CVAT-url (bv. https://app.cvat.ai): ")).rstrip("/")
        user = os.environ.get("CVAT_USER") or input("CVAT-gebruiker: ")
        password = os.environ.get("CVAT_PASSWORD") or getpass.getpass(f"CVAT-wachtwoord voor {user}: ")
        self.s = requests.Session()
        self.s.auth = (user, password)

    def call(self, method: str, path: str, **kw):
        r = self.s.request(method, f"{self.url}/api/{path}", **kw)
        if not r.ok:
            sys.exit(f"CVAT {method} {path}: {r.status_code} {r.text[:300]}")
        return r.json() if r.content else None

    def all(self, path: str, **params):
        params.setdefault("page_size", 100)
        data = self.call("GET", path, params=params)
        results = data["results"]
        while data.get("next"):
            data = self.s.get(data["next"]).json()
            results += data["results"]
        return results

    def wait(self, rq_id: str) -> None:
        while True:
            st = self.call("GET", f"requests/{rq_id}")
            if st["status"] == "finished":
                return
            if st["status"] == "failed":
                sys.exit(f"CVAT: {st.get('message')}")
            time.sleep(2)

    def project(self, classes: list[str]) -> int:
        for p in self.all("projects", search=PROJECT):
            if p["name"] == PROJECT:
                have = {lab["name"] for lab in self.all("labels", project_id=p["id"])}
                missing = [s for s in label_spec(classes) if s["name"] not in have]
                if missing:
                    self.call("PATCH", f"projects/{p['id']}", json={"labels": missing})
                return p["id"]
        p = self.call("POST", "projects", json={"name": PROJECT, "labels": label_spec(classes)})
        print(f"project {PROJECT!r} aangemaakt")
        return p["id"]

    def labels(self, project_id: int) -> dict[str, int]:
        return {lab["name"]: lab["id"] for lab in self.all("labels", project_id=project_id)}

    def frames(self, task_id: int) -> list[dict]:
        return self.call("GET", f"tasks/{task_id}/data/meta")["frames"]


def push(args, classes: list[str]) -> None:
    stems = list(args.names)
    if args.split:
        stems += load_split()[args.split]
    if not stems:
        sys.exit("Geef foto-namen en/of --split train|val|test.")
    files = []
    for stem in stems:
        hits = list(IMAGES.glob(f"{stem}.*"))
        if not hits:
            sys.exit(f"Geen foto voor {stem!r} in data/images")
        files += hits
    files = sorted(set(files))
    cv = Cvat()
    pid = cv.project(classes)
    name = args.name or f"{args.split or 'fotos'} {datetime.now():%Y-%m-%d %H:%M}"
    task = cv.call("POST", "tasks", json={"name": name, "project_id": pid})
    print(f"taak {task['id']} {name!r}: {len(files)} foto's uploaden...")
    upload = [(f"client_files[{i}]", (p.name, p.open("rb"), "image/jpeg")) for i, p in enumerate(files)]
    rq = cv.call("POST", f"tasks/{task['id']}/data", files=upload,
                 data={"image_quality": 95, "sorting_method": "lexicographical"})  # fmt: skip
    cv.wait(rq["rq_id"])
    ids = cv.labels(pid)
    shapes = []
    for frame, meta in enumerate(cv.frames(task["id"])):
        path = LABELS / f"{Path(meta['name']).stem}.json"
        if not path.exists():
            continue
        for s in to_shapes(json.loads(path.read_text())):
            shapes.append({"type": s["type"], "frame": frame, "label_id": ids[s["label"]], "points": s["points"],
                           "rotation": s["rotation"], "occluded": s["occluded"], "group": s["group"],
                           "z_order": 0, "source": "manual", "attributes": []})  # fmt: skip
    cv.call("PUT", f"tasks/{task['id']}/annotations", json={"version": 0, "tags": [], "shapes": shapes, "tracks": []})
    print(f"klaar: {len(shapes)} shapes. Open {cv.url}/tasks/{task['id']}")


def pull(args, classes: list[str]) -> None:
    cv = Cvat()
    pid = cv.project(classes)
    tasks = cv.all("tasks", project_id=pid)
    if not tasks:
        sys.exit("Geen taken in het CVAT-project.")
    if args.task:
        match = [t for t in tasks if str(t["id"]) == args.task or t["name"] == args.task]
        if not match:
            sys.exit(f"Taak {args.task!r} niet gevonden. Taken: " + ", ".join(f"{t['id']} {t['name']!r}" for t in tasks))
        task = match[0]
    else:
        task = max(tasks, key=lambda t: t["id"])
    print(f"taak {task['id']} {task['name']!r}")
    names = {i: n for n, i in cv.labels(pid).items()}
    frames = cv.frames(task["id"])
    data = cv.call("GET", f"tasks/{task['id']}/annotations")
    if data.get("tracks"):
        print(f"LET OP: {len(data['tracks'])} tracks overgeslagen (gebruik gewone shapes, geen tracks)")
    per_frame: dict[int, list] = {i: [] for i in range(len(frames))}
    for s in data["shapes"]:
        per_frame[s["frame"]].append({"type": s["type"], "label": names[s["label_id"]], "points": s["points"],
                                      "rotation": s.get("rotation", 0.0), "occluded": s["occluded"],
                                      "group": s.get("group") or 0})  # fmt: skip
    items = [(Path(m["name"]).name, m["width"], m["height"], per_frame[i]) for i, m in enumerate(frames)]
    save_labels(items, classes, args.dry_run)


# ---------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="labels -> CVAT for images 1.1 (.zip of .xml)")
    p.add_argument("names", nargs="*", help="foto-namen (standaard alle)")
    p.add_argument("--out", type=Path, default=ROOT / "runs/cvat/labels.zip")
    p = sub.add_parser("import", help="CVAT-export (.zip of .xml) -> data/labels")
    p.add_argument("file", type=Path)
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("labels", help="labelspecificatie voor de Raw-editor van CVAT")
    p = sub.add_parser("push", help="taak maken in CVAT met foto's en labels")
    p.add_argument("names", nargs="*", help="foto-namen")
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--name", help="naam van de taak")
    p = sub.add_parser("pull", help="labels van een CVAT-taak terugzetten")
    p.add_argument("task", nargs="?", help="taak-id of -naam (standaard de nieuwste)")
    p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    classes = load_classes()

    if args.cmd == "labels":
        print(json.dumps(label_spec(classes), indent=2))
    elif args.cmd == "export":
        anns = load_anns(args.names)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        xml = write_xml(anns, classes)
        if args.out.suffix == ".xml":
            args.out.write_bytes(xml)
        else:
            with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("annotations.xml", xml)
        print(f"{len(anns)} foto's -> {args.out}")
    elif args.cmd == "import":
        images = None
        if args.file.suffix == ".zip":
            with zipfile.ZipFile(args.file) as z:
                xml = z.read("annotations.xml")
                images = {Path(n).name: z.read(n) for n in z.namelist() if n.startswith("images/") and not n.endswith("/")}
        else:
            xml = args.file.read_bytes()
        save_labels(read_xml(xml), classes, args.dry_run, images)
    elif args.cmd == "push":
        push(args, classes)
    elif args.cmd == "pull":
        pull(args, classes)


if __name__ == "__main__":
    main()
