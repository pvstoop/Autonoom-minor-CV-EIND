# CV-NN: onderdelen detecteren met een neuraal netwerk

Zoeken 8 classes in images.

[uv](https://docs.astral.sh/uv/) is nodig

```bash
uv sync
uv run python scripts/prepare_data.py

uv run python scripts/train_yolo.py --name eerste
uv run python scripts/evaluate.py runs/yolo/eerste/weights/best.pt

uv run python scripts/train_rfdetr.py --name eerste
uv run python scripts/evaluate.py runs/rfdetr/eerste/checkpoint_final.pth
```

| Path | Inhoud |
|---|---|
| `data/images/` | 75 foto's: 19 op een egale ondergrond (`ps_...`), 56 op een dambord (`board_...`) |
| `data/labels/` | Eén JSON-bestand per foto |
| `data/split.yaml` | Welke foto in train, val of test zit: 36, 2 en 37 |
| `classes.txt` | De 8 klassen |
| `scripts/` | Data klaarzetten, trainen, testen en labels tekenen |
| `third_party/rf-detr/` | RF-DETR met onze OBB-kop; hier hoef je meestal niets aan te doen |

Labels

```json
{"class": "zeskantbout", "shape": "obb", "center": [751.0, 322.0], "size": [203.7, 42.0], "angle": 13.6, "head": [652.0, 298.0]}
```

- center, size en angle vormen de geroteerde box, in pixels van de foto (1536 x 2048).
- head is de kop van het onderdeel. Bij head is size = [lengte, breedte] en wijst angle van de kop naar de punt.
  Moeren, ringen en koppelbussen hebben geen kop.
- occluded: true betekent dat het onderdeel niet helemaal zichtbaar is.
- _negeer is geen klasse maar een gebied dat niet meetelt, bijvoorbeeld een onderdeel dat te ver bedekt is om te herkennen.


ZIe labels met: uv run python scripts/show.py board_IMG_6241 (voorbeeld in runs/labels/).

Foto's toevoegen

1. Zet de foto in data/images/ en het label in data/labels/, met dezelfde naam.
2. Zet de naam in data/split.yaml. Foto's van dezelfde opstelling horen in dezelfde groep.
3. Draai prepare_data.py opnieuw en train opnieuw.

Resultaten tot nu toe (mAP50)

| Model | Getraind op | Eigen foto's | Dambord, nieuwe opstellingen |
|---|---|---|---|
| YOLO11m | eigen | 91.4 | 47.3 ± 4.4 |
| YOLO11m | eigen + dambord | 88.6 | 86.6 ± 3.1 |
| RF-DETR OBB | eigen | 96.0 | 76.9 ± 1.8 |
| RF-DETR OBB | eigen + dambord | 94.1 | 94.4 ± 0.9 |

Gemeten in CV-EIND, gemiddeld over 4 folds.

