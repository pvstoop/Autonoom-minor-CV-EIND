# CV-EIND: onderdelen detecteren met YOLO-OBB

Zoekt 8 soorten onderdelen in een foto: waar ligt het (geroteerde box), hoe ligt het (hoek, en bij bouten aan welke
kant de kop zit), wat is het en welke maat (bijvoorbeeld kleine of grote ring). Met YOLO11-OBB.

Wat ik heb geprobeerd en wat eruit kwam staat in [docs/experimenten.md](docs/experimenten.md).
Welke foto's er zijn en hoe ze verdeeld zijn staat in [docs/dataset.md](docs/dataset.md).

[uv](https://docs.astral.sh/uv/) is nodig; die installeert zelf Python 3.14 en de pakketten.

Het getrainde model staat in `models/` (via Git LFS: eerst `git lfs install` en `git lfs pull`). Testen:

```bash
uv sync
uv run python scripts/prepare_data.py                               # labels -> data/yolo
uv run python scripts/evaluate.py models/yolo11m_obb.pt            # scores + tekeningen in runs/pred/
```

Zelf trainen:

```bash
uv run python scripts/train_yolo.py --name m --model yolo11m-obb.pt
uv run python scripts/evaluate.py runs/yolo/m/weights/best.pt
```

| Pad | Inhoud |
|---|---|
| `data/images/` | 75 foto's: 19 op een egale ondergrond (`ps_...`), 56 op een dambord (`board_...`) |
| `data/labels/` | Eén JSON-bestand per foto |
| `data/split.yaml` | Welke foto in train, val of test zit: 31, 7 en 37 |
| `data/maten.yaml` | Welke klassen 2 maten hebben en waar de grens ligt (gemaakt door `sizes.py`) |
| `models/` | Het getrainde model |
| `classes.txt` | De 8 klassen |
| `scripts/` | Data klaarzetten, trainen, testen, maten zoeken, labels tekenen en de grafieken maken |
| `results/` | De cijfers achter de grafieken in docs/experimenten.md |

## Hoe het werkt

- `prepare_data.py` zet de labels om naar YOLO. Bij elke bout en koppeling komt een kleine box `kop` op de kop, als 9e klasse.
- `train_yolo.py` traint YOLO11-OBB, met draaien, spiegelen, schalen, helderheid en kleur als augmentatie.
- `sizes.py` zoekt welke klassen 2 maten hebben: de lengte gedeeld door de moer op dezelfde foto, in 2 groepen
  verdeeld. De grenzen komen in `data/maten.yaml`.
- `evaluate.py` voorspelt de testfoto's (met `--val` de val-foto's), koppelt elke gevonden kop aan het dichtstbijzijnde
  uiteinde van een bout of koppeling, zet de maat erbij, en print mAP50, F1, de hoekfout en hoe vaak de kop en de maat goed zijn, apart voor de egale ondergrond en het
  dambord. Kies instellingen op val, en gebruik test alleen voor het eindcijfer.

## Labels

```json
{"class": "zeskantbout", "shape": "obb", "center": [751.0, 322.0], "size": [203.7, 42.0], "angle": 13.6, "head": [652.0, 298.0]}
```

- center, size en angle vormen de geroteerde box, in pixels van de foto (1536 x 2048).
- head is de kop van het onderdeel. Bij head is size = [lengte, breedte] en wijst angle van de kop naar de punt.
  Moeren, ringen en koppelbussen hebben geen kop.
- occluded: true betekent dat het onderdeel niet helemaal zichtbaar is.
- _negeer is geen klasse maar een gebied dat niet meetelt, bijvoorbeeld een onderdeel dat te ver bedekt is om te herkennen.

Bekijk labels met `uv run python scripts/show.py board_IMG_6241` (tekening in runs/labels/).

## Foto's toevoegen

1. Zet de foto in data/images/ en het label in data/labels/, met dezelfde naam.
2. Zet de naam in data/split.yaml. Foto's van dezelfde opstelling horen in dezelfde groep.
3. Draai prepare_data.py opnieuw en train opnieuw.

## Resultaat (testset, gemiddeld over 3 keer trainen)

| | mAP50 | F1 | Kop goed |
|---|---|---|---|
| Egale ondergrond | 94.6 ± 1.5 | 95.7 | 96% |
| Dambord, nieuwe opstellingen | 86.8 ± 0.7 | 89.2 | 95% |

Het model in `models/` is 1 van die 3 keer: 96.3 op de egale ondergrond en 86.1 op het dambord. De maat is daarbij
100% goed op de egale ondergrond en 92% op het dambord.




## Klassieke CV (`scripts/classical_pipeline.py`)

OpenCV pakt de onderdelen eruit (adaptive threshold op egale achtergrond; HSV, grijswaarden en een ROI-masker op het dambord) en een Random Forest kijkt naar 11 kenmerken zoals vorm, hoekpunten, vulling en richting van de boutkop.

| Variant | mAP@50 | F1 | F1 egaal | F1 dambord | Boutkop | Tijd |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest (11 kenmerken / close 5x5 / containment) | 6.3% | 6.1% | 35.0% | 1.3% | 45.5% (5/11) | 0.39s |
| SVM (RBF-kernel op 11 kenmerken) | 2.3% | 6.4% | 33.5% | 1.8% | 38.5% (5/13) | 0.12s |
| Random Forest zonder `approxPolyDP` hoekpunten | 6.2% | 6.4% | 35.4% | 1.4% | 50.0% (6/12) | 0.39s |
| Random Forest zonder helderheid (`brightness`) | 6.1% | 6.3% | 34.6% | 1.6% | 41.7% (5/12) | 0.41s |
| Zonder filter op binnengaten (containment) | 6.2% | 6.1% | 33.5% | 1.3% | 45.5% (5/11) | 0.38s |
| Morfologie `MORPH_CLOSE` 3x3 i.p.v. 5x5 | 4.4% | 6.0% | 31.8% | 2.0% | 28.6% (4/14) | 0.38s |

- `results/klassiek/varianten_vergelijking.csv`: alle varianten naast elkaar.
- `results/klassiek/kernresultaten.csv`: korte samenvatting op de testset.
- `results/klassiek/voorspellingen.csv`: losse detecties, scores en hoeken.
- `results/klassiek/voorbeelden/1e_editie/` t/m `5e_editie/`: testfoto's per stap (`5e_editie` heeft NMS op gedraaide boxen tegen dubbele boxen op glans).
