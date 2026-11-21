"""Train YOLO-OBB. De voorgetrainde gewichten worden de eerste keer gedownload.

uv run python scripts/train_yolo.py --name eerste

Voor de experimenten zijn er ook --seed, --patience en --geen-aug (zie --help). Elke andere
instelling van Ultralytics kan met --set, bijvoorbeeld --set lr0=0.005 --set mosaic=0.
"""

import argparse

import yaml
from common import ROOT
from ultralytics import YOLO

# De camera kijkt van boven: elke draaiing en spiegeling is realistisch.
# De belichting wisselt sterk.
AUGMENTATION = dict(
    degrees=180,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.5,
    scale=0.5,
    translate=0.1,
    mosaic=1.0,
    mixup=0.0,
)
NO_AUGMENTATION = {key: 0 for key in AUGMENTATION}


def setting(text: str) -> tuple[str, object]:
    """Maak van 'naam=waarde' een paar; de waarde wordt een getal, True/False of tekst."""
    key, value = text.split("=", 1)
    return key, yaml.safe_load(value)


def main() -> None:
    """Train YOLO-OBB met de instellingen uit de argumenten."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo")
    ap.add_argument(
        "--model", default="yolo11m-obb.pt", help="ook: yolo11n-obb.pt, yolo11s-obb.pt, ..."
    )
    ap.add_argument("--data", default=str(ROOT / "data/yolo/data.yaml"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--patience", type=int, default=60, help="stop na zoveel epochs zonder winst (0 = nooit)"
    )
    ap.add_argument("--geen-aug", action="store_true", help="zonder augmentatie")
    ap.add_argument("--set", type=setting, action="append", default=[], metavar="NAAM=WAARDE",
                    help="extra Ultralytics-instelling (mag vaker)")  # fmt: skip
    args = ap.parse_args()
    model = YOLO(args.model)
    augmentation = NO_AUGMENTATION if args.geen_aug else AUGMENTATION
    augmentation = {**augmentation, **dict(args.set)}
    patience = args.patience or args.epochs  # Ultralytics kent geen 0, dus: nooit eerder stoppen
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=str(ROOT / "runs/yolo"), name=args.name, exist_ok=True, patience=patience,
                workers=4, seed=args.seed, **augmentation)  # fmt: skip
    print("Model:", ROOT / "runs/yolo" / args.name / "weights/best.pt")


if __name__ == "__main__":
    main()
