"""Train YOLO-OBB. De voorgetrainde gewichten worden de eerste keer gedownload.

uv run python scripts/train_yolo.py --name eerste
"""

import argparse

from common import ROOT
from ultralytics import YOLO

# De camera kijkt van boven: elke draaiing en spiegeling is realistisch, de belichting wisselt sterk
AUGMENTATION = dict(degrees=180, flipud=0.5, fliplr=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.5,
                    scale=0.5, translate=0.1, mosaic=1.0, mixup=0.0)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo")
    ap.add_argument("--model", default="yolo11m-obb.pt", help="ook: yolo11n-obb.pt, yolo11s-obb.pt, ...")
    ap.add_argument("--data", default=str(ROOT / "data/yolo/data.yaml"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=str(ROOT / "runs/yolo"), name=args.name, exist_ok=True, patience=60,
                workers=4, seed=0, **AUGMENTATION)  # fmt: skip
    print("Model:", ROOT / "runs/yolo" / args.name / "weights/best.pt")


if __name__ == "__main__":
    main()
