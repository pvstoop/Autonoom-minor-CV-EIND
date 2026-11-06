"""Train RF-DETR met OBB-kop. De voorgetrainde gewichten worden de eerste keer gedownload.

uv run python scripts/train_rfdetr.py --name eerste
"""

import argparse
from pathlib import Path

import rfdetr
import torch
from common import ROOT

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}
AUGMENTATION = {
    "D4": {"p": 1.0},  # alle kwartslagen en spiegelingen
    "RandomBrightnessContrast": {"brightness_limit": 0.3, "contrast_limit": 0.3, "p": 0.7},
    "HueSaturationValue": {"hue_shift_limit": 5, "sat_shift_limit": 30, "val_shift_limit": 30, "p": 0.5},
}


def write_final_checkpoint(out: Path) -> Path:
    """We houden het model na de laatste epoch: met maar 2 valfoto's is 'beste epoch' kiezen vooral ruis."""
    best = torch.load(out / "checkpoint_best_total.pth", map_location="cpu", weights_only=False)
    last = torch.load(out / "last_ema.pth", map_location="cpu", weights_only=False)
    final = out / "checkpoint_final.pth"
    torch.save(best | {"model": last["model"], "epoch": last.get("epoch")}, final)
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rfdetr")
    ap.add_argument("--size", choices=SIZES, default="small")
    ap.add_argument("--data", default=str(ROOT / "data/coco"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--resolution", type=int, default=768, help="moet een veelvoud van 32 zijn")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradiënten optellen: effectief batch x accum")
    args = ap.parse_args()
    out = ROOT / "runs/rfdetr" / args.name
    model = getattr(rfdetr, SIZES[args.size])(resolution=args.resolution, obb_head=True)
    model.train(dataset_dir=args.data, epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.accum,
                output_dir=str(out), aug_config=AUGMENTATION, run_test=True, early_stopping=False,
                num_workers=4, tensorboard=False, checkpoint_interval=50)  # fmt: skip
    print("Model:", write_final_checkpoint(out))


if __name__ == "__main__":
    main()
