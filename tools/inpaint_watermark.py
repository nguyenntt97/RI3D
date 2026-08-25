"""Write a watermark-free copy of a scene's photographs for training.

Stage `wmi`. Runs after SfM, over `<sfm_dir>/images/`, and writes
`<sfm_dir>/images_wmclean/` which the scene reader prefers automatically.

Why a second image set rather than masking harder: the `wm` mask makes those
pixels *unsupervised*, so nothing constrains the Gaussians behind the overlay and
novel views render garbage there. Replacing the overlay with plausible content and
supervising it at reduced weight (`--wm_loss_weight`) gives that region something
to converge to, while keeping honest photographic evidence worth more than
invented evidence.

Outputs are PNG regardless of the input format, so the ~88% of each frame the mask
never touches stays bit-exact rather than picking up JPEG re-encoding artifacts.

Usage:
    python tools/inpaint_watermark.py -s output/sceneC/ggpt_sfm
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from utils.watermark_utils import clean_watermark, find_watermark_mask
from utils import wandb_utils as wb

CLEAN_DIRNAME = "images_wmclean"


def main():
    ap = argparse.ArgumentParser(description="Paint watermarks out of a scene's photographs")
    ap.add_argument("-s", "--source_path", required=True,
                    help="SfM directory containing images/, e.g. output/sceneC/ggpt_sfm")
    ap.add_argument("-o", "--output_dir", default=None,
                    help=f"Defaults to <source_path>/{CLEAN_DIRNAME}")
    ap.add_argument("--mask", "--watermark_mask", dest="mask", default=None,
                    help="Override the discovered watermark mask")
    ap.add_argument("--prompt", default="a photo of an empty interior")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--min_side", type=int, default=448)
    ap.add_argument("--force", action="store_true",
                    help="Re-inpaint images that already have a cleaned copy")
    args = ap.parse_args()
    wb.attach()
    wb.define_axis("stage_wmi", "image_idx")

    image_dir = os.path.join(args.source_path, "images")
    if not os.path.isdir(image_dir):
        print(f"[x] {image_dir} does not exist")
        sys.exit(1)

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(set(paths))
    if not paths:
        print(f"[x] No images in {image_dir}")
        sys.exit(1)

    probe = np.asarray(Image.open(paths[0]).convert("RGB"))
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        mask = None if m is None else m > 127
    else:
        mask = find_watermark_mask(args.source_path, probe.shape[0], probe.shape[1])
    if mask is None:
        # Not an error: most scenes have no watermark, and the reader simply keeps
        # using images/ when no cleaned directory appears.
        print("[i] No watermark mask for this scene; nothing to inpaint.")
        wb.log_summary({"stage_wmi/mask_coverage": 0.0, "stage_wmi/n_total": len(paths),
                        "stage_wmi/n_inpainted": 0, "stage_wmi/n_skipped": len(paths)})
        wb.finish()
        return

    out_dir = args.output_dir or os.path.join(args.source_path, CLEAN_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    pipe = None
    n_done = 0
    n_skipped = 0
    started = time.time()
    for i, path in enumerate(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(out_dir, f"{stem}.png")
        if os.path.exists(dst) and not args.force:
            print(f"[i] {stem}: already cleaned")
            n_skipped += 1
            continue

        img = np.asarray(Image.open(path).convert("RGB"))
        if img.shape[:2] != mask.shape[:2]:
            print(f"[!] {stem}: {img.shape[1]}x{img.shape[0]} does not match the "
                  f"{mask.shape[1]}x{mask.shape[0]} mask; copying through unchanged")
            Image.fromarray(img).save(dst)
            n_skipped += 1
            continue

        if pipe is None:
            from utils.realfill_utils import InPaint
            print("[i] Loading SD2 inpainting ...")
            pipe = InPaint("")

        t = time.time()
        out = clean_watermark(img, mask, pipe, prompt=args.prompt, steps=args.steps,
                              min_side=args.min_side, guidance_scale=args.guidance,
                              verbose=False)
        # The composite must leave the rest of the photograph untouched; if this
        # ever trips, the fill is damaging real evidence rather than replacing the
        # overlay.
        assert np.array_equal(img[~mask], out[~mask]), f"{stem}: pixels outside the mask changed"
        Image.fromarray(out).save(dst)
        n_done += 1
        dt = time.time() - t
        wb.log({"stage_wmi/image_idx": i, "stage_wmi/seconds": dt})
        print(f"[i] {stem}: {dt:.1f}s")

    wb.log_summary({"stage_wmi/n_inpainted": n_done, "stage_wmi/n_skipped": n_skipped,
                    "stage_wmi/n_total": len(paths),
                    "stage_wmi/mask_coverage": float(mask.mean()),
                    "stage_wmi/total_seconds": time.time() - started})
    wb.finish()
    print(f"[i] {n_done} image(s) inpainted into {out_dir}")
    print("[i] The scene reader picks these up automatically. Supervise them at reduced "
          "weight with --wm_loss_weight (they are plausible, not photographic).")


if __name__ == "__main__":
    main()
