"""Locate a scene's watermark mask and paint the watermark out of a photograph.

Why this exists: the pipeline masks watermark pixels out of its *losses* but never
out of the images themselves. `Camera.__init__` is deliberate about that -- a
`loss_mask` leaves those pixels unsupervised rather than repainting them with a
background colour the losses would dutifully reproduce.

That is right for training a 3DGS model and wrong for building diffusion training
pairs. `leave_one_out_stage1.py:152` saves `gt.png` straight from
`infer_cam.original_image`, so any consumer pairing a render against it is teaching
a generative model to reproduce the watermark at fixed pixel coordinates. Stage 1c
consumes those pairs; `utils/dataset_lora.py:214` does the same for the stage-3
LoRA.

`clean_watermark` produces a target with plausible scene content where the overlay
was, using the SD2 inpainting checkpoint the project already ships for stage 4.
"""

import os

import cv2
import numpy as np
from PIL import Image


CLEAN_DIRNAME = "images_wmclean"

_announced = set()


def resolve_image_path(image_path):
    """Prefer a watermark-free copy of `image_path` when one exists.

    tools/inpaint_watermark.py writes `<sfm_dir>/images_wmclean/<stem>.png`
    alongside `images/`. Swapping here rather than at each call site means every
    consumer that goes through readMipTransforms -- stages 1a, 1b, 2a, 2b, the
    stage-3 LoRA dataset and the stage-5 data module -- picks the cleaned frames
    up without further plumbing.

    The stem is preserved, so `image_name` and therefore the `depth_rel/` and
    `train_test_split_*.json` lookups keyed off it are unaffected.
    """
    img_dir = os.path.dirname(image_path)
    sfm_dir = os.path.dirname(img_dir)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for cand in (
        os.path.join(sfm_dir, CLEAN_DIRNAME, stem + ".png"),
        os.path.join(sfm_dir, CLEAN_DIRNAME, os.path.basename(image_path)),
    ):
        if os.path.exists(cand):
            d = os.path.dirname(cand)
            if d not in _announced:
                _announced.add(d)
                print(f"[i] Using watermark-free photographs from {d}")
            return cand
    return image_path


# Ordered by specificity. Mirrors threestudio/data/loo_mip.py:_load_watermark_mask,
# which resolves the same file from a camera rather than from a scene directory.
def _mask_candidates(scene_path):
    scene_path = os.path.abspath(scene_path)
    parent = os.path.dirname(scene_path)
    return [
        os.path.join(scene_path, "masks", "watermark_mask.png"),
        os.path.join(scene_path, "images", "masks", "watermark_mask.png"),
        # inference.py's `wm` stage writes one level above <backend>_sfm/
        os.path.join(parent, "watermark_mask.png"),
        os.path.join(parent, "masks", "watermark_mask.png"),
        os.path.join(scene_path, "watermark_mask.png"),
    ]


def find_watermark_mask(scene_path, height=None, width=None, verbose=True):
    """Return a boolean (H, W) array, True where a watermark covers the photo.

    Returns None when the scene carries no mask, which is the common case --
    callers should treat that as "nothing to do", not as an error.

    A mask whose resolution does not match the images is skipped rather than
    aborting: the SfM stage also writes a pointmap-resolution copy alongside the
    full-resolution one, and a stale small mask must not shadow a correct big one
    later in the chain.
    """
    seen = set()
    for path in _mask_candidates(scene_path):
        path = os.path.normpath(path)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if height is not None and width is not None and m.shape[:2] != (height, width):
            if verbose:
                print(f"[!] {path} is {m.shape[1]}x{m.shape[0]} but images are "
                      f"{width}x{height}; skipping it.")
            continue
        mask = m > 127
        if verbose:
            print(f"[i] Watermark mask {path}: covers {mask.mean() * 100:.2f}% of the frame")
        return mask
    return None


def _square_box(x, y, w, h, pad, min_side, img_w, img_h):
    """A padded square around a blob, clamped to the image.

    Square because the inpainting UNet runs on a square latent; padding because a
    diffusion model needs surrounding context to invent something consistent, and
    a box cropped tight to the logo gives it almost none.
    """
    side = max(w, h) + 2 * pad
    side = min(max(side, min_side), min(img_w, img_h))
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))
    return x0, y0, int(side)


# Steers SD2 away from the two things that go wrong here: re-drawing the overlay
# it was asked to remove, and inventing an object where a plain wall belongs.
WATERMARK_NEGATIVE = (
    "text, letters, logo, watermark, sign, person, face, blur, lowres, "
    "cropped, worst quality"
)


def clean_watermark(image, mask, inpainter, prompt="a photo of an empty interior",
                    steps=50, pad=64, min_side=448, proc_res=512, guidance_scale=7.5,
                    verbose=True):
    """Inpaint every masked blob out of `image`.

    Args:
        image: (H, W, 3) uint8 RGB.
        mask: (H, W) bool, True = watermark.
        inpainter: a utils.realfill_utils.InPaint instance.
        prompt: InPaint's own default is the rare token `xxy5syt00`, meaningless
            to a checkpoint never fine-tuned on it. With classifier-free guidance
            on, the prompt genuinely steers the fill, so describe the scene.
        guidance_scale: above 1 so the prompt and WATERMARK_NEGATIVE actually
            apply. InPaint defaults to 1, which disables classifier-free guidance
            and lets the model free-run -- measured on sceneC that filled a logo
            on a blank ceiling with an invented object rather than more ceiling.
        min_side: a floor on the crop size. Blobs sitting on featureless surfaces
            give the model nothing to continue, so err toward more context.
        proc_res: SD2-inpainting's native resolution. Blobs are small, so a padded
            crop scaled *up* to 512 carries more detail than the source, whereas
            running the whole 1024x768 frame through would both blur it and let
            the model rewrite regions the mask never selected.

    Returns a new (H, W, 3) uint8 array. Pixels outside the mask are copied
    verbatim -- the diffusion output is composited in only where the mask says so,
    so a failed inpaint can never damage the rest of the photograph.
    """
    h, w = mask.shape[:2]
    out = image.copy()

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return out

    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        x0, y0, side = _square_box(x, y, bw, bh, pad, min_side, w, h)

        crop = out[y0:y0 + side, x0:x0 + side]
        # Only this blob: overlapping padded boxes would otherwise re-inpaint a
        # neighbour that has already been handled.
        crop_mask = (labels[y0:y0 + side, x0:x0 + side] == i)
        if not crop_mask.any():
            continue

        crop_r = cv2.resize(crop, (proc_res, proc_res), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(crop_mask.astype(np.uint8) * 255, (proc_res, proc_res),
                            interpolation=cv2.INTER_NEAREST)

        result = inpainter.inpaint(
            Image.fromarray(crop_r), Image.fromarray(mask_r),
            prompt=prompt, strength=1.0, num_inference_steps=steps,
            guidance_scale=guidance_scale, negative_prompt=WATERMARK_NEGATIVE,
        )
        result = cv2.resize(np.asarray(result)[..., :3], (side, side),
                            interpolation=cv2.INTER_CUBIC)

        crop[crop_mask] = result[crop_mask]
        if verbose:
            print(f"    blob {i}/{n_labels - 1}: {area} px at ({x},{y}) "
                  f"via {side}x{side} crop")

    return out


def _main():
    """Standalone tuning loop: clean a few photographs and write comparisons.

    The settings that matter here (prompt, guidance, context size) are scene
    dependent and only judgeable by eye, so being able to try them on one image
    beats discovering a bad fill after a full fine-tune.
    """
    import argparse
    import glob
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.realfill_utils import InPaint

    ap = argparse.ArgumentParser(description="Paint watermarks out of a scene's photographs")
    ap.add_argument("-s", "--scene_path", required=True,
                    help="SfM directory, e.g. output/sceneC/ggpt_sfm")
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("-i", "--image", default=None,
                    help="Clean just this file instead of sampling from <scene>/images")
    ap.add_argument("-n", "--limit", type=int, default=2,
                    help="How many photographs to clean when --image is not given")
    ap.add_argument("--mask", default=None, help="Override the discovered mask")
    ap.add_argument("--prompt", default="a photo of an empty interior")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--min_side", type=int, default=448)
    args = ap.parse_args()

    if args.image:
        paths = [args.image]
    else:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
            paths.extend(glob.glob(os.path.join(args.scene_path, "images", ext)))
        paths = sorted(paths)[:args.limit]
    if not paths:
        print(f"[x] No images found under {args.scene_path}/images")
        sys.exit(1)

    probe = np.asarray(Image.open(paths[0]).convert("RGB"))
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        mask = None if m is None else m > 127
    else:
        mask = find_watermark_mask(args.scene_path, probe.shape[0], probe.shape[1])
    if mask is None:
        print(f"[x] No watermark mask for {args.scene_path}; nothing to clean.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = InPaint("")

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img = np.asarray(Image.open(path).convert("RGB"))
        if img.shape[:2] != mask.shape[:2]:
            print(f"[!] {name}: {img.shape[1]}x{img.shape[0]} does not match the mask; skipping")
            continue

        t = time.time()
        out = clean_watermark(img, mask, pipe, prompt=args.prompt, steps=args.steps,
                              min_side=args.min_side, guidance_scale=args.guidance)
        # The composite must leave everything outside the mask untouched -- if this
        # ever fails, the fill is damaging the photograph rather than repairing it.
        assert np.array_equal(img[~mask], out[~mask]), \
            f"{name}: pixels outside the mask changed"

        Image.fromarray(out).save(os.path.join(args.output_dir, f"{name}_cleaned.png"))
        stack = np.concatenate([img, np.full((8, img.shape[1], 3), 255, np.uint8), out])
        Image.fromarray(stack).save(os.path.join(args.output_dir, f"{name}_compare.png"))
        print(f"[i] {name}: {time.time() - t:.1f}s, outside-mask pixels untouched")

    print(f"[i] Look at {args.output_dir}/*_compare.png (original above, cleaned below)")


if __name__ == "__main__":
    _main()
