"""Export an RI3D scene into the directory layout GSFixer expects.

Stage 1c step 1. Produces three things from a trained stage-1b model:

  <out>/finetune/{rgb,rendered_gs}/    scene-specific fine-tuning pairs
  <out>/rendered_novel_views/gs_image_<tag>/   orbit renders to repair
  <out>/novel_views.json              the poses those renders came from

The layout is exactly what GSFix3D reads, so `scripts/gsfixer/train.py` and
`scripts/gsfixer/inference.py` run against it unmodified -- their `--data_type`
switch is only consulted under `--eval`, which we never pass.

Usage:
    python tools/gsfix_export.py \
        -s output/sceneC/ggpt_sfm -m output/gs_init/sceneC_6 \
        --loo_dir output/gs_init/sceneC_loo_6 \
        -o output/gsfix_data/sceneC_6 \
        -r 4 --sparse_view_num 6 --sh_degree 2 --white_background
"""

import json
import os
import re
import shutil
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.arguments import ModelParams, PipelineParams
from utils.graphics_utils import getWorld2View
from utils.watermark_utils import CLEAN_DIRNAME, find_watermark_mask


def latest_gs_ply(model_dir):
    """Highest-iteration checkpoint under <model_dir>/point_cloud/iteration_*.

    Iteration counts are not fixed in this repo -- OptimizationParams.iterations
    is 10_000 rather than upstream 3DGS's 30_000 -- so the directory name has to
    be discovered rather than assumed.
    """
    pc_dir = os.path.join(model_dir, "point_cloud")
    if not os.path.isdir(pc_dir):
        return None
    iters = []
    for d in os.listdir(pc_dir):
        ply = os.path.join(pc_dir, d, "point_cloud.ply")
        if d.startswith("iteration_") and os.path.isfile(ply):
            try:
                iters.append((int(d.split("_")[-1]), ply))
            except ValueError:
                continue
    return max(iters)[1] if iters else None


def link_or_copy(src, dst):
    """Hardlink `src` to `dst`, falling back to a copy across filesystems.

    Every fine-tune pair repeats the same six ground-truth photos, so copying
    them outright would multiply a handful of megabytes by a few hundred for no
    reason. GSFix3D's loader stats each entry with os.path.isfile, which is happy
    with a hardlink.
    """
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def make_gt_cleaner(out_dir, mask, prompt, steps, verbose=True):
    """Return a fn mapping a leave directory's gt.png to a watermark-free copy.

    Cleaning happens once per leave directory, not once per pair: the same six
    photographs back a few hundred pairs, so caching turns ~240 diffusion passes
    into 6. Results are cached on disk too, so re-running the export with a
    different --pairs_window costs nothing.
    """
    from utils.realfill_utils import InPaint
    from utils.watermark_utils import clean_watermark

    cache_dir = os.path.join(out_dir, "finetune", "_gt_clean")
    os.makedirs(cache_dir, exist_ok=True)
    state = {"pipe": None}

    def clean(gt_path, leave_name):
        cached = os.path.join(cache_dir, f"{leave_name}.png")
        if os.path.exists(cached):
            return cached
        if state["pipe"] is None:
            # Deferred: loading SD2 onto the GPU is only worth it once we know
            # there is actually something to clean.
            print("[i] Loading SD2 inpainting for watermark removal ...")
            state["pipe"] = InPaint("")
        img = np.asarray(Image.open(gt_path).convert("RGB"))
        if img.shape[:2] != mask.shape[:2]:
            print(f"[!] {leave_name}: gt.png is {img.shape[1]}x{img.shape[0]} but the mask is "
                  f"{mask.shape[1]}x{mask.shape[0]}; leaving the watermark in place.")
            return gt_path
        print(f"[i] {leave_name}: painting out the watermark")
        out = clean_watermark(img, mask, state["pipe"], prompt=prompt, steps=steps,
                              verbose=verbose)
        Image.fromarray(out).save(cached)
        return cached

    return clean


def export_finetune_pairs(loo_dir, out_dir, pairs_window, gt_cleaner=None):
    """Pair leave-one-out renders with their withheld photograph.

    Only samples within `pairs_window` iterations of the first are kept, and that
    window matters. leave_one_out_stage1.py swaps its camera pool at iteration
    6000 (lines 77-80): below it the model trains on N-1 views, but above it the
    pool becomes `scene.getTrainCameras()` -- all N, including the very view being
    rendered into left_image/. Degradation therefore shrinks monotonically along
    the sequence, and the tail converges on the ground truth. Training GSFixer on
    the tail would teach it the identity map.

    utils/dataset_lora.py:GSCacheDataset applies the same guard for the stage-3
    LoRA, via cache_max_iter=240.
    """
    rgb_dir = os.path.join(out_dir, "finetune", "rgb")
    gs_dir = os.path.join(out_dir, "finetune", "rendered_gs")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(gs_dir, exist_ok=True)

    leave_dirs = sorted(
        d for d in os.listdir(loo_dir)
        if d.startswith("leave_") and os.path.isdir(os.path.join(loo_dir, d))
    )
    if not leave_dirs:
        raise FileNotFoundError(
            f"No leave_*/ directories under {loo_dir}. Run stage 2a first:\n"
            f"  python inference.py -i <scene> -o <out> --stages 2a -n <N>"
        )

    n_pairs = 0
    for leave in leave_dirs:
        leave_path = os.path.join(loo_dir, leave)
        gt_path = os.path.join(leave_path, "gt.png")
        sample_dir = os.path.join(leave_path, "left_image")
        if not os.path.isfile(gt_path) or not os.path.isdir(sample_dir):
            print(f"[!] {leave}: missing gt.png or left_image/, skipping")
            continue

        samples = []
        for name in os.listdir(sample_dir):
            m = re.fullmatch(r"sample_(\d+)\.png", name)
            if m:
                samples.append((int(m.group(1)), os.path.join(sample_dir, name)))
        if not samples:
            print(f"[!] {leave}: left_image/ is empty, skipping")
            continue

        samples.sort()
        min_it = samples[0][0]
        kept = [p for it, p in samples if it < min_it + pairs_window]

        # gt.png comes straight from Camera.original_image, so on a watermarked
        # scene it still carries the overlay even though every loss that touched
        # it was masked. Pairing it as-is would train the repair model to paint
        # the watermark back at fixed pixel coordinates.
        target_path = gt_cleaner(gt_path, leave) if gt_cleaner else gt_path

        for path in kept:
            name = f"{n_pairs:05d}.png"
            # Identical filenames in both directories: BaseFinetuneDataset sorts
            # rgb/ and rendered_gs/ independently and zips them by position, so
            # the two listings have to agree index for index.
            link_or_copy(path, os.path.join(gs_dir, name))
            link_or_copy(target_path, os.path.join(rgb_dir, name))
            n_pairs += 1

        print(f"[i] {leave}: {len(kept)}/{len(samples)} samples "
              f"(iterations {min_it}..{min_it + pairs_window - 1})")

    print(f"[i] Wrote {n_pairs} fine-tune pairs to {os.path.join(out_dir, 'finetune')}")
    return n_pairs


@torch.no_grad()
def export_novel_views(scene, gaussians, pipeline, background, out_dir, tag):
    """Render the ellipse orbit and record its poses.

    Same trajectory scripts/render.py:render_path uses and stage 5 samples from
    (threestudio/data/loo_mip.py), but deliberately *not* the same render call:

    - No `clean_indices`. render.py passes it to cull near-camera floaters for the
      demo video, and gaussian_renderer/__init__.py:50-55 implements that by
      assigning `opacity.data` on the output of a sigmoid. Harmless under
      no_grad, corrupting under autograd -- so the lift in
      scripts/refine_gs_gsfix.py cannot use it, and the conditioning image has to
      be rendered the same way the lift renders or the optimizer would be told to
      erase whatever the culling hid.
    - `test=False`, i.e. the supersample-then-pool path. That is how stage 1b
      trained and how render.py's metric pass (render_set) evaluates, so the
      whole stage stays on one rendering path.
    """
    render_dir = os.path.join(out_dir, "rendered_novel_views", f"gs_image_{tag}")
    before_dir = os.path.join(out_dir, "render_before")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(before_dir, exist_ok=True)

    views = scene.getRenderCameras()
    meta = []
    for idx, view in enumerate(tqdm(views, desc="Rendering orbit")):
        pkg = render(view, gaussians, pipeline, background, test=False)
        # gsplat returns unbounded radiance; sceneC peaks around 2.5. Saving
        # without clamping would wrap into garbage in the PNG.
        image = torch.clamp(pkg["render"], 0.0, 1.0)

        name = f"{idx:05d}.png"
        torchvision.utils.save_image(image, os.path.join(render_dir, name))
        link_or_copy(os.path.join(render_dir, name), os.path.join(before_dir, name))

        # GSFix3D reads `extrinsic` as world->camera and hands its [:3,:3] block
        # straight to a Camera that transposes it again -- the same convention
        # utils/graphics_utils.py:getWorld2View builds, so no remapping needed.
        meta.append({
            "file": name,
            "extrinsic": getWorld2View(view.R, view.T).astype(np.float64).tolist(),
            "FoVx": float(view.FoVx),
            "FoVy": float(view.FoVy),
            # Taken from the rendered tensor, not view.image_width/height:
            # render(test=False) calls Camera.refactor(2) and leaves those two
            # attributes holding the supersampled 2x size, while the image that
            # comes back out of the AvgPool is the native one. FoVx/FoVy are
            # untouched by refactor unless it is given a fov_scale, so they are
            # safe to read directly.
            "width": int(image.shape[-1]),
            "height": int(image.shape[-2]),
        })

    with open(os.path.join(out_dir, "novel_views.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[i] Wrote {len(views)} orbit renders to {render_dir}")
    return len(views)


def main():
    parser = ArgumentParser(description="Export an RI3D scene for GSFixer")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Bundle destination, e.g. output/gsfix_data/sceneC_6")
    parser.add_argument("--loo_dir", default=None,
                        help="Leave-one-out directory from stage 2a. Omit to skip "
                             "fine-tune pair export.")
    parser.add_argument("--load_ply", default=None,
                        help="PLY to render. Defaults to the highest iteration_* "
                             "checkpoint under --model_path.")
    parser.add_argument("--pairs_window", type=int, default=240,
                        help="Keep leave-one-out samples within this many iterations "
                             "of the first. Mirrors GSCacheDataset's cache_max_iter; "
                             "see export_finetune_pairs for why the window matters.")
    parser.add_argument("--recon_tag", default="ri3d",
                        help="Suffix GSFixer uses to locate renders "
                             "(gs_image_<tag>); pass the same value to its "
                             "inference.py --recon_method_type.")
    parser.add_argument("--watermark_mask", default=None,
                        help="PNG mask (255 = watermark) used to paint the overlay out of the "
                             "fine-tune targets. Auto-discovered from the scene directory when "
                             "omitted; scenes without one skip this entirely.")
    parser.add_argument("--no_watermark_clean", dest="watermark_clean",
                        action="store_false", default=True,
                        help="Pair against the raw gt.png even on a watermarked scene. This "
                             "trains the repair model to reproduce the watermark; only useful "
                             "for reproducing the un-fixed behaviour.")
    parser.add_argument("--inpaint_prompt", default="a photo",
                        help="Prompt for the SD2 watermark inpainting. Barely steers anything at "
                             "guidance_scale=1, but beats InPaint's xxy5syt00 default, which means "
                             "nothing to a checkpoint never fine-tuned on it.")
    parser.add_argument("--inpaint_steps", type=int, default=50,
                        help="Denoising steps per watermark blob")
    parser.add_argument("--sparse_view_num", type=int, required=True)
    parser.add_argument("--init_pcd_name", default="origin", type=str)
    parser.add_argument("--use_mask", default=True)
    parser.add_argument("--transform_the_world", action="store_true")
    parser.add_argument("--depth_conf_thr", type=float, default=0.0)
    parser.add_argument("--mono_depth_weight", type=float, default=0.005)
    parser.add_argument("--skip_novel_views", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    # readColmapSceneInfo asserts `eval == False` whenever sparse_view_num > 0.
    args.eval = False
    args.is_renderrr = True

    os.makedirs(args.output_dir, exist_ok=True)

    if args.loo_dir:
        cleaner = None
        clean_images = os.path.join(args.source_path, CLEAN_DIRNAME)
        if os.path.isdir(clean_images):
            # Stage `wmi` already replaced the watermark in the photographs
            # themselves, so leave_one_out trained on -- and saved gt.png from --
            # cleaned frames. Re-inpainting an already-clean image would only
            # degrade it.
            print(f"[i] {clean_images} exists; gt.png is already watermark-free, "
                  "skipping the inpainting pass.")
        elif args.watermark_clean:
            if args.watermark_mask:
                m = cv2.imread(args.watermark_mask, cv2.IMREAD_GRAYSCALE)
                mask = None if m is None else m > 127
                if mask is None:
                    print(f"[!] Could not read {args.watermark_mask}")
            else:
                mask = find_watermark_mask(args.source_path)
            if mask is not None:
                cleaner = make_gt_cleaner(args.output_dir, mask, args.inpaint_prompt,
                                          args.inpaint_steps)
        else:
            print("[!] --no_watermark_clean: targets keep any burned-in watermark, which "
                  "teaches GSFixer to reproduce it.")
        export_finetune_pairs(args.loo_dir, args.output_dir, args.pairs_window,
                              gt_cleaner=cleaner)
    else:
        print("[!] --loo_dir not given; skipping fine-tune pairs. GSFixer will have "
              "to run zero-shot.")

    if args.skip_novel_views:
        return

    ply_path = args.load_ply or latest_gs_ply(args.model_path)
    if ply_path is None or not os.path.isfile(ply_path):
        print(f"[x] No trained PLY found under {args.model_path}/point_cloud/iteration_*")
        sys.exit(1)
    print(f"[i] Rendering from {ply_path}")

    gaussians = GaussianModel(args.sh_degree, args.sparse_view_num,
                              spherical_gaussians=True)
    scene = Scene(lp.extract(args), gaussians, shuffle=False, extra_opts=args,
                  load_ply=ply_path)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    export_novel_views(scene, gaussians, pp.extract(args), background,
                       args.output_dir, args.recon_tag)


if __name__ == "__main__":
    main()
