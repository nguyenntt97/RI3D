"""Lift GSFixer-repaired novel views back into the Gaussians.

Stage 1c step 4. An RI3D-native port of GSFix3D/scripts/gsfix3d/refine_gs.py.

Ported rather than called because that script reads only Replica/ScanNet++ camera
formats, optimizes `_scaling` as three free channels (RI3D's model is isotropic --
scene/gaussian_model.py:141 derives all three from channel 0), and renders through
diff-gaussian-rasterization, whose submodule directory in the GSFix3D checkout is
empty. Staying in-repo keeps one renderer (gsplat) and one model class.

Two phases, following the paper:

  A. Per repaired view, a short burst of optimization with densification, so the
     repaired content actually gets geometry to live on.
  B. Joint epochs over the sparse photographs plus every repaired view, which is
     what pulls the model back onto the real data.

Usage:
    python scripts/refine_gs_gsfix.py \
        -s output/sceneC/ggpt_sfm -m output/gs_init/sceneC_6 \
        --fixed_image_path output/gsfix_data/sceneC_6/fixed/rgb \
        --output_dir output/gs_gsfix/sceneC_6 \
        -r 4 --sparse_view_num 6 --sh_degree 2 --white_background
"""

import os
import random
import sys
from argparse import ArgumentParser, Namespace
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.loss_utils import masked_l1_loss, masked_ssim
from utils import wandb_utils as wb


def latest_gs_ply(model_dir):
    """Highest-iteration checkpoint under <model_dir>/point_cloud/iteration_*."""
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


def load_image(path, device="cuda"):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def photometric_loss(image, gt, loss_mask=None):
    """0.8 L1 + 0.2 (1 - SSIM), the weighting refine_gs.py uses.

    Routed through the masked variants so a scene's watermark mask is honoured:
    sceneC's covers 11.4% of every photograph, and those pixels were never
    supervised in stage 1b. Supervising them here would train the burned-in logo
    into the model as scene content. `mask=None` falls through to the plain
    losses (utils/loss_utils.py:77), so novel views cost nothing extra.
    """
    return 0.8 * masked_l1_loss(image, gt, loss_mask) + \
           0.2 * (1.0 - masked_ssim(image, gt, loss_mask))


def anchor_batch(train_cams):
    """One (camera, ground truth, mask) triple per sparse photograph."""
    return [(c, c.original_image[:3].cuda(), getattr(c, "loss_mask", None))
            for c in train_cams]


def refine(args, dataset, opt, pipe):
    gaussians = GaussianModel(dataset.sh_degree, args.sparse_view_num,
                              spherical_gaussians=True)

    ply_path = args.load_ply or latest_gs_ply(args.base_model_path)
    if ply_path is None or not os.path.isfile(ply_path):
        print(f"[x] No trained PLY under {args.base_model_path}/point_cloud/iteration_*")
        sys.exit(1)
    print(f"[i] Refining {ply_path}")

    scene = Scene(dataset, gaussians, shuffle=False, extra_opts=args, load_ply=ply_path)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    render_cams = scene.getRenderCameras()
    anchors = anchor_batch(train_cams)

    fixed_paths = sorted(glob(os.path.join(args.fixed_image_path, "*.png")))
    if not fixed_paths:
        print(f"[x] No repaired images in {args.fixed_image_path}")
        sys.exit(1)
    if len(fixed_paths) != len(render_cams):
        print(f"[x] {len(fixed_paths)} repaired images but {len(render_cams)} render "
              f"cameras. The orbit is deterministic given the training cameras, so a "
              f"mismatch means the export ran against a different scene, -n, or -r.")
        sys.exit(1)

    # tools/gsfix_export.py names each orbit frame by its index and GSFixer keeps
    # the basename, so sorted order is index order on both sides.
    novel = list(zip(render_cams, fixed_paths))

    wb.attach()
    # One axis across both phases: they run in wall order, so a single x reads
    # left-to-right through the A->B transition. The phase lives in the metric
    # name instead, which keeps the three curves separable.
    wb.define_axis("stage_1c", "step")
    wb.log_summary({"stage_1c/n_novel": len(novel), "stage_1c/iters": args.iters,
                    "stage_1c/kf_iters": args.kf_iters,
                    "stage_1c/anchor_every": args.anchor_every})
    gstep = 0

    def step(cam, gt, loss_mask, densify_at=None, tag="phaseB"):
        nonlocal gstep
        pkg = render(cam, gaussians, pipe, background, test=False)
        image = pkg["render"]
        loss = photometric_loss(image, gt, loss_mask)
        loss.backward()

        with torch.no_grad():
            if densify_at is not None:
                visibility, radii = pkg["visibility_filter"], pkg["radii"]
                gaussians.max_radii2D[visibility] = torch.max(
                    gaussians.max_radii2D[visibility], radii[visibility])
                gaussians.add_densification_stats(pkg["viewspace_points"], visibility)
                if densify_at:
                    # max_screen_size=None: the screen-size prune belongs with
                    # opacity resets, which this stage does not do.
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, None)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        gstep += 1
        value = loss.item()
        if wb.due(gstep):
            wb.log({"stage_1c/step": gstep, f"stage_1c/loss_{tag}": value,
                    "stage_1c/num_gauss": gaussians.get_xyz.shape[0]})
        return value

    # ---- Phase A: fit each repaired view, densifying as we go ----
    print(f"[i] Phase A: {len(novel)} repaired views x {args.iters} iterations")
    for i, (cam, path) in enumerate(tqdm(novel, desc="Repaired views")):
        gt = load_image(path)
        for it in range(args.iters):
            step(cam, gt, None, densify_at=(it > 0 and it % 5 == 0), tag="phaseA")

        # Phase A has no ground-truth term, so left alone it drifts the model off
        # the photographs over ~2400 steps. Interleaving anchors bounds that.
        # --anchor_every 0 reproduces refine_gs.py exactly.
        if args.anchor_every and (i + 1) % args.anchor_every == 0:
            a_cam, a_gt, a_mask = random.choice(anchors)
            step(a_cam, a_gt, a_mask, tag="anchor")

        del gt

    print(f"[i] After phase A: {gaussians.get_xyz.shape[0]} gaussians")
    wb.log_summary({"stage_1c/n_gauss_after_A": gaussians.get_xyz.shape[0]})

    # ---- Phase B: joint epochs over photographs + repaired views ----
    # Repaired images are held on CPU and moved per step; 120 x 3 x 768 x 1024
    # floats is ~1.1 GB, which is worth not parking on the GPU.
    extended = [(c, g.cpu(), m) for c, g, m in anchors]
    extended += [(cam, load_image(path, device="cpu"), None) for cam, path in novel]

    print(f"[i] Phase B: {args.kf_iters} epochs over {len(extended)} views")
    for _ in tqdm(range(args.kf_iters), desc="Joint epochs"):
        order = list(range(len(extended)))
        random.shuffle(order)
        for idx in order:
            cam, gt, loss_mask = extended[idx]
            step(cam, gt.cuda(), loss_mask)

    print(f"[i] Final: {gaussians.get_xyz.shape[0]} gaussians")
    wb.log_summary({"stage_1c/n_gauss_final": gaussians.get_xyz.shape[0]})
    wb.finish()
    scene.save(args.save_iteration)
    print(f"[i] Saved {args.output_dir}/point_cloud/iteration_{args.save_iteration}/point_cloud.ply")


def main():
    parser = ArgumentParser(description="Lift GSFixer repairs into the Gaussians")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--fixed_image_path", required=True,
                        help="GSFixer output directory (its <output_dir>/rgb)")
    parser.add_argument("--output_dir", required=True,
                        help="Destination for the refined model")
    parser.add_argument("--load_ply", default=None,
                        help="PLY to refine. Defaults to the highest iteration_* "
                             "under --model_path.")
    parser.add_argument("--iters", type=int, default=20,
                        help="Phase-A iterations per repaired view")
    parser.add_argument("--kf_iters", type=int, default=50,
                        help="Phase-B epochs over the extended set")
    parser.add_argument("--anchor_every", type=int, default=1,
                        help="Inject one sparse-photograph step every N repaired "
                             "views during phase A. 0 disables (upstream behaviour).")
    parser.add_argument("--save_iteration", type=int, default=10000,
                        help="Directory number for the saved checkpoint. Matches the "
                             "stage-1b convention so render.py and inference.py's "
                             "_latest_gs_ply find it without special-casing; it is "
                             "not an iteration count.")
    parser.add_argument("--sparse_view_num", type=int, required=True)
    parser.add_argument("--init_pcd_name", default="origin", type=str)
    parser.add_argument("--use_mask", default=True)
    parser.add_argument("--transform_the_world", action="store_true")
    parser.add_argument("--depth_conf_thr", type=float, default=0.0)
    parser.add_argument("--mono_depth_weight", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    # readColmapSceneInfo asserts eval == False whenever sparse_view_num > 0.
    args.eval = False
    args.is_renderrr = False

    # Scene.save() writes under dataset.model_path; point it at the destination so
    # the stage-1b model it was loaded from is left untouched.
    args.base_model_path = args.model_path
    os.makedirs(args.output_dir, exist_ok=True)
    args.model_path = args.output_dir

    safe_state(args.quiet)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = lp.extract(args)

    # render.py resolves its arguments through get_combined_args, which reads
    # <model_path>/cfg_args -- without one the refined model cannot be rendered
    # or evaluated the same way every other stage output can.
    with open(os.path.join(args.output_dir, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    refine(args, dataset, op.extract(args), pp.extract(args))


if __name__ == "__main__":
    main()
