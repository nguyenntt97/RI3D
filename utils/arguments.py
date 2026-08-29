#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 4
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        # Belongs here rather than on the individual training scripts: Scene hands
        # loadCam the *extracted* ModelParams group, and extract() only copies
        # attributes declared in this class. A flag added to a script's own parser
        # silently never arrives.
        #
        # Weight for pixels under the watermark mask. 0 ignores them entirely --
        # the only safe value while the photographs still carry the watermark.
        # After stage `wmi` has inpainted them, a small positive value lets that
        # region constrain the Gaussians without invented content outweighing real
        # photographs.
        self.wm_loss_weight = 0.0
        # Minimum distance the novel-view orbit must keep from reconstructed
        # geometry. The ellipse is fitted from camera positions alone and knows
        # nothing about walls, so indoors it can pass straight through them --
        # measured on sceneC, 29 of 120 poses sat within 0.02 of a surface.
        #   < 0  auto: match the closest real photograph's own clearance. The SfM
        #        gauge is arbitrary per solve, so an absolute default would not
        #        travel between scenes or backends; deriving it from the capture
        #        does.
        #   = 0  disabled, the historical unconstrained path.
        #   > 0  explicit distance in world units.
        self.orbit_min_clearance = -1.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.max_num_splats = 5_000_000 # Stop densifying after this number of splats is reached
        self.iterations = 10_000 # [default 30_000] Each iteration corresponds to reconstructing 1 image. The number of points being optimized increases over
        self.position_lr_init = 0.00016 # [default 0.00016] Learning rate should be smaller for more extensive scenes
        self.position_lr_final = 0.0000016 # [default 0.0000016] Learning rate should be smaller for more extensive scenes
        self.position_lr_delay_mult = 0.01 # [default 0.01]
        self.position_lr_max_steps = 30_000 # [default 30_000]
        self.feature_lr = 0.0025 # [default 0.0025]
        self.opacity_lr = 0.05 # [default 0.05]
        self.scaling_lr = 0.001 # [default 0.005]
        self.rotation_lr = 0.001 # [default 0.001]
        self.percent_dense = 0.01 # [default 0.01] percent_dense * scene_extent = threshold size to determine whether to split (current is too large) or clone (current is small) gaussian
        self.lambda_dssim = 0.2 # [default 0.2] Loss = (1-lambda) * L1_loss + lambda * D-SSIM_Loss. L1 = abs(pred_pixel - true_pixel). SSIM = similarity between 2 images (luminance, contrast, structure)
        self.lambda_silhouette = 0.01 # [default 0.01] use bce loss for silhouette
        self.densification_interval = 100 # [default 100] Increase this to avoid running out of memory (how many iterations in between densifying/splitting gaussians)
        self.opacity_reset_interval = 400 # [default 3000] Decrease all opacities (alpha) close to zero -> algo will automatically increase opacities again for important gaussians -> cull the rest
        self.remove_outliers_interval = 500 # [default 500]
        self.densify_from_iter = 500 # [default 500] After this many iterations, start densifying
        self.densify_until_iter = int(0.6 * self.iterations) # [default 15_000] Decrease this to avoid running out of memory (after this many iterations, stop densifying)
        self.densify_grad_threshold = 0.0002 # [default 0.0002; Section 5.2: tau_pos] Increase this to avoid running out of memory. If very high, no densification will occur
        self.start_sample_pseudo = 400000 # not use
        self.end_sample_pseudo = 1000000 # not use
        self.sample_pseudo_interval = 10 # not use
        self.random_background = False

        # Which Gaussian backbone the densifying training loops run.
        #   "3dgs"   gsplat rasterization + vanilla clone/split on a gradient
        #            threshold. The default, and the only one that needs no
        #            CUDA build.
        #   "fastgs" FastGS (CVPR 2026): its own rasterizer plus multi-view
        #            consistent densification. Requires `uv sync --extra fastgs`;
        #            see docs/fastgs.md.
        # Validate with resolve_gs_backbone() -- ParamGroup infers `type` from the
        # default and has no way to express argparse `choices`, so a typo would
        # otherwise sail through and silently select the default.
        self.gs_backbone = "3dgs"

        # FastGS knobs. Inert under "3dgs". Defaults are upstream's except where
        # noted; upstream tunes for 100+ view captures at 30k iterations, and this
        # pipeline runs 3 views at 10k, so expect to retune.
        self.fastgs_loss_thresh = 0.1          # normalized L1 above which a pixel is "high error"
        self.fastgs_grad_thresh = 0.0002       # clone candidates, normal screen-space gradient
        self.fastgs_grad_abs_thresh = 0.0012   # split candidates, Abs-GS gradient
        self.fastgs_dense = 0.001              # scale/extent partition between clone and split
        self.fastgs_mult = 0.5                 # compact-box multiplier (tiles touched per splat)
        self.fastgs_score_cams = 10            # cameras sampled per scoring pass, clamped to the stack
        self.fastgs_importance_thresh = 5.0    # flagged-pixel count a Gaussian needs to densify
        self.fastgs_highfeature_lr = 0.005     # f_rest; divided by 20 the way feature_lr is
        self.fastgs_lowfeature_lr = 0.0025     # f_dc
        super().__init__(parser, "Optimization Parameters")

GS_BACKBONES = ("3dgs", "fastgs")


def resolve_gs_backbone(opt):
    """Validate and return `opt.gs_backbone`.

    ParamGroup builds its arguments from attribute types alone, so `--gs_backbone`
    accepts any string. Without this an unrecognised value -- a typo, or the
    upstream spelling "fast_gs" -- would run the default backbone under a name
    that says otherwise, which is the kind of thing you only notice after a full
    A/B has produced two identical numbers.
    """
    backbone = getattr(opt, "gs_backbone", "3dgs")
    if backbone not in GS_BACKBONES:
        raise ValueError(
            f"Unknown --gs_backbone {backbone!r}; expected one of {', '.join(GS_BACKBONES)}."
        )
    return backbone


LOO_DENSIFY_FRACTION = 0.6


def apply_loo_iterations(args, loo_iterations):
    """Rescale everything a leave-one-out run couples to its iteration count.

    Shortening a leave-one-out run by passing `--iterations` alone does not work,
    and fails silently. Four values move together and only one of them follows:

      - `densify_until_iter` is derived in OptimizationParams.__init__ as
        0.6 * iterations, i.e. from the *default* 10_000. It stays 6000 however
        `--iterations` is set. Sample capture in leave_one_out_stage1.py is gated
        on `iteration > densify_until_iter`, so a short run writes no
        `left_image/` samples at all -- the entire product of the stage.
      - `checkpoint_iterations` defaults to [6000], the checkpoint stage 2b
        resumes from.
      - stage 2b's resume filename is built from the same number.
      - `position_lr_max_steps` is 30_000, so a short run never gets far enough
        down the xyz schedule for the model to settle.

    Call this on the argparse namespace *before* OptimizationParams.extract, which
    copies these attributes across. Returns the densify/capture boundary, which is
    also the checkpoint iteration.
    """
    args.iterations = loo_iterations
    boundary = int(LOO_DENSIFY_FRACTION * loo_iterations)
    args.densify_until_iter = boundary
    args.checkpoint_iterations = [boundary]
    args.position_lr_max_steps = loo_iterations
    return boundary


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
