#
# FastGS backbone renderer.
#
# Wraps the vendored `diff_gaussian_rasterization_fastgs` kernel (FastGS, CVPR
# 2026; itself derived from 3DGS / Taming-3DGS / Speedy-Splat) in the same
# call signature and return dict as gaussian_renderer.render, so the training
# loops can swap between the two on `--gs_backbone`.
#
# The import is deliberately module-level rather than lazy: it is the one
# dependency in the tree that needs a CUDA build, and a run launched with
# `--gs_backbone fastgs` should fail at import with a clear message rather than
# minutes in, at the first densification.
#

import math

import torch

try:
    from diff_gaussian_rasterization_fastgs import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as exc:  # pragma: no cover - depends on the local build
    raise ImportError(
        "--gs_backbone fastgs needs the FastGS rasterizer, which is not installed. "
        "Build it with `uv sync --extra fastgs`; see docs/fastgs.md for the CUDA "
        "toolchain prerequisites."
    ) from exc
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _view_space_z(viewpoint_camera, xyz):
    """Per-Gaussian depth along the camera's view axis, [N, 1].

    `world_view_transform` is stored transposed (the gsplat path at
    gaussian_renderer/__init__.py takes `.T` to recover W2C), so world points
    multiply from the left.
    """
    ones = torch.ones_like(xyz[:, :1])
    xyz_h = torch.cat((xyz, ones), dim=-1)
    return (xyz_h @ viewpoint_camera.world_view_transform)[:, 2:3]


def render_fastgs(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                  scaling_modifier=1.0, override_color=None, mult=0.5,
                  get_flag=False, metric_map=None, test=False):
    """Render the scene with the FastGS rasterizer.

    Returns the same keys as `gaussian_renderer.render`, plus
    `accum_metric_counts` -- the per-Gaussian count of flagged pixels that the
    Gaussian actually contributed to, which is the signal FastGS densifies on.
    That count is only populated when `get_flag` is set and a `metric_map` is
    supplied; otherwise it is zeros.

    The vendored kernel is built with NUM_CHAFFELS == 4: channel 3 carries
    view-space depth, alpha-composited in the same blend loop as colour, and
    accumulated alpha comes back as its own output. Depth therefore costs one
    extra channel rather than the second full rasterization this used to do --
    which was both a 43% throughput tax and, at ~4.7M splats, enough extra
    buffer pressure to exhaust the allocator mid-run.

    Background tensor (bg_color) must be on GPU!
    """
    # Same supersample-then-pool anti-aliasing as the gsplat path: rasterize at
    # 2x and average down. `refactor` is derived from `original_image`, not from
    # the current size, so repeated calls are idempotent.
    if test:
        pool_op = lambda x: x
    else:
        viewpoint_camera.refactor(2)
        pool_op = torch.nn.AvgPool2d(2).to(pc.device)

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    xyz = pc.get_xyz
    opacity = pc.get_opacity

    # 4 columns, not 2: the FastGS backward writes the plain screen-space
    # gradient into 0:2 and the Abs-GS gradient into 2:4. GaussianModel
    # .add_densification_stats splits them back out.
    screenspace_points = torch.zeros((xyz.shape[0], 4), dtype=xyz.dtype,
                                     requires_grad=True, device=pc.device) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # The kernel indexes metric_map by raster-resolution pixel id, so a map built
    # against the pooled output has to be upsampled to match before it gets here.
    if metric_map is None:
        metric_map = torch.zeros(H * W, dtype=torch.int, device=pc.device)
        get_flag = False

    def build_settings(bg, want_counts):
        return GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            mult=mult,
            prefiltered=False,
            debug=pipe.debug,
            get_flag=want_counts,
            metric_map=metric_map,
        )

    if pipe.compute_cov3D_python:
        scales = rotations = None
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        cov3D_precomp = None

    # Bound up front. Upstream only assigns `dc` inside the SH branch, so its
    # convert_SHs_python and override_color paths raise NameError.
    #
    # Precomputed colours bypass preprocess's SH block, which is also what fills
    # the depth channel, so these paths have to supply all four themselves.
    def with_depth(rgb):
        return torch.cat((rgb, _view_space_z(viewpoint_camera, xyz)), dim=-1)

    dc = shs = colors_precomp = None
    if override_color is not None:
        colors_precomp = with_depth(override_color)
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = with_depth(torch.clamp_min(sh2rgb + 0.5, 0.0))
    else:
        dc, shs = pc.get_features_dc, pc.get_features_rest

    # The depth channel must composite over a background of 0 so the output is
    # sum(w_i * z_i), matching gsplat's "RGB+D" accumulated depth exactly.
    bg4 = torch.cat((bg_color, torch.zeros_like(bg_color[:1])))
    rasterizer = GaussianRasterizer(raster_settings=build_settings(bg4, get_flag))
    rendered, alpha, radii, accum_metric_counts = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": pool_op(rendered[:3]),
        "rendered_depth": pool_op(rendered[3:4]),
        "rendered_alpha": pool_op(alpha),
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "accum_metric_counts": accum_metric_counts,
    }
