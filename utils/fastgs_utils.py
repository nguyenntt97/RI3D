"""Multi-view consistency scoring for the FastGS backbone.

Ported from FastGS/utils/fast_utils.py. This is FastGS's main contribution over
vanilla 3DGS: instead of densifying every Gaussian whose screen-space gradient
crosses a threshold, it renders a handful of training views, flags the pixels
each one reconstructs badly, and asks the rasterizer which Gaussians actually
contributed to those pixels. Only Gaussians that several views agree on are
grown; the rest are candidates for pruning.

Four things differ from upstream, each marked at its site: camera sampling is
clamped (this pipeline trains on 3 views, upstream on 100+), the error map is
resampled to rasterizer resolution, the min-max normalisations are guarded, and
everything stays on the model's device.
"""

import random

import torch
import torch.nn.functional as F

from gaussian_renderer.fastgs import render_fastgs
from utils.loss_utils import masked_l1_loss, masked_ssim


def sampling_cameras(viewpoint_stack, num_cams=10):
    """Draw up to `num_cams` cameras from `viewpoint_stack` without replacement.

    The clamp is not cosmetic. Upstream pops exactly 10, which is safe on a
    Mip-NeRF 360 capture but raises on the first densification here: stage 1b
    trains on `--num_views` cameras (3 by default) and stage 2a on one fewer.
    """
    stack = list(viewpoint_stack)
    num_cams = min(num_cams, len(stack))
    return [stack.pop(random.randint(0, len(stack) - 1)) for _ in range(num_cams)]


def _normalize01(value):
    """Min-max to [0, 1], flat-mapped to zeros when the input is constant.

    A 3-view scene can easily produce a degenerate error map -- every pixel
    below threshold, or a single view dominating -- where upstream's unguarded
    `(x - min) / (max - min)` divides by zero and poisons the score with NaN.
    """
    value = torch.nan_to_num(value, nan=0.0)
    lo, hi = torch.min(value), torch.max(value)
    if not torch.isfinite(hi - lo) or (hi - lo) <= 0:
        return torch.zeros_like(value)
    return (value - lo) / (hi - lo)


def _error_map(render_image, gt_image):
    """Per-pixel normalized L1 between a render and its ground truth, [H, W]."""
    return _normalize01(torch.mean(torch.abs(render_image - gt_image), 0).detach())


def compute_gaussian_score(camlist, gaussians, pipe, bg, opt, densify=False):
    """Score every Gaussian by how consistently views disagree with it.

    Returns `(importance_score, pruning_score)`:

      importance_score  per-Gaussian count of flagged pixels it contributed to,
                        averaged over views and floored. Gaussians above
                        `fastgs_importance_thresh` are eligible to densify.
                        Only computed when `densify` is set.
      pruning_score     the same counts weighted by each view's photometric
                        loss and normalized to [0, 1]. High means the Gaussian
                        sits where the reconstruction is worst.

    Call under `torch.no_grad()`; nothing here needs a backward pass.
    """
    device = gaussians.device
    full_metric_counts = None
    full_metric_score = None

    for viewpoint_cam in camlist:
        # First pass: what does this view get wrong?
        render_image = render_fastgs(viewpoint_cam, gaussians, pipe, bg,
                                     mult=opt.fastgs_mult)["render"]
        gt_image = viewpoint_cam.original_image.to(render_image.dtype).to(device)

        loss_mask = getattr(viewpoint_cam, "loss_mask", None)
        photometric_loss = ((1.0 - opt.lambda_dssim) * masked_l1_loss(render_image, gt_image, loss_mask)
                            + opt.lambda_dssim * (1.0 - masked_ssim(render_image, gt_image, loss_mask)))

        metric_map = (_error_map(render_image, gt_image) > opt.fastgs_loss_thresh).int()
        if loss_mask is not None:
            # Watermarked pixels are not supervised, so an error there says
            # nothing about the Gaussians behind it.
            metric_map = metric_map * (loss_mask.to(device) > 0).reshape(metric_map.shape).int()

        # The kernel indexes metric_map by raster-resolution pixel id, and
        # render_fastgs rasterizes at 2x before pooling down, so the map has to
        # be resampled to match or it addresses the wrong pixels entirely.
        raster_h, raster_w = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
        if metric_map.shape != (raster_h, raster_w):
            metric_map = F.interpolate(metric_map[None, None].float(),
                                       size=(raster_h, raster_w), mode="nearest")[0, 0]
        metric_map = metric_map.reshape(-1).contiguous().to(torch.int32).to(device)

        # Second pass: same view, now asking the rasterizer to tally per-Gaussian
        # hits against the flagged pixels.
        counts = render_fastgs(viewpoint_cam, gaussians, pipe, bg,
                               mult=opt.fastgs_mult,
                               get_flag=True, metric_map=metric_map)["accum_metric_counts"]
        counts = counts.to(torch.float32)

        if densify:
            full_metric_counts = counts.clone() if full_metric_counts is None else full_metric_counts + counts
        weighted = photometric_loss * counts
        full_metric_score = weighted.clone() if full_metric_score is None else full_metric_score + weighted

    pruning_score = _normalize01(full_metric_score)
    importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor') if densify else None
    return importance_score, pruning_score
