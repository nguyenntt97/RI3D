# FastGS backbone (`--gs_backbone fastgs`)

An alternative Gaussian backbone for the densifying stages, ported from
[FastGS](https://github.com/fastgs/FastGS) (CVPR 2026). Off by default: the pipeline runs vanilla
3DGS on gsplat unless you ask for it.

```bash
python inference.py -i data/mipnerf360/bicycle --stages 1a,1b --num_views 3 --gs_backbone fastgs
```

## What it changes

FastGS is not a different splat representation — the `.ply` it writes is ordinary 3DGS, and every
downstream stage and viewer reads it unchanged. What differs is the densification and optimization
loop:

1. **Its own CUDA rasterizer** (derived from 3DGS / Taming-3DGS / Speedy-Splat) with tighter
   per-splat tile boxes, controlled by `--fastgs_mult`.
2. **Multi-view consistent densification.** At each densification interval it renders a sample of
   training views, flags the pixels each reconstructs badly, and asks the rasterizer for a
   per-Gaussian count of flagged pixels that Gaussian actually contributed to. Only Gaussians
   several views agree on are cloned or split; the rest become pruning candidates. This is the part
   vanilla 3DGS has no equivalent of.
3. **Two gradient thresholds** — the plain screen-space gradient decides clones,
   the Abs-GS gradient decides splits — partitioned by splat scale against `--fastgs_dense`.
4. **A strided optimizer schedule.** High-order SH lives in a second Adam stepped every 16
   iterations; past the halfway and two-thirds marks the whole update is strided to 1/32 and 1/64.

## Which stages it affects

| stage | script | affected |
|---|---|---|
| 1a · Gaussian init | `scripts/train_gs_init.py` | no — does not densify |
| 1b · Base 3DGS | `scripts/train_gs.py` | **yes** |
| 2a · Leave-one-out pass 1 | `scripts/leave_one_out_stage1.py` | **yes** |
| 2b · Leave-one-out pass 2 | `scripts/leave_one_out_stage2.py` | flag threaded, but no densification |
| 1c · GSFix3D repair | `scripts/refine_gs_gsfix.py` | no — still vanilla 3DGS |
| 5a/5b · Prior-guided | `threestudio/systems/gaussian_object_system_mip.py` | no — still vanilla 3DGS |

2b does not densify, but it resumes 2a's checkpoint, and the optimizer layout is a property of the
backbone that wrote it (fastgs holds `f_rest` in a second Adam). `restore()` refuses a mismatch with
an explicit error rather than silently mis-loading, so pass the same `--gs_backbone` to both. Running
`inference.py` does this for you.

## Environment requirements

This is the only dependency in the tree that compiles CUDA at install time, which is why it is an
opt-in extra rather than a hard dependency.

**Prerequisites**

- A CUDA toolkit whose `nvcc` matches the CUDA your PyTorch was built against. Check with:
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
  nvcc --version
  ```
  Mixing major versions between the wheel and the compiler produces link-time or runtime ABI errors.
  Note the repo still pins `torch==2.1.0` (cu121) in `pyproject.toml` while the environment this is
  developed against runs a considerably newer torch — build against whichever one actually runs the
  pipeline.
- `CUDA_HOME` pointing at that toolkit.
- A GPU of compute capability **7.0 or higher**. If the build machine's GPU differs from the run
  machine's, set `TORCH_CUDA_ARCH_LIST` explicitly (e.g. `export TORCH_CUDA_ARCH_LIST="8.9"`).
- A C++ compiler compatible with your PyTorch extensions build.

**Install**

```bash
uv sync --extra fastgs
python -c "import diff_gaussian_rasterization_fastgs; print('ok')"
```

The extension is vendored at [`third_party/diff-gaussian-rasterization_fastgs/`](../third_party/diff-gaussian-rasterization_fastgs/),
alongside `third_party/CLIP` and `third_party/minLoRA`. Two local changes to upstream:

- `rasterize_points.cu` was migrated from `Tensor::data<T>()` to `Tensor::data_ptr<T>()` in 42
  places. The former was removed from libtorch and the file does not compile against a modern torch
  without it.
- glm's test suite and generated docs were deleted (23 MB → 3.7 MB); the headers the build needs are
  untouched.

**Licence.** The rasterizer carries the INRIA 3D Gaussian Splatting licence
([`LICENSE.md`](../third_party/diff-gaussian-rasterization_fastgs/LICENSE.md)): free for
non-commercial, research and evaluation use only. That is stricter than the rest of this repo.

## Parameters

All live in `OptimizationParams` (`utils/arguments.py`) and are inert under `--gs_backbone 3dgs`.

| flag | default | notes |
|---|---|---|
| `--gs_backbone` | `3dgs` | `3dgs` or `fastgs` |
| `--fastgs_loss_thresh` | 0.1 | normalized L1 above which a pixel counts as high-error |
| `--fastgs_grad_thresh` | 0.0002 | clone candidates, plain screen-space gradient |
| `--fastgs_grad_abs_thresh` | 0.0012 | split candidates, Abs-GS gradient |
| `--fastgs_dense` | 0.001 | fraction of scene extent partitioning clone from split |
| `--fastgs_mult` | 0.5 | compact-box multiplier; how many tiles each splat touches |
| `--fastgs_score_cams` | 10 | cameras sampled per scoring pass, clamped to what the stage has |
| `--fastgs_importance_thresh` | 5.0 | flagged-pixel count a Gaussian needs before it densifies |
| `--fastgs_highfeature_lr` | 0.005 | `f_rest` learning rate (divided by 20, as `feature_lr` is) |
| `--fastgs_lowfeature_lr` | 0.0025 | `f_dc` learning rate |

Upstream's hardcoded iteration schedules — the 15k/20k optimizer stride switches, and the late-stage
prune between 15k and 30k — are expressed here as fractions of `--iterations`, because this repo's
default run is 10k rather than 30k. Passing upstream's numbers verbatim would have left the late
prune as dead code.


## Status: works; slower-converging than 3dgs

`--gs_backbone fastgs` completes a full 10k stage 1b. Matched A/B on sceneC (6 views, 4.7M splat
init, `--wm_loss_weight 0.2`, identical flags otherwise):

| | `3dgs` | `fastgs` |
|---|---|---|
| wall clock | 364.6 s | **251.9 s** (1.45x) |
| final splats | 4,457,171 | **3,011,451** (-32%) |
| test PSNR | **40.75** | 35.32 |
| train PSNR | **39.70** | 34.08 |

So it is 1.45x faster and a third smaller, but **5.4 dB worse**. Prime suspect is the strided
optimizer schedule: past half-way it steps every 32 iterations and past two-thirds every 64, so a
10k run gets roughly 5k full-rate steps against 3dgs's 10k. Upstream tunes that for 30k. The late
`final_prune_fastgs` passes and the weak multi-view signal on 6 views are secondary suspects. None
of this has been tuned yet.

### The bug that made it crash, and the fix

`--gs_backbone fastgs` used to die with `cudaErrorIllegalAddress` between iteration ~400 and ~900.
Root cause: **`duplicateWithKeys` wrote a different number of tile entries than the count pass had
promised**, overrunning the Gaussian's slice of the binning buffer (and the end of the buffer for
the last Gaussian). Caught by a device-side assertion:

```
[fastgs OOB] dup idx=4520719 off=5268162 wrote=2 promised=1 R=5572316
[fastgs OOB] dup idx=2787672 off=3933400 wrote=1 promised=2 R=5776239
```

Both passes call the same `__device__ inline` box helper in `auxiliary.h`, but it is inlined into
two different kernels, so nvcc contracted multiply-adds differently in each. Splats whose ellipse
grazes a tile boundary then round one way when counted and the other way when written. The fix is
`-fmad=false` in `setup.py`, forcing both passes to identical arithmetic. After it, the assertion
fires zero times and the run completes.

Three further bugs were found and fixed along the way:

1. `geom.rgb` sized `P * 3` while the depth channel made it `P * 4`, with `tiles_touched` allocated
   from the next bytes of the same chunk — *introduced by the depth patch, fixed*.
2. `backward.cu`'s `sampled_ar` prefetch read past the buffer on the final bucket — *upstream*.
3. The same block indexed `pixel_colors` past the bottom tile row — *upstream*.

The device-side assertions are left compiled in. They cost one integer comparison per write on the
healthy path and would catch a regression of exactly this class; set `FASTGS_DEBUG_BUCKETS=1` for
the per-rasterization bucket accounting as well.

Note for anyone diagnosing this kernel: it never checks `cudaMalloc`'s return value, so allocation
failures also surface as an illegal access at an unrelated call rather than as a clean OOM. And
`compute-sanitizer` is not usable here -- its own overhead exhausts a 31 GB card before the fault
is reached, even at 472k splats and quarter resolution.

## Known constraints

**Tuned for dense captures.** FastGS's consistency score assumes roughly 10 sampled cameras out of
100+. Stage 1b here trains on `--num_views` (3 by default) and stage 2a on one fewer, so the signal
is weak and `--fastgs_importance_thresh` will very likely need lowering from 5. If the splat count
stays flat across densification intervals, that threshold is the first thing to check — it means the
importance mask is coming back empty. FastGS has a released sparse-view branch
([`fast-dropgaussian`](https://github.com/fastgs/FastGS/tree/fast-dropgaussian)) worth reading
before tuning blind.

**Depth comes from a patched kernel.** Upstream's rasterizer is compiled for 3 channels and emits
neither depth nor alpha, but stage 1b's monocular depth loss needs both. The vendored copy is built
with `NUM_CHAFFELS == 4`: channel 3 carries view-space depth and composites in the same blend loop,
and accumulated alpha is returned as its own output. Verified against gsplat on sceneC at 4.7M
splats: depth and alpha agree to 0.000% median relative error, RGB to 60.5 dB, and the depth
gradient into `xyz` has median cosine similarity 0.9986. An earlier design recovered depth from a
second full rasterization; that cost 62% of an iteration and was what exhausted the allocator.

**SSIM is this repo's, not FastGS's.** Upstream uses `fused_ssim` for speed. That cannot honour the
watermark loss mask, so the masked L1 + masked SSIM from `utils/loss_utils.py` are used instead —
correctness of the `wm`/`wmi` stages over raw throughput.

**Stage 2a scores on the leave-one-out set.** The held-out view is the product of that stage and is
excluded from the scoring cameras, not just from the training stack.

## Verifying a build

```bash
# 1. the default backbone must be unaffected
python inference.py -i <scene> --stages 1a,1b --num_views 3

# 2. short fastgs run; watch that the splat count actually moves at densification intervals
python -W ignore scripts/train_gs.py -s <sfm_dir> -m /tmp/fastgs_smoke \
  -r 4 --sparse_view_num 3 --sh_degree 2 --white_background --random_background \
  --ply_path <stage_1a>/point_cloud/iteration_1/point_cloud.ply \
  --iterations 2000 --gs_backbone fastgs

# 3. A/B on the same scene; compare wall-clock, splat count and eval PSNR
#    from the wandb stage_1b/* series, which already logs all three
python inference.py -i <scene> --stages 1b --num_views 3 --wandb
python inference.py -i <scene> --stages 1b --num_views 3 --wandb --gs_backbone fastgs
```

## Citation

```
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}
```
