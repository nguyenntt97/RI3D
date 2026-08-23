# Isometric Dollhouse Rendering of Indoor 3DGS

An indoor scene is photographed from *inside*. Every Gaussian that lands on the ceiling therefore
sits between an elevated camera and the room, and any top-down or isometric view renders the
underside of a roof. `scripts/view_gs_web.py` serves a model in a browser and removes that
occluder at view time; `utils/dollhouse_utils.py` holds the geometry and runs standalone.

This document records how the cut is derived, what was measured on the way, and the defects found
while building it. The user-facing summary is in [`pipeline.md`](pipeline.md) under
*Viewing an indoor result*.

Measurements throughout are on `output/gs_init/sceneC_6` — the stage-1b model of `examples/sceneC`,
a 6-view indoor bedroom, solved with the `ggpt` backend. It is the input to stage `1c`, and 1c's
output is format-identical, so numbers carry over.

---

## 1. Entry points and contract

| | |
|---|---|
| `scripts/view_gs_web.py` | browser viewer (viser + nerfview + gsplat) and headless PNG mode |
| `utils/dollhouse_utils.py` | up axis, room frame, ceiling detection, cut, isometric camera |

`-m` accepts a PLY **or** a model directory, in which case the highest
`point_cloud/iteration_*/point_cloud.ply` wins. Iteration counts are not fixed in this repo
(`--iterations` defaults to 10_000, not upstream 3DGS's 30_000), so nothing may hardcode a number.

`-c` points at the SfM `cameras.json`. It is optional but strongly wanted: it is the only reliable
source of the world up axis (§3) and it supplies the camera-height guard on the cut (§4).

Nothing here is stage-specific. Stages 1b, 1c, 5a and 5b all write plain 3DGS PLYs.

**Every operation is a view-time filter.** No PLY is ever rewritten. The cut is a boolean mask
plus one indexed gather, recomputed on GUI change rather than per frame, which is what makes it a
live slider instead of a reload.

---

## 2. Why the cut is geometric and not backface culling

### The technique that does not apply

The standard dollhouse technique is to drop splats whose normal faces away from the camera. A
ceiling fitted from inside points its normal down, so an elevated camera discards it for free —
and the near walls (§8) with it, in the same pass, per-splat rather than per-slab. A 3D Gaussian's
normal is its shortest scale axis, so the test needs anisotropic splats.

**Stage 1c's model does not have them.** Nor do 1b, 2a or 2b.

### Isotropy is per-producer, not a property of the format

`GaussianModel.__init__` (`scene/gaussian_model.py:51`) takes a `spherical_gaussians` flag
defaulting to `False`. When it is set, `get_scaling` (`:142`) returns channel 0 repeated:

```python
if self.spherical_gaussians:
    scaling = self._scaling[:, :1].repeat(1, 3)
    return self.scaling_activation(scaling)
```

`save_ply` (`:477`) writes `scaling_inverse_activation(self.get_scaling)` — the *effective* scale,
not the raw parameter — so the flag is baked into the file rather than being a runtime detail a
reader could undo. Which producers set it:

| producer | construction site | `spherical_gaussians` |
|---|---|---|
| 1a init | `train_gs_init.py:40` | absent → **False** |
| 1b base | `train_gs.py:39` | **True** |
| 2a / 2b leave-one-out | `leave_one_out_stage{1,2}.py:34` | **True** |
| **1c GSFix3D repair** | `refine_gs_gsfix.py:88`, `gsfix_export.py:342` | **True** |
| 5a / 5b | `threestudio/systems/gaussian_object_system_mip.py:265` | absent → **False** |

Measured on the saved PLYs, `min(scale)/max(scale)` per splat:

| model | gaussians | channels identical | ratio p1 / p50 / p99 |
|---|---|---|---|
| sceneC stage 1b | 4,454,882 | **yes** | **1.000 / 1.000 / 1.000** |
| sceneA stage 5b export | 5,165,292 | no | 0.161 / **0.805** / 0.985 |

sceneC's 1b model is isotropic to the last splat — all three log-scale channels have column mean
`−6.521468`, `exp(scale)` p50 `0.00145`, and `rot_0` p50 is `1.0` against a quaternion norm of
`1.0`, so the rotations are identity too. There is no shortest axis and no orientation to read a
normal off.

`refine_gs_gsfix.py`'s docstring records the same constraint from the other direction: GSFix3D's
upstream `refine_gs.py` optimizes `_scaling` as three free channels, which is one of the reasons
that script was ported rather than called.

### Consequence

The cut is made in room coordinates instead — §3 for the frame, §4 for the ceiling, §8 for the
walls. This is the whole reason the approach is geometric rather than a two-line normal test.

> **On a stage-5 export, backface culling would work.** 5a/5b splats are genuinely anisotropic
> (p50 ratio 0.805), so a normal test is available there and would be the better mechanism — it is
> per-splat, so it would not take furniture with it the way §8's slab does. It is not implemented,
> because stage `1c` is this viewer's stated target and the geometric path covers every producer.
> If it is ever added it must stay a *fallback-guarded* option: applying it to a 1b/1c model
> silently removes nothing, since every normal there is arbitrary.

---

## 3. The room frame

All heights, cuts and slabs are expressed in an orthonormal basis `(e1, e2, up)` fitted to the
room, so they mean the same thing regardless of where the SfM gauge landed. `build_room_frame`
returns a `RoomFrame` carrying that basis, a centre, and a robust box.

### Up axis

`cameras.json` stores OpenCV c2w, whose **second column is the camera's down axis**. Indoor
photographs are shot close to level, so averaging those columns and negating recovers world up.

| scene | views | up | min pairwise dot of the down columns |
|---|---|---|---|
| sceneC (`ggpt`) | 6 | `[−0.0002, −1.0000, +0.0027]` | **0.998** |
| sceneA (`mast3r`) | 3 | `[−0.1288, +0.0670, −0.9894]` | 0.993 |

The two gauges are unrelated — sceneC's up is essentially `−y`, sceneA's essentially `−z` — which
is exactly why this is estimated rather than assumed. The reported dot is the honest confidence
measure: it is how much the photographer tilted.

Without `cameras.json` the fallback picks the thinnest extent axis and guesses the negative sense.
Rooms are wider than they are tall, so it is usually right, but nothing in an isotropic Gaussian
cloud actually identifies up. **The fallback is a guess and says so.** `--up` takes `x`/`-y`/… or
three floats.

### Wall-aligned horizontal axes

`e1`/`e2` are chosen by minimising the area of the robust horizontal bounding rectangle over
rotations in [0°, 90°), sampled at 90 angles on up to 400k points. A rectangular room has a sharp
minimum where the rectangle sits square to the walls; a round or cluttered one has a shallow
minimum, which costs nothing because any horizontal basis is then as good as another.

sceneC lands at **0.0°** (the GGPT gauge already happened to be wall-aligned), sceneA at **41.0°**.

### Box, centre and target

The box is a per-axis weighted percentile of the room coordinates, opacity-weighted. `set_box`
redraws it at a new percentile without touching the basis (§9).

`center` is pinned to the box the frame was *built* with, so cached room coordinates stay valid
when the box is redrawn; `target` is the current box's centre and is what a view looks at. Keeping
these separate is what lets the tightness slider reframe without invalidating the GPU-side
coordinate tensor.

---

## 4. Ceiling detection

A ceiling is a horizontal plane, so it occupies a narrow band of heights and appears as a spike
near the top of an opacity-weighted height histogram — distinct from the flat plateau that walls
and furniture make through the middle of the room.

### The distribution

sceneC, 64 bins over the 0.5–99.5 weighted percentile range, opacity-weighted, splats below
opacity 0.1 excluded (3,956,898 of 4,454,882 survive that):

| region | height (raw, `means·up`) | density, normalised to the floor peak |
|---|---|---|
| floor spike | `−0.327` | **1.000** |
| room plateau (walls, furniture) | `−0.25` … `+0.74` | ~0.18, very flat |
| ceiling shoulder begins | `+0.75` | 0.127 rising |
| **ceiling spike** | **`+0.876`** | **0.361** |
| above the ceiling | `+0.92` | 0.138 falling |

Raw height percentiles for the same set: p0 `−0.610`, p0.5 `−0.546`, p50 `−0.032`,
p99.5 `+0.933`, p100 `+0.979`.

### The algorithm

1. Robust range from the 0.5/99.5 weighted percentiles; 192 bins, box-smoothed over 5.
2. **Floor** = strongest peak in the bottom 35% of the range.
3. **Plateau** = median density between `floor + 8%` and `hi − 15%` of the span.
4. **Ceiling** = strongest peak in the top 25%.
5. `prominence = (peak − plateau) / plateau`.
6. **Cut** = walk down the near side of the spike to where density returns to
   `plateau + 0.2 × (peak − plateau)`, then a further 2% of span of margin.

Step 6 is the one that matters. A fixed offset below the peak does not travel: a ceiling the
cameras barely saw reconstructs as a thin slab and a well-observed one as a broad band, and the cut
has to clear whichever it is. Walking to the base of the spike adapts automatically.

### Results

Heights below are **room-frame** (offset from the box centre), unlike the raw `means·up` values in
the table above:

| scene | floor | ceiling | cut | prominence | verdict |
|---|---|---|---|---|---|
| sceneC @ percentile 0.5 | `−0.5051` | `+0.6902` | `+0.6220` | **1.00** | confident; 92.0% kept (356,711 removed) |
| sceneC @ percentile 2.0 | `−0.4930` | `+0.7024` | `+0.6342` | **1.00** | confident; 89.7% kept |
| sceneA | `−1.3010` | `+1.0228` | `+1.2568` | **0.09** | `[LOW CONFIDENCE]`, fallback |

> ⚠ **`prominence` is the number to check, not the cut.** sceneA has no distinct ceiling plane, and
> the detector says so rather than confidently cutting the wrong thing. Below `min_prominence`
> (0.25) it falls back to a cut at 90% of the height range, which is a framing convenience and
> **not a detection**. `describe()` prints `[LOW CONFIDENCE]` and the GUI shows a warning.

Two further guards force that same fallback:

| guard | why |
|---|---|
| cut must clear `max(camera height) + 5% of span` | the cameras were *inside* the room; a cut below them slices the room rather than lifting its roof |
| cut must sit above `floor + 30% of span` | a cut near the floor is never the right answer |

sceneC's camera heights in room coordinates are `[−0.189, −0.217, −0.264, −0.221, −0.224, −0.209]`
against a cut of `+0.622` at the same percentile — a comfortable margin, and the guard never fires. It exists for the
degenerate solves where the ceiling spike is really the top of a wall.

> **The clip-versus-range detail is load-bearing.** An early version built the histogram from
> `np.clip(h, lo, hi)`, which piles every floater into the two edge bins and invents a spike
> exactly where the ceiling search looks. `np.histogram`'s `range=` already drops the tails; use
> that and never clip.

---

## 5. Defect: spherical harmonics read in the wrong order

### Symptom

A viewer written against the generic 3DGS convention renders this repo's PLYs with colour
artefacts that grow with viewing angle — mild head-on, obvious from an isometric camera.
`../G4Splat/scripts/view_gaussians.py:109` has the defect.

### Cause

There are two conventions for flattening `f_rest`, and this repo writes the less common one.
`gaussian_model.py:476`:

```python
f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1)
```

`_features_rest` is `[N, K−1, 3]`, so the transpose makes it `[N, 3, K−1]` — **channel-major**:
`f_rest_0..7` are the red channel's coefficients 1–8, `f_rest_8..15` green, `f_rest_16..23` blue.
`load_ply` at `gaussian_model.py:520` mirrors it with `reshape(N, 3, K²−1).transpose(1, 2)`.

The G4Splat loader instead does `reshape(len(xyz), -1, 3)`, which is coefficient-major. That
scrambles SH bands across colour channels: every coefficient is present, each attached to the
wrong band and the wrong channel.

### Fix

`view_gs_web.py:load_gaussians` uses `reshape(n, 3, -1).transpose(0, 2, 1)`, matching the writer.
sceneC's PLY carries 24 `f_rest` properties → 8 non-DC coefficients per channel → 9 total → SH
degree **2**, which matches `--sh_degree 2` in the pipeline defaults.

> **`--sh_degree 0` is worth trying anyway.** An isometric camera sits far outside the distribution
> of viewing directions the model was ever supervised on, so the higher bands extrapolate rather
> than interpolate. Truncating to the DC term is flat but stable.

---

## 6. Defect: `look_at` handedness

### Symptom

Every isometric render came out rotated 180° — the room read as a box viewed from underneath, with
the floor at the top of the frame.

### Cause

`look_at_c2w` built the OpenCV basis `(right, down, forward)` as `right = fwd × (−up)`, reasoning
that OpenCV's y axis points down so the world down vector is the one to use. It is not: the basis
is right-handed about `forward`, so `right = fwd × up`.

The check: facing `+x` with `+z` up, `x × z = −y`, which is the right hand. The buggy form gives
`+y` (left) and then `down = fwd × right = +z` (up). Both axes flip, which composes to a 180°
rotation about the view direction rather than a mirror — which is why it looked like a plausible
camera pointing the wrong way rather than obviously reversed text.

### Fix

`right = np.cross(fwd, up)`, with the degenerate branch (looking straight along up) falling back
to an arbitrary perpendicular. Confirmed visually and by the parallel-projection test in §7.

---

## 7. Orthographic projection

The view is genuinely orthographic — gsplat's `camera_model="ortho"` — not a long-focal
perspective approximation.

### How gsplat's ortho intrinsics work

`gsplat/cuda/_torch_impl.py:169` `_ortho_proj`:

```python
means2d = means[..., :2] * Ks[..., [0, 1], [0, 1]] + Ks[..., [0, 1], [2, 2]]
```

No depth divide. `fx`/`fy` are simply **pixels per world unit** and `cx`/`cy` the principal point,
so `ortho_K(W, H, world_height)` sets `f = H / world_height` with square pixels and lets the
horizontal coverage follow from the aspect ratio. `CameraModel` is
`Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]` (`cuda/_wrapper.py:37`).

### Keeping orbit controls

`perspective_to_ortho` takes the pose the browser reports and changes only the projection: the
world height is read off the client's fov and its distance to the target, so zooming the orbit
still zooms. The camera is additionally pushed back by `2 × room diagonal` so nothing lands behind
gsplat's near plane (default 0.01) — under a parallel projection that shifts nothing in the image.

### Verification

Two room edges of equal world length at different depths, projected at the same pose:

| projection | near edge | far edge | ratio |
|---|---|---|---|
| perspective (fov 50°) | 139.8 px | 142.4 px | 0.9819 |
| **ortho** | 139.5 px | 139.5 px | **1.0000** |

Parallel to four decimal places. The perspective ratio is close to 1 only because the test camera
sits at twice the scene diagonal; the point is that ortho is exact.

---

## 8. Near walls occlude too

Removing the ceiling is necessary but not sufficient. Walls are reconstructed on their **inside**
faces, and a 3D Gaussian looks the same from behind, so an exterior isometric view still meets two
blank near walls.

`--wall_frac` drops a slab off whichever one or two walls face the camera, recomputed as the
azimuth changes. Walls more than ~81° off the view direction (`|cos| < 0.15`) are edge-on and
skipped — there is nothing meaningful to remove.

**Thickness is a fraction of room *height*, not of each horizontal axis.** Floor plans are not
square — sceneC is 2.97 × 1.46 at percentile 0.5 — so an axis-relative slab would bite twice as
deep off the long wall as the short one. Room height is the one dimension that reliably tracks
physical scale.

sceneC at percentile 3.0, azimuth 45°:

| `wall_frac` | gaussians kept | reads as |
|---|---|---|
| 0.00 | 3,933,769 | ceiling gone, near walls still block the room |
| 0.06 | 2,795,578 | interior visible; bed, curtains, doorway |
| 0.12 | 2,548,155 | **clearest** — full dollhouse |
| 0.20 | 2,092,983 | over-cut; walls mostly gone |

0.06–0.12 reads well. **It defaults to off** because it also removes whatever furniture stands
against those walls, which is a blunt trade the viewer should make deliberately rather than
inherit.

---

## 9. Room box tightness and floaters

Sparse-view indoor 3DGS parks floaters outside the room, mostly streaming out through windows and
doorways — the places where the cameras had unconstrained depth. On sceneC they form a visible
tail that stretches the robust box along its long axis:

| percentile | room box `(u1, u2, h)` | diagonal |
|---|---|---|
| 0.5 | `[2.974, 1.464, 1.481]` | 3.630 |
| 2.0 | `[1.431, 1.737, 1.427]` | 2.665 |
| 3.0 | `[1.685, 1.413, 1.412]` | — |

Between 0.5 and 2.0 the long axis nearly halves. Since the box drives both the floater crop and
the isometric framing (`fit_ortho_height` fits its corners), at 0.5 the room ends up small in a
frame mostly occupied by junk. The default is therefore **2.0**, not the 0.5 that would be right
for a clean scene.

> The `u1`/`u2` swap between 0.5 and 2.0 is not a bug. Dropping the tail changes which rotation
> minimises the bounding rectangle, so the min-area search picks the equivalent basis rotated by
> 90°. Nothing downstream depends on which horizontal axis is which.

How tight to draw the box is a judgement about a particular reconstruction, so it is a slider.
Making it live needs percentiles cheaper than a re-sort: `prepare_box_sampler` sorts each axis once
at load and stores the cumulative opacity weights, after which `set_box` is three binary searches.
Cost is 3 × N × (4 + 4) bytes — 107 MB for sceneC's 4.45M splats.

---

## 10. Environment

The viewer is **not** part of the pipeline's dependency set. `pyproject.toml` carries it as
`[project.optional-dependencies] viewer`; install with `uv sync --extra viewer`. Nothing in the
pipeline imports viser or nerfview, and gsplat — which does the actual rasterizing — is already a
hard dependency. `--screenshot` needs neither, so a missing viser is a clear message pointing at
that flag rather than a traceback.

Verified against viser **1.1.0**, nerfview **0.1.3**, gsplat **1.5.3**, torch **2.10.0+cu128** on
an RTX 5090 (sm_120).

Two version-specific details:

- **nerfview 0.1.3 accepts the `(camera_state, img_wh)` render_fn contract** (`_renderer.py:164`),
  with a deprecation warning path for the newer `RenderTabState` signature. `CameraState` carries
  `fov`, `aspect`, `c2w` and a `get_K(img_wh)` helper; `Viewer.get_camera_state` builds `c2w`
  straight from `camera.wxyz`/`camera.position`, so the pose is OpenCV and feeds gsplat's
  `viewmats` after one inverse.
- **gsplat's `backgrounds` wants a bare `[channels]`, not the documented `[C, channels]`.** Under
  the default `packed=True`, `image_dims = means2d.shape[:-2]` collapses to `()` because `means2d`
  is `[nnz, 2]` (`cuda/_wrapper.py:868`), so the assert at :884 demands `(3,)`. Passing `(1, 3)`
  raises `AssertionError`. Confirmed empirically both ways.

Loading sceneC's 697 MB / 4.45M-splat PLY takes ~2.0 s and the resident tensors are ~680 MB at SH
degree 2. `--max_gaussians` random-subsamples for a contended GPU; `--sh_degree 0` cuts the colour
tensor by 9×.

---

## 11. What this stage does not do

- **It does not modify the model.** Nothing is written back to any PLY. If a cut model is wanted
  as an artifact, that is a separate export not currently implemented.
- **It is not a multi-storey solution.** The detector finds *one* ceiling. A two-floor scan would
  need per-storey segmentation along the up axis first.
- **It does not separate wall from furniture.** §8's slab is geometric and takes both. Doing better
  needs either anisotropic splats (§2) or a plane-fitting pass.
- **It has no bearing on reconstruction quality.** Every artefact visible in a dollhouse render was
  already in the model; the isometric view mostly makes sparse-view failures easier to see, which
  is a large part of why it is useful.
