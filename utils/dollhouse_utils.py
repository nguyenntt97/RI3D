"""Dollhouse post-processing for indoor 3DGS: strip the ceiling so an isometric
view can actually see into the room.

An indoor scan is photographed from *inside*, so every Gaussian that ends up on
the ceiling sits between an elevated camera and the room. Point a viewer at the
scene from above and you render the underside of a roof. This module removes
that occluder geometrically and builds the isometric camera to look through the
hole it leaves.

Everything here is a *view-time* filter -- nothing is written back to the PLY --
so the cut can be dragged live in a viewer and reset without reloading.

Why geometric and not backface culling
--------------------------------------
The obvious dollhouse trick is to cull splats whose normal faces away from the
camera: a ceiling fitted from inside points its normal down, so an elevated
camera would drop it for free (and the near walls with it). A Gaussian's normal
is its shortest scale axis, so that needs anisotropic splats.

**Stage 1c's model does not have them.** `GaussianModel.__init__`
(`scene/gaussian_model.py:51`) takes a `spherical_gaussians` flag, and
`get_scaling` (`:142`) returns channel 0 repeated three times when it is set.
`save_ply` (`:477`) writes `scaling_inverse_activation(get_scaling)` rather than
the raw parameter, so the choice is baked into the file. Stages 1b, 2a, 2b and
1c all pass the flag; measured on `sceneC_6`, all 4,454,882 splats have
`scale_0 == scale_1 == scale_2` and `rot = (1,0,0,0)`. No shortest axis, no
orientation, nothing to read a normal off -- so the cut is done in room
coordinates instead.

Stages 1a and 5a/5b do *not* pass the flag and their PLYs are genuinely
anisotropic (sceneA's 5b export: min/max scale ratio p50 0.805), so a normal
test would work there. It is not implemented, and if it ever is it has to stay
guarded: run against a 1b/1c model it silently removes nothing.

The room frame
--------------
`build_room_frame` fixes an orthonormal basis (e1, e2, up) with e1/e2 aligned to
the walls, found by minimising the area of the horizontal bounding rectangle over
rotations -- rectangular rooms have a sharp minimum there. Heights, the ceiling
cut and the wall slabs are all expressed in that frame, so they mean the same
thing regardless of how the SfM gauge happened to land.

Standalone diagnosis:

    python utils/dollhouse_utils.py -m output/gs_gsfix/sceneC_6 \
        -c output/sceneC/ggpt_sfm/cameras.json
"""

import json
import math
import os
from dataclasses import dataclass, field
from glob import glob

import numpy as np


# --------------------------------------------------------------------------- #
# model location
# --------------------------------------------------------------------------- #

def latest_gs_ply(path):
    """Resolve a PLY from a file path or a 3DGS model directory.

    Iteration counts are not fixed in this repo (`--iterations` defaults to
    10_000, not upstream's 30_000), so a model directory is scanned for the
    highest `point_cloud/iteration_*` rather than assuming a number.
    """
    if os.path.isfile(path):
        return path
    pc_dir = os.path.join(path, "point_cloud")
    iters = []
    if os.path.isdir(pc_dir):
        for d in os.listdir(pc_dir):
            ply = os.path.join(pc_dir, d, "point_cloud.ply")
            if d.startswith("iteration_") and os.path.isfile(ply):
                try:
                    iters.append((int(d.split("_")[-1]), ply))
                except ValueError:
                    continue
    if iters:
        return max(iters)[1]
    loose = sorted(glob(os.path.join(path, "*.ply")))
    if loose:
        return loose[0]
    raise FileNotFoundError(
        f"No PLY under {path} (looked for point_cloud/iteration_*/point_cloud.ply)"
    )


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #

def weighted_percentile(x, w, q):
    """Percentiles of `x` weighted by `w`. `q` in [0, 100], scalar or sequence."""
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w)
    total = cum[-1]
    if total <= 0:
        return np.percentile(x, q)
    # midpoint rule, so a single dominant weight does not pin the result to a bin edge
    pos = (cum - 0.5 * w) / total * 100.0
    return np.interp(np.asarray(q, dtype=np.float64), pos, x)


def _smooth(y, k):
    """Box filter of width `k` with edge replication."""
    if k <= 1:
        return y
    pad = k // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(k) / k, mode="valid")


# --------------------------------------------------------------------------- #
# up axis
# --------------------------------------------------------------------------- #

def estimate_up(cameras_json=None, means=None, override=None):
    """World up as a unit vector, plus a string saying where it came from.

    `override` accepts `x`, `-y`, ... or three comma-separated floats.

    The camera path is the reliable one. `cameras.json` stores OpenCV c2w, whose
    second column is the camera's *down* axis; indoor photographs are shot close
    to level, so averaging those and negating recovers world up. On sceneC the
    six view directions agree to a pairwise dot of 0.998.

    Without cameras there is nothing in a Gaussian cloud that identifies up
    (this repo's splats are isotropic and unoriented), so the fallback only picks
    the axis whose extent is smallest -- rooms are wider than they are tall.
    That is a guess; pass `--up` if it looks wrong.
    """
    if override:
        named = {
            "x": (1, 0, 0), "-x": (-1, 0, 0),
            "y": (0, 1, 0), "-y": (0, -1, 0),
            "z": (0, 0, 1), "-z": (0, 0, -1),
        }
        key = override.strip().lower()
        if key in named:
            v = np.array(named[key], dtype=np.float64)
        else:
            v = np.array([float(t) for t in override.split(",")], dtype=np.float64)
            if v.shape != (3,):
                raise ValueError(f"--up wants an axis name or 3 floats, got {override!r}")
        return v / np.linalg.norm(v), f"override {override}"

    if cameras_json and os.path.isfile(cameras_json):
        with open(cameras_json) as f:
            cams = json.load(f)
        c2w = np.asarray(cams["cams2world"], dtype=np.float64)
        down = c2w[:, :3, 1]
        down /= np.linalg.norm(down, axis=1, keepdims=True)
        agree = float((down @ down.T).min())
        up = -down.mean(0)
        up /= np.linalg.norm(up)
        return up, f"cameras.json ({len(c2w)} views, min pairwise dot {agree:.3f})"

    if means is not None:
        spread = means.max(0) - means.min(0)
        axis = int(np.argmin(spread))
        up = np.zeros(3)
        up[axis] = -1.0  # SfM worlds here are y-down; guess the negative sense
        return up, f"guessed from extent (thinnest axis {'xyz'[axis]}) -- verify this"

    raise ValueError("estimate_up needs cameras_json, means, or override")


# --------------------------------------------------------------------------- #
# room frame
# --------------------------------------------------------------------------- #

@dataclass
class RoomFrame:
    """Orthonormal room basis and the robust box of the scene inside it."""
    center: np.ndarray          # (3,) world
    up: np.ndarray              # (3,) world, unit
    e1: np.ndarray              # (3,) world, horizontal, along a wall
    e2: np.ndarray              # (3,) world, horizontal, perpendicular to e1
    lo: np.ndarray              # (3,) robust min in (e1, e2, up) coords, centred
    hi: np.ndarray              # (3,) robust max
    theta: float = 0.0          # wall alignment angle, radians
    percentile: float = 0.5
    _axis_cdf: list = field(default=None, repr=False)

    def prepare_box_sampler(self, coords, weights):
        """Cache per-axis sorted coordinates so `set_box` is cheap.

        Sparse-view indoor 3DGS parks floaters outside the room -- most of them
        streaming out through windows and doorways, which is where the cameras
        had unconstrained depth. On sceneC those stretch the robust box from
        1.69 to 2.97 along its long axis, and since the box drives both the crop
        and the isometric framing, the room ends up small in a frame mostly
        occupied by junk. How tight to draw the box is a judgement call about a
        particular reconstruction, so it wants to be a slider rather than a
        constant -- and a slider needs percentiles cheaper than a re-sort.
        """
        self._axis_cdf = []
        for i in range(3):
            order = np.argsort(coords[:, i])
            v = np.ascontiguousarray(coords[order, i], dtype=np.float32)
            cw = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
            self._axis_cdf.append((v, (cw / cw[-1]).astype(np.float32)))

    def set_box(self, percentile):
        """Redraw the robust box at `percentile`. Leaves the basis untouched."""
        if not self._axis_cdf:
            return self.lo, self.hi
        p = np.clip(percentile, 0.0, 45.0) / 100.0
        lo, hi = np.empty(3), np.empty(3)
        for i, (v, cw) in enumerate(self._axis_cdf):
            lo[i] = v[min(int(np.searchsorted(cw, p)), len(v) - 1)]
            hi[i] = v[min(int(np.searchsorted(cw, 1.0 - p)), len(v) - 1)]
        self.lo, self.hi, self.percentile = lo, hi, float(percentile)
        return lo, hi

    @property
    def target(self):
        """World-space centre of the current box -- what a view should look at.

        Distinct from `center`, which is pinned to the box the frame was built
        with so that cached room coordinates stay valid when the box is redrawn.
        """
        return self.center + (0.5 * (self.lo + self.hi)) @ self.basis

    @property
    def basis(self):
        """(3,3) whose rows map a world offset to (u1, u2, h)."""
        return np.stack([self.e1, self.e2, self.up], axis=0)

    @property
    def size(self):
        return self.hi - self.lo

    @property
    def diagonal(self):
        return float(np.linalg.norm(self.size))

    def to_frame(self, xyz):
        """World points -> centred room coordinates (u1, u2, h)."""
        return (np.asarray(xyz) - self.center) @ self.basis.T

    def horizontal_dir(self, azimuth_deg):
        """Unit world vector at `azimuth_deg` in the horizontal plane."""
        a = math.radians(azimuth_deg)
        return math.cos(a) * self.e1 + math.sin(a) * self.e2


def build_room_frame(means, weights, up, percentile=0.5, n_angles=90, max_points=400_000):
    """Wall-aligned room frame.

    The horizontal axes are chosen by minimising the area of the robust bounding
    rectangle over rotations in [0, 90) degrees. A rectangular room has a sharp
    minimum when the rectangle sits square to the walls; a round or cluttered
    room has a shallow one, which costs nothing because any horizontal basis is
    then as good as another.
    """
    means = np.asarray(means, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)

    # any horizontal seed, taken from the world axis least parallel to up
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(up)))] = 1.0
    a1 = seed - up * (seed @ up)
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(up, a1)

    if len(means) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(means), max_points, replace=False)
        pts, wts = means[sel], np.asarray(weights, dtype=np.float64)[sel]
    else:
        pts, wts = means, np.asarray(weights, dtype=np.float64)

    p1, p2 = pts @ a1, pts @ a2
    q = (percentile, 100.0 - percentile)
    best = (np.inf, 0.0)
    for theta in np.linspace(0.0, np.pi / 2, n_angles, endpoint=False):
        c, s = math.cos(theta), math.sin(theta)
        r1, r2 = c * p1 + s * p2, -s * p1 + c * p2
        lo1, hi1 = weighted_percentile(r1, wts, q)
        lo2, hi2 = weighted_percentile(r2, wts, q)
        area = (hi1 - lo1) * (hi2 - lo2)
        if area < best[0]:
            best = (area, theta)
    theta = best[1]
    c, s = math.cos(theta), math.sin(theta)
    e1, e2 = c * a1 + s * a2, -s * a1 + c * a2

    coords = np.stack([means @ e1, means @ e2, means @ up], axis=1)
    w = np.asarray(weights, dtype=np.float64)
    lo = np.array([weighted_percentile(coords[:, i], w, percentile) for i in range(3)])
    hi = np.array([weighted_percentile(coords[:, i], w, 100.0 - percentile) for i in range(3)])
    mid = 0.5 * (lo + hi)
    center = mid[0] * e1 + mid[1] * e2 + mid[2] * up
    return RoomFrame(center=center, up=up, e1=e1, e2=e2,
                     lo=lo - mid, hi=hi - mid, theta=float(theta), percentile=percentile)


# --------------------------------------------------------------------------- #
# ceiling detection
# --------------------------------------------------------------------------- #

@dataclass
class Levels:
    """Where the floor and ceiling sit, and where to cut. Room-frame heights."""
    floor: float
    ceiling: float
    cut: float
    prominence: float
    confident: bool
    note: str
    hist: np.ndarray = field(default=None, repr=False)
    centers: np.ndarray = field(default=None, repr=False)

    def describe(self):
        flag = "" if self.confident else "  [LOW CONFIDENCE]"
        return (f"floor {self.floor:+.4f}   ceiling {self.ceiling:+.4f}   "
                f"cut {self.cut:+.4f}   prominence {self.prominence:.2f}{flag}\n"
                f"    {self.note}")


def detect_levels(heights, weights, cam_heights=None, bins=192, smooth=5,
                  margin_frac=0.02, min_prominence=0.25, base_frac=0.2):
    """Find the ceiling from an opacity-weighted height histogram.

    A ceiling is a horizontal plane, so it lands in a narrow band of heights and
    shows up as a spike near the top of the distribution -- distinct from the
    flat plateau that walls and furniture make through the middle of the room.
    On sceneC the plateau sits at ~0.18 of peak density and the ceiling spike
    reaches 0.36, against a floor spike of 1.00.

    The cut is placed at the *base* of that spike rather than a fixed distance
    below its centre: walking down from the peak to half prominence adapts to how
    thick the reconstructed slab is, which varies a lot with how much of the
    ceiling the cameras actually saw.

    Returns `Levels`. `confident` is False when no spike stands out, which is the
    honest answer for an open-plan or outdoor scene; a fallback cut near the top
    of the range is still returned so the caller always has something to use.
    """
    h = np.asarray(heights, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    lo, hi = weighted_percentile(h, w, (0.5, 99.5))
    span = float(hi - lo)
    if span <= 0:
        return Levels(lo, hi, hi, 0.0, False, "degenerate height range")

    # no clipping: np.histogram's `range` already drops the tails, whereas
    # clipping into it would pile every floater into the edge bins and invent a
    # spike exactly where the ceiling search looks
    hist, edges = np.histogram(h, bins=bins, range=(lo, hi), weights=w)
    hist = _smooth(hist.astype(np.float64), smooth)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    low = ctr < lo + 0.35 * span
    floor = float(ctr[low][int(np.argmax(hist[low]))]) if low.any() else float(lo)

    # plateau level: the room's own walls and contents, away from both slabs
    midband = (ctr > floor + 0.08 * span) & (ctr < hi - 0.15 * span)
    mid = float(np.median(hist[midband])) if midband.sum() >= 4 else float(np.median(hist))
    mid = max(mid, 1e-9)

    top = ctr > hi - 0.25 * span
    ti = np.flatnonzero(top)
    ci = int(ti[np.argmax(hist[ti])])
    peak = float(hist[ci])
    ceiling = float(ctr[ci])
    prominence = (peak - mid) / mid

    # Walk down the near side of the spike to where density returns to the
    # room's own plateau. A fixed offset below the peak does not travel: a
    # ceiling the cameras barely saw reconstructs as a thin slab, one they saw
    # well as a broad band, and the cut has to clear whichever this is.
    thresh = mid + base_frac * (peak - mid)
    j = ci
    while j > 0 and hist[j] > thresh:
        j -= 1
    cut = float(ctr[j]) - margin_frac * span

    confident = prominence >= min_prominence
    note = (f"ceiling spike {prominence:.2f}x above the {mid:.3g} plateau; "
            f"cut at the base of the slab")
    if not confident:
        cut = float(hi - 0.10 * span)
        note = (f"no distinct ceiling spike (prominence {prominence:.2f} < "
                f"{min_prominence}); cut at 90% of the height range instead")

    # a cut below the cameras would slice the room, not the roof off it
    if cam_heights is not None and len(cam_heights):
        floor_of_cut = float(np.max(cam_heights)) + 0.05 * span
        if cut < floor_of_cut:
            cut = float(hi - 0.10 * span)
            confident = False
            note = (f"detected cut fell below the highest camera "
                    f"({np.max(cam_heights):+.4f}); fell back to 90% of the range")
    if cut < floor + 0.30 * span:
        cut = float(hi - 0.10 * span)
        confident = False
        note = "detected cut fell too close to the floor; fell back to 90% of the range"

    return Levels(floor=floor, ceiling=ceiling, cut=cut, prominence=float(prominence),
                  confident=confident, note=note, hist=hist / max(hist.max(), 1e-12),
                  centers=ctr)


def histogram_ascii(levels, width=54, rows=28):
    """Compact text rendering of the height histogram with the cut marked."""
    if levels.hist is None:
        return ""
    hist, ctr = levels.hist, levels.centers
    step = max(1, len(hist) // rows)
    out = []
    for i in range(0, len(hist), step):
        v = float(hist[i:i + step].max())
        c = float(ctr[i:i + step].mean())
        mark = ""
        if abs(c - levels.cut) < (ctr[1] - ctr[0]) * step:
            mark = "  <== CUT"
        elif abs(c - levels.ceiling) < (ctr[1] - ctr[0]) * step:
            mark = "  <-- ceiling"
        elif abs(c - levels.floor) < (ctr[1] - ctr[0]) * step:
            mark = "  <-- floor"
        out.append(f"  {c:+8.4f} |{'#' * int(v * width):<{width}}|{mark}")
    return "\n".join(out[::-1])  # tall end first, the way a room reads


# --------------------------------------------------------------------------- #
# the cut itself
# --------------------------------------------------------------------------- #

def dollhouse_mask(coords, frame, cut=None, wall_frac=0.0, azimuth_deg=None,
                   crop=True, crop_pad=0.02):
    """Boolean keep-mask over Gaussians, in room coordinates.

    coords : (N, 3) room-frame (u1, u2, h), i.e. `frame.to_frame(means)`.
    cut    : drop everything above this height. None keeps the ceiling.
    wall_frac / azimuth_deg : additionally drop a slab of that fractional
        thickness off the one or two walls facing a camera at `azimuth_deg`.
        A room scanned from inside has walls reconstructed on their *inside*
        faces, so an exterior isometric view is blocked by the near walls even
        after the ceiling is gone. This is the blunt fix -- it also takes out
        whatever furniture stands against those walls, which is why it is off by
        default and exposed as a slider.
    crop   : drop anything outside the robust room box, which is where sparse-view
        3DGS parks its floaters.
    """
    keep = np.ones(len(coords), dtype=bool)
    if cut is not None:
        keep &= coords[:, 2] <= cut
    if crop:
        pad = crop_pad * frame.size
        keep &= np.all(coords >= (frame.lo - pad), axis=1)
        keep &= np.all(coords <= (frame.hi + pad), axis=1)
    if wall_frac > 0 and azimuth_deg is not None:
        a = math.radians(azimuth_deg)
        # Thickness is a fraction of room *height*, not of each horizontal axis.
        # Floor plans are not square -- sceneC is 2.97 x 1.46 -- so an axis-relative
        # slab would bite twice as deep off the long wall as the short one. Room
        # height is the one dimension that reliably tracks physical scale.
        t = wall_frac * (frame.hi[2] - frame.lo[2])
        for axis, comp in ((0, math.cos(a)), (1, math.sin(a))):
            if abs(comp) < 0.15:      # edge-on wall: nothing meaningful to remove
                continue
            if comp > 0:
                keep &= coords[:, axis] <= frame.hi[axis] - t
            else:
                keep &= coords[:, axis] >= frame.lo[axis] + t
    return keep


# --------------------------------------------------------------------------- #
# isometric camera
# --------------------------------------------------------------------------- #

#: True isometric: the view direction makes equal angles with all three axes.
ISO_ELEVATION_DEG = math.degrees(math.atan(1.0 / math.sqrt(2.0)))  # 35.264...


def look_at_c2w(position, target, up):
    """OpenCV camera-to-world (x right, y down, z forward) looking at `target`."""
    position = np.asarray(position, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - position
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=np.float64)
    # OpenCV basis is (right, down, forward) and right-handed, so right = fwd x up:
    # facing +x under +z up gives right = x cross z = -y, which is the right hand.
    # Taking fwd x (-up) instead flips both axes, i.e. renders the room upside down.
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-8:                        # looking straight along up
        alt = np.zeros(3)
        alt[int(np.argmin(np.abs(up)))] = 1.0
        right = np.cross(fwd, alt)
        n = np.linalg.norm(right)
    right /= n
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, fwd
    c2w[:3, 3] = position
    return c2w


def isometric_c2w(frame, azimuth_deg=45.0, elevation_deg=ISO_ELEVATION_DEG,
                  distance=None, target=None):
    """Camera-to-world for an isometric view of the room.

    `distance` only sets where the camera sits; under an orthographic projection
    it does not change the framing, so the default is generous enough to keep the
    whole room in front of the near plane.
    """
    target = frame.target if target is None else np.asarray(target, dtype=np.float64)
    if distance is None:
        distance = 2.0 * frame.diagonal
    el = math.radians(elevation_deg)
    direction = math.cos(el) * frame.horizontal_dir(azimuth_deg) + math.sin(el) * frame.up
    return look_at_c2w(target + direction * distance, target, frame.up)


def ortho_K(width, height, world_height):
    """Intrinsics for gsplat's `camera_model="ortho"`.

    Under that model a point projects as `pixel = xy_cam * f + c` with no depth
    divide, so f is simply pixels per world unit. Square pixels, so the
    horizontal coverage follows from the aspect ratio.
    """
    f = height / max(world_height, 1e-9)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def fit_ortho_height(frame, c2w, aspect=1.0, margin=1.06):
    """World-space *vertical* extent that just contains the room from this pose.

    Both extents have to fit, and the horizontal one is measured against a frame
    that is `aspect` times wider, so it converts to a vertical requirement by
    dividing rather than being taken as-is.
    """
    corners = np.array([[x, y, z] for x in (frame.lo[0], frame.hi[0])
                        for y in (frame.lo[1], frame.hi[1])
                        for z in (frame.lo[2], frame.hi[2])])
    world = frame.center + corners @ frame.basis
    cam = (world - c2w[:3, 3]) @ c2w[:3, :3]     # world -> camera (R is orthonormal)
    need_h = cam[:, 1].max() - cam[:, 1].min()
    need_w = cam[:, 0].max() - cam[:, 0].min()
    return float(max(need_h, need_w / max(aspect, 1e-6)) * margin)


def perspective_to_ortho(c2w, fov, target, pullback):
    """Match an orthographic framing to a perspective camera at the target depth.

    Keeps the browser's orbit controls usable: the pose still comes from the
    client, only the projection changes. The camera is additionally pushed back
    so nothing lands behind gsplat's near plane -- under an orthographic
    projection that shifts nothing in the image.
    """
    pos, fwd = c2w[:3, 3], c2w[:3, 2]
    dist = max(float((np.asarray(target) - pos) @ fwd), 1e-3)
    world_height = 2.0 * dist * math.tan(fov / 2.0)
    out = c2w.copy()
    out[:3, 3] = pos - fwd * pullback
    return out, world_height


# --------------------------------------------------------------------------- #
# CLI: diagnose a model without opening a browser
# --------------------------------------------------------------------------- #

def _main():
    import argparse
    from plyfile import PlyData

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-m", "--model", required=True, help="3DGS PLY or model directory")
    ap.add_argument("-c", "--cameras", default=None, help="cameras.json for up-axis estimation")
    ap.add_argument("--up", default=None, help="override: x|-x|y|-y|z|-z or 'a,b,c'")
    ap.add_argument("--opacity_min", type=float, default=0.1)
    ap.add_argument("--percentile", type=float, default=0.5)
    args = ap.parse_args()

    ply = latest_gs_ply(args.model)
    print(f"[i] {ply}")
    v = PlyData.read(ply)["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    print(f"[i] {len(means):,} gaussians")

    sel = opac > args.opacity_min
    print(f"[i] {sel.sum():,} above opacity {args.opacity_min}")
    up, src = estimate_up(args.cameras, means=means[sel], override=args.up)
    print(f"[i] up = {np.round(up, 4)}   from {src}")

    frame = build_room_frame(means[sel], opac[sel], up, percentile=args.percentile)
    print(f"[i] room {np.round(frame.size, 4)} (u1, u2, h), diagonal {frame.diagonal:.4f}, "
          f"walls rotated {math.degrees(frame.theta):.1f}deg")

    cam_h = None
    if args.cameras and os.path.isfile(args.cameras):
        with open(args.cameras) as f:
            c2w = np.asarray(json.load(f)["cams2world"], dtype=np.float64)
        cam_h = (c2w[:, :3, 3] - frame.center) @ up
        print(f"[i] camera heights {np.round(cam_h, 4)}")

    coords = frame.to_frame(means)
    lv = detect_levels(coords[sel][:, 2], opac[sel], cam_heights=cam_h)
    print(f"[i] {lv.describe()}")
    print(histogram_ascii(lv))

    keep = dollhouse_mask(coords, frame, cut=lv.cut)
    print(f"[i] dollhouse keeps {keep.sum():,} / {len(keep):,} "
          f"({100.0 * keep.mean():.1f}%) -- removed {(~keep).sum():,}")


if __name__ == "__main__":
    _main()
