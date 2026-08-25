#!/usr/bin/env python3
"""Web viewer for indoor 3DGS with an isometric dollhouse cut.

Renders a stage-1c (or any) 3DGS model in a browser through viser + nerfview +
gsplat, and adds the post-processing an *indoor* scan needs before an isometric
view shows anything useful. A room is photographed from inside, so both the
ceiling and the near walls sit between an elevated camera and the room. Both are
detected and removed at view time: the ceiling once, the walls per viewing angle,
so orbiting always looks into the room rather than at the back of an occluder.

Each wall is cut back by its *own* measured thickness -- the four do not
reconstruct alike -- and a wall the detector cannot find is left standing rather
than guessed at. `--wall_scale` scales the cut, `--no_wall_cut` disables it.

The projection is genuinely orthographic -- gsplat's `camera_model="ortho"` --
rather than a long-lens perspective fake, so parallel edges stay parallel and the
result reads as an isometric drawing. Orbit controls still work; only the
projection changes.

Interactive:

    python scripts/view_gs_web.py -m output/gs_gsfix/sceneC_6 \\
        -c output/sceneC/ggpt_sfm/cameras.json --port 8080

Headless contact sheet (no browser, writes PNGs):

    python scripts/view_gs_web.py -m output/gs_gsfix/sceneC_6 \\
        -c output/sceneC/ggpt_sfm/cameras.json --screenshot out/iso

Falls back to the stage-1b model when 1c has not been run -- both are plain 3DGS
PLYs and nothing here depends on which stage produced it.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dollhouse_utils import (  # noqa: E402
    ISO_ELEVATION_DEG, build_room_frame, detect_levels, detect_wall_planes,
    estimate_up, fit_ortho_height, histogram_ascii, isometric_c2w,
    latest_gs_ply, ortho_K, perspective_to_ortho, wall_bound,
)

try:
    from gsplat.rendering import rasterization
except ImportError as e:
    sys.exit(f"[x] gsplat is required: {e}")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_gaussians(ply_path, device="cuda", sh_degree=None, max_gaussians=None):
    """3DGS PLY -> gsplat-ready tensors.

    `sh_degree` truncates the spherical harmonics. That is worth reaching for
    here: an isometric camera sits far outside the distribution of viewing
    directions the model was ever supervised on, so the higher SH bands
    extrapolate into colour artefacts. Degree 0 is flat but honest.
    """
    v = PlyData.read(str(ply_path))["vertex"]
    names = {p.name for p in v.properties}
    n = len(v["x"])

    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opac = np.asarray(v["opacity"], dtype=np.float32)
    if opac.min() < 0.0 or opac.max() > 1.0:
        opac = 1.0 / (1.0 + np.exp(-opac))

    scale_names = sorted((p.name for p in v.properties if p.name.startswith("scale_")),
                         key=lambda s: int(s.split("_")[-1]))
    scales = np.exp(np.stack([v[s] for s in scale_names], axis=1).astype(np.float32))
    if scales.shape[1] == 2:                       # 2DGS: give the third axis a sliver
        scales = np.concatenate([scales, np.full((n, 1), 1e-4, np.float32)], axis=1)

    rot_names = sorted((p.name for p in v.properties if p.name.startswith("rot_")),
                       key=lambda s: int(s.split("_")[-1]))
    if len(rot_names) == 4:
        quats = np.stack([v[r] for r in rot_names], axis=1).astype(np.float32)
    else:
        quats = np.tile(np.array([[1, 0, 0, 0]], np.float32), (n, 1))

    dc = sorted((p.name for p in v.properties if p.name.startswith("f_dc_")),
                key=lambda s: int(s.split("_")[-1]))
    rest = sorted((p.name for p in v.properties if p.name.startswith("f_rest_")),
                  key=lambda s: int(s.split("_")[-1]))
    if len(dc) == 3:
        sh0 = np.stack([v[c] for c in dc], axis=1).astype(np.float32)[:, None, :]
        if rest:
            # PLY stores f_rest channel-major: all R coefficients, then G, then B
            shN = np.stack([v[c] for c in rest], axis=1).astype(np.float32)
            shN = shN.reshape(n, 3, -1).transpose(0, 2, 1)
            colors = np.concatenate([sh0, shN], axis=1)
        else:
            colors = sh0
        file_degree = int(round(math.sqrt(colors.shape[1]))) - 1
        if sh_degree is not None and sh_degree < file_degree:
            colors = colors[:, : (sh_degree + 1) ** 2, :]
            file_degree = sh_degree
        active_degree = file_degree
    elif {"red", "green", "blue"} <= names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        colors = np.clip(rgb / 255.0 if rgb.max() > 1.0 else rgb, 0.0, 1.0)
        active_degree = None
    else:
        colors = np.full((n, 3), 0.8, np.float32)
        active_degree = None

    if max_gaussians and n > max_gaussians:
        sel = np.random.default_rng(0).choice(n, max_gaussians, replace=False)
        means, opac, scales, quats, colors = (a[sel] for a in
                                              (means, opac, scales, quats, colors))
        print(f"[i] subsampled {n:,} -> {max_gaussians:,} gaussians")

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return {
        "means": t(means), "opacities": t(opac), "scales": t(scales),
        "quats": F.normalize(t(quats), p=2, dim=-1), "colors": t(colors),
        "sh_degree": active_degree,
        "means_np": means, "opac_np": opac,
    }


# --------------------------------------------------------------------------- #
# filter state
# --------------------------------------------------------------------------- #

class Dollhouse:
    """Holds the model, the room frame, and the currently applied cut."""

    MODES = ("dollhouse", "removed only", "everything")

    def __init__(self, gs, frame, levels, walls=(), background=(1.0, 1.0, 1.0),
                 device="cuda"):
        self.gs, self.frame, self.levels, self.device = gs, frame, levels, device
        self.walls = list(walls)
        # gsplat 1.5.3 documents `backgrounds` as [C, channels], but under the
        # default packed=True path `image_dims` collapses to () (cuda/_wrapper.py:867),
        # so the assert actually wants a bare [channels].
        self.background = torch.tensor(background, dtype=torch.float32, device=device)
        basis = torch.from_numpy(frame.basis.astype(np.float32)).to(device)
        center = torch.from_numpy(frame.center.astype(np.float32)).to(device)
        self.coords = (gs["means"] - center) @ basis.T          # room coords, on GPU
        self.coords_np = frame.to_frame(gs["means_np"])
        self.cut = levels.cut
        self.box_percentile = frame.percentile
        self.wall_cut = True
        self.wall_scale = 1.0
        self.wall_frac = 0.0          # legacy absolute override; 0 = use detection
        self.azimuth = 45.0
        self.opacity_min = 0.0
        self.crop = True
        self.mode = "dollhouse"
        self.view = {}
        self.rebuild()

    def rebuild(self):
        """Recompute the keep set. Called on GUI change, not per frame."""
        gs, f = self.gs, self.frame
        f.set_box(self.box_percentile)
        lo = torch.from_numpy(f.lo.astype(np.float32)).to(self.device)
        hi = torch.from_numpy(f.hi.astype(np.float32)).to(self.device)
        keep = torch.ones(len(self.coords), dtype=torch.bool, device=self.device)

        if self.mode != "everything":
            keep &= self.coords[:, 2] <= self.cut
            if self.crop:
                pad = 0.02 * (hi - lo)
                keep &= ((self.coords >= (lo - pad)) & (self.coords <= (hi + pad))).all(1)
            if self.wall_cut:
                # Same boundaries as dollhouse_mask, through the same helper, so
                # the GPU path here and the numpy one cannot drift apart.
                a = math.radians(self.azimuth)
                for ax, comp in ((0, math.cos(a)), (1, math.sin(a))):
                    if abs(comp) < 0.15:      # edge-on wall: nothing to remove
                        continue
                    side = 1 if comp > 0 else -1
                    b = wall_bound(f, ax, side, self.walls, self.wall_scale,
                                   self.wall_frac)
                    if b is None:             # low-confidence wall: leave it up
                        continue
                    keep &= (self.coords[:, ax] <= b) if side > 0 \
                        else (self.coords[:, ax] >= b)
        if self.opacity_min > 0:
            keep &= gs["opacities"] > self.opacity_min
        if self.mode == "removed only":
            keep = ~keep
            if self.opacity_min > 0:
                keep &= gs["opacities"] > self.opacity_min

        idx = torch.nonzero(keep, as_tuple=True)[0]
        self.view = {k: gs[k][idx] for k in
                     ("means", "quats", "scales", "opacities", "colors")}
        self.kept, self.total = int(idx.numel()), int(keep.numel())
        return self.kept

    @torch.no_grad()
    def render(self, c2w, K, width, height, camera_model="pinhole"):
        v = self.view
        if v["means"].shape[0] == 0:
            return np.full((height, width, 3), 255, np.uint8)
        viewmat = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32))[None].to(self.device)
        Ks = torch.from_numpy(np.asarray(K, np.float32))[None].to(self.device)
        rgb, _, _ = rasterization(
            means=v["means"], quats=v["quats"], scales=v["scales"],
            opacities=v["opacities"], colors=v["colors"],
            viewmats=viewmat, Ks=Ks, width=width, height=height,
            sh_degree=self.gs["sh_degree"], camera_model=camera_model,
            backgrounds=self.background,
        )
        return (rgb[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    def render_isometric(self, width, height, azimuth, elevation, zoom=1.0):
        """One orthographic isometric frame, framed to the room box."""
        c2w = isometric_c2w(self.frame, azimuth, elevation)
        world_h = fit_ortho_height(self.frame, c2w, width / height) / max(zoom, 1e-3)
        K = ortho_K(width, height, world_h)
        return self.render(c2w, K, width, height, camera_model="ortho")


# --------------------------------------------------------------------------- #
# headless
# --------------------------------------------------------------------------- #

def _label(img, text):
    """Caption a tile, so a contact sheet is readable without its filenames."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(24, 24, 28))
    d.text((8, 7), text, fill=(255, 255, 255))
    return np.asarray(im)


def screenshot(house, out_prefix, width, height, elevation, zoom, azimuths):
    """Headless POC: the occlusion, the fix, and the fix from four corners."""
    from PIL import Image
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    az0 = azimuths[0]
    wall_cut = house.wall_cut

    def shot(az):
        house.azimuth = az
        house.rebuild()
        return house.render_isometric(width, height, az, elevation, zoom), house.kept

    # 1. the problem: an indoor scan viewed from above is a roof
    house.mode, house.wall_cut = "everything", False
    before, n_before = shot(az0)
    Image.fromarray(before).save(f"{out}_ceiling_on.png")

    # 2. the ceiling alone is not enough -- the near walls still block the room
    house.mode, house.wall_cut = "dollhouse", False
    cut_only, n_cut = shot(az0)

    # 3. the full dollhouse: near walls taken down too
    house.wall_cut = wall_cut
    walls, n_walls = shot(az0)
    Image.fromarray(walls).save(f"{out}_dollhouse.png")

    label3 = (f"+ near walls x{house.wall_scale:.2f}" if not house.wall_frac
              else f"+ near-wall slab {house.wall_frac:.2f}")
    compare = np.concatenate([
        _label(before, f"ceiling intact  ({n_before:,})"),
        _label(cut_only, f"ceiling cut at h={house.cut:+.3f}  ({n_cut:,})"),
        _label(walls, f"{label3}  ({n_walls:,})"),
    ], axis=1)
    Image.fromarray(compare).save(f"{out}_compare.png")
    print(f"[i] {out}_compare.png   before / ceiling cut / full dollhouse")

    # 4. the dollhouse from every corner
    tiles = []
    for az in azimuths:
        img, kept = shot(az)
        Image.fromarray(img).save(f"{out}_az{int(az):03d}.png")
        print(f"[i] {out}_az{int(az):03d}.png  ({kept:,} gaussians)")
        tiles.append(_label(img, f"azimuth {int(az)} deg"))

    cols = 2 if len(tiles) > 2 else len(tiles)
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    sheet = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
    Image.fromarray(sheet).save(f"{out}_sheet.png")
    print(f"[i] {out}_sheet.png  (contact sheet)")


# --------------------------------------------------------------------------- #
# interactive
# --------------------------------------------------------------------------- #

def serve(house, host, port, ortho_default=True):
    try:
        import viser
        from nerfview import Viewer
    except ImportError as e:
        # Deliberately not a hard dependency: --screenshot needs neither, and
        # gsplat (which is one) does the rasterizing either way.
        sys.exit(f"[x] the interactive viewer needs viser + nerfview ({e}).\n"
                 f"    uv sync --extra viewer     (or use --screenshot instead)")

    server = viser.ViserServer(host=host, port=port, verbose=False)
    server.scene.set_up_direction(tuple(float(x) for x in house.frame.up))

    state = {"ortho": ortho_default}
    frame, lv = house.frame, house.levels
    span = float(frame.hi[2] - frame.lo[2])

    with server.gui.add_folder("Dollhouse cut"):
        g_mode = server.gui.add_dropdown("Show", Dollhouse.MODES, initial_value="dollhouse")
        g_cut = server.gui.add_slider("Ceiling cut", float(frame.lo[2]), float(frame.hi[2]),
                                      span / 400.0, float(house.cut))
        g_wallon = server.gui.add_checkbox("Cut near walls", house.wall_cut)
        g_wall = server.gui.add_slider("Wall cut x", 0.0, 2.0, 0.05, house.wall_scale)
        g_opac = server.gui.add_slider("Min opacity", 0.0, 0.9, 0.01, 0.0)
        g_crop = server.gui.add_checkbox("Crop floaters to room box", True)
        g_box = server.gui.add_slider("Room box tightness (%)", 0.1, 12.0, 0.1,
                                      float(house.box_percentile))
        g_auto = server.gui.add_button("Reset cut to detected ceiling")
        g_info = server.gui.add_markdown("")

    with server.gui.add_folder("Isometric view"):
        g_ortho = server.gui.add_checkbox("Orthographic", ortho_default)
        g_az = server.gui.add_slider("Azimuth", 0.0, 360.0, 1.0, 45.0)
        g_el = server.gui.add_slider("Elevation", 5.0, 89.0, 0.5, ISO_ELEVATION_DEG)
        g_snap = server.gui.add_button("Snap camera to isometric")
        g_true = server.gui.add_button("True isometric (35.26 deg)")

    def info():
        pct = 100.0 * house.kept / max(house.total, 1)
        wall_txt = " · ".join(
            f"`{w.name}` {w.thickness:.3f}" if w.confident else f"`{w.name}` —"
            for w in house.walls
        )
        weak = [w.name for w in house.walls if not w.confident]
        g_info.content = (
            f"**{house.kept:,}** / {house.total:,} gaussians ({pct:.1f}%)\n\n"
            f"floor `{lv.floor:+.3f}` · ceiling `{lv.ceiling:+.3f}` · "
            f"cut `{house.cut:+.3f}`"
            + ("" if lv.confident else "\n\n⚠ ceiling detection was low-confidence")
            + (f"\n\nwall thickness — {wall_txt}" if house.walls else "")
            + (f"\n\n⚠ no wall found for {', '.join(weak)}; left uncut" if weak else "")
        )

    def refresh(_=None):
        house.mode = g_mode.value
        house.cut = g_cut.value
        house.wall_cut = g_wallon.value
        house.wall_scale = g_wall.value
        house.opacity_min = g_opac.value
        house.crop = g_crop.value
        house.box_percentile = g_box.value
        house.azimuth = g_az.value
        house.rebuild()
        info()
        viewer.rerender(None)

    for w in (g_mode, g_cut, g_wallon, g_wall, g_opac, g_crop, g_box):
        w.on_update(refresh)

    @g_auto.on_click
    def _(_):
        g_cut.value = float(lv.cut)     # fires refresh through on_update

    def snap(_=None):
        """Put the client camera on the isometric pose, framed to the room.

        Orbit distance is what sets the zoom in both projections -- under ortho
        because `perspective_to_ortho` reads the scale off it, under perspective
        because that is what distance does. So it is solved for rather than left
        at a fixed multiple of the scene size, which would leave the room
        occupying a quarter of the frame on first connect.
        """
        house.azimuth = g_az.value
        house.rebuild()
        for client in server.get_clients().values():
            fov = float(client.camera.fov)
            probe = isometric_c2w(frame, g_az.value, g_el.value)
            need = fit_ortho_height(frame, probe, float(client.camera.aspect))
            dist = need / (2.0 * math.tan(fov / 2.0))
            c2w = isometric_c2w(frame, g_az.value, g_el.value, distance=dist)
            client.camera.up_direction = tuple(float(x) for x in frame.up)
            client.camera.position = tuple(float(x) for x in c2w[:3, 3])
            client.camera.look_at = tuple(float(x) for x in frame.target)
        info()

    g_snap.on_click(snap)
    g_az.on_update(lambda _: (refresh(), snap()))
    g_el.on_update(snap)

    @g_true.on_click
    def _(_):
        g_el.value = ISO_ELEVATION_DEG

    @g_ortho.on_update
    def _(_):
        state["ortho"] = g_ortho.value
        viewer.rerender(None)

    pullback = 2.0 * frame.diagonal

    @torch.no_grad()
    def render_fn(camera_state, img_wh):
        width, height = img_wh
        c2w = camera_state.c2w.astype(np.float64)
        if state["ortho"]:
            c2w, world_h = perspective_to_ortho(c2w, camera_state.fov,
                                                frame.target, pullback)
            return house.render(c2w, ortho_K(width, height, world_h),
                                width, height, camera_model="ortho")
        return house.render(c2w, camera_state.get_K(img_wh), width, height)

    viewer = Viewer(server=server, render_fn=render_fn, mode="rendering")

    @server.on_client_connect
    def _(client):
        client.camera.up_direction = tuple(float(x) for x in frame.up)
        snap()

    info()
    print(f"\n  Dollhouse viewer  ->  http://127.0.0.1:{port}")
    print(f"  {house.kept:,}/{house.total:,} gaussians after the cut  "
          f"({'ortho' if ortho_default else 'perspective'})\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[i] stopped")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True,
                    help="3DGS PLY, or a model directory holding point_cloud/iteration_*")
    ap.add_argument("-c", "--cameras", default=None,
                    help="SfM cameras.json; used to estimate the world up axis")
    ap.add_argument("--up", default=None, help="override up: x|-x|y|-y|z|-z or 'a,b,c'")
    ap.add_argument("--cut", type=float, default=None,
                    help="ceiling cut height in room coords; default is detected")
    ap.add_argument("--sh_degree", type=int, default=None,
                    help="truncate spherical harmonics (0 = flat colour, often "
                         "steadier from an isometric camera)")
    ap.add_argument("--max_gaussians", type=int, default=None,
                    help="random subsample, for a tight GPU")
    ap.add_argument("--opacity_min", type=float, default=0.1,
                    help="opacity floor for the geometry *analysis* only")
    ap.add_argument("--percentile", type=float, default=2.0,
                    help="robust percentile for the room box; raise it to crop "
                         "the floaters indoor scans stream out through windows")
    ap.add_argument("--background", default="white", choices=["white", "black"])
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--screenshot", default=None,
                    help="headless: write <prefix>_az*.png and exit")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--elevation", type=float, default=ISO_ELEVATION_DEG)
    ap.add_argument("--azimuths", default="45,135,225,315")
    ap.add_argument("--zoom", type=float, default=1.0)

    ap.add_argument("--wall_scale", type=float, default=1.0,
                    help="multiplier on each wall's own detected thickness "
                         "(1.0 = exactly the detected shell, 0 = no wall cut)")
    ap.add_argument("--no_wall_cut", action="store_true",
                    help="keep the near walls; cut only the ceiling")
    ap.add_argument("--wall_frac", type=float, default=0.0,
                    help="legacy override: one absolute slab thickness, as a "
                         "fraction of room height, for all four walls")

    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--perspective", action="store_true",
                    help="start in perspective instead of orthographic")
    args = ap.parse_args()

    ply = latest_gs_ply(args.model)
    print(f"[i] loading {ply}")
    t0 = time.time()
    gs = load_gaussians(ply, args.device, args.sh_degree, args.max_gaussians)
    print(f"[i] {gs['means'].shape[0]:,} gaussians, SH degree {gs['sh_degree']}, "
          f"{time.time() - t0:.1f}s")

    sel = gs["opac_np"] > args.opacity_min
    up, src = estimate_up(args.cameras, means=gs["means_np"][sel], override=args.up)
    print(f"[i] up = {np.round(up, 4)}  ({src})")

    frame = build_room_frame(gs["means_np"][sel], gs["opac_np"][sel], up,
                             percentile=args.percentile)
    print(f"[i] room {np.round(frame.size, 3)} (u1, u2, h), diagonal {frame.diagonal:.3f}")

    cam_h = None
    if args.cameras and os.path.isfile(args.cameras):
        import json
        with open(args.cameras) as f:
            c2w = np.asarray(json.load(f)["cams2world"], dtype=np.float64)
        cam_h = (c2w[:, :3, 3] - frame.center) @ up

    frame.prepare_box_sampler(frame.to_frame(gs["means_np"]), gs["opac_np"])
    frame.set_box(args.percentile)
    coords_sel = frame.to_frame(gs["means_np"][sel])
    levels = detect_levels(coords_sel[:, 2], gs["opac_np"][sel], cam_heights=cam_h)
    if args.cut is not None:
        levels.cut = args.cut
        levels.note = f"cut overridden on the command line ({args.cut:+.4f})"
    print(f"[i] {levels.describe()}")
    if args.screenshot:
        print(histogram_ascii(levels))

    walls = detect_wall_planes(coords_sel, gs["opac_np"][sel], frame,
                               levels.cut, levels.floor,
                               room_confident=levels.confident)
    print("[i] walls:")
    for w in walls:
        print(f"      {w.describe()}")
    weak = [w.name for w in walls if not w.confident]
    if weak:
        print(f"    {len(weak)} of 4 not confident ({', '.join(weak)}); left uncut")

    bg = (1.0, 1.0, 1.0) if args.background == "white" else (0.0, 0.0, 0.0)
    house = Dollhouse(gs, frame, levels, walls=walls, background=bg, device=args.device)
    house.wall_cut = not args.no_wall_cut
    house.wall_scale = args.wall_scale
    house.wall_frac = args.wall_frac
    house.rebuild()
    print(f"[i] dollhouse keeps {house.kept:,} / {house.total:,} "
          f"({100.0 * house.kept / max(house.total, 1):.1f}%)")

    if args.screenshot:
        azimuths = [float(a) for a in args.azimuths.split(",")]
        screenshot(house, args.screenshot, args.width, args.height,
                   args.elevation, args.zoom, azimuths)
        return
    serve(house, args.host, args.port, ortho_default=not args.perspective)


if __name__ == "__main__":
    main()
