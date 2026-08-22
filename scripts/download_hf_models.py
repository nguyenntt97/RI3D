from huggingface_hub import hf_hub_download, snapshot_download

# Stable Diffusion v1.5 checkpoint (for ControlNet backbone)
hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned.ckpt",
    local_dir="models",
)

# ControlNet Tile v1.1 weights
hf_hub_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    filename="control_v11f1e_sd15_tile.pth",
    local_dir="models",
)

# SD2 Inpainting model (used as base for RealFill fine-tuning in Stage 4)
snapshot_download(
    repo_id="sd2-community/stable-diffusion-2-inpainting",
    local_dir="models/stable-diffusion-2-inpainting",
)

# GSFixer base checkpoint (used as base for the Stage 1c repair prior).
# `gsfixer-base` is the single-condition variant: its UNet takes 8 latent
# channels (4 noise + 4 render). The `gsfixer-full` sibling takes 12, the extra
# 4 being a mesh render -- RI3D has no mesh, so the base model is the right one.
# GSFix3D's trainer resolves the checkpoint as <base_ckpt_dir>/<pretrained_path>,
# so this must be a real local directory and not just a hub cache entry.
snapshot_download(
    repo_id="goldoak1421/gsfixer-base",
    local_dir="models/gsfixer-base",
)
