"""Neutralise GSFix3D's unconditional xformers call. Injected via PYTHONPATH.

Python imports `sitecustomize` automatically at interpreter startup if it is
importable, which lets us patch a dependency of a third-party script without
editing that script -- GSFix3D stays a clean upstream checkout.

The problem: xformers >= 0.0.35 removed `xformers.ops.memory_efficient_attention`
entirely (upstream deprecated it in favour of PyTorch's SDPA). diffusers still
takes the xformers path because `is_xformers_available()` only checks that the
package imports, then dies in its smoke test:

    AttributeError: module 'xformers.ops' has no attribute
                    'memory_efficient_attention'

Two GSFix3D entry points reach it, and they are *different methods on different
classes* -- patching only the first leaves inference broken:

  * `src/trainer/base_trainer.py:105` calls `self.model.unet.enable_...()`,
    i.e. `ModelMixin.enable_xformers_memory_efficient_attention`.
  * `scripts/gsfixer/inference.py:209` calls `pipe.enable_...()`, i.e.
    `DiffusionPipeline.enable_xformers_memory_efficient_attention`, which walks
    every component and calls `set_use_memory_efficient_attention_xformers` on
    each. That call site does guard itself, but only catches ImportError, and
    this is an AttributeError.

Both funnel into `ModelMixin.set_use_memory_efficient_attention_xformers`, so
that is the one patch that has to land; the two `enable_*` no-ops just stop the
walk before it starts.

Making it a no-op costs nothing: diffusers then keeps `AttnProcessor2_0`, which
dispatches to `torch.nn.functional.scaled_dot_product_attention` and is already
memory-efficient. On this machine's RTX 5090 that is FlashAttention either way.
"""

try:
    from diffusers.models.modeling_utils import ModelMixin

    def _xformers_noop(self, *args, **kwargs):
        return None

    # The chokepoint: reached both directly and via the pipeline's recursion.
    ModelMixin.set_use_memory_efficient_attention_xformers = _xformers_noop
    ModelMixin.enable_xformers_memory_efficient_attention = _xformers_noop

    # Imported second so a restructure here cannot cost us the patches above.
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    DiffusionPipeline.enable_xformers_memory_efficient_attention = _xformers_noop
    DiffusionPipeline.set_use_memory_efficient_attention_xformers = _xformers_noop
except Exception:
    # diffusers absent or restructured -- nothing to patch, and this module must
    # never be able to break an unrelated interpreter start.
    pass
