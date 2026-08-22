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

GSFix3D's `src/trainer/base_trainer.py:105` calls
`enable_xformers_memory_efficient_attention()` unguarded, so the fine-tune cannot
start. (Its `scripts/gsfixer/inference.py` does guard the same call, but only
catches ImportError, which this is not -- so inference needs the patch too.)

Making it a no-op costs nothing: diffusers then keeps `AttnProcessor2_0`, which
dispatches to `torch.nn.functional.scaled_dot_product_attention` and is already
memory-efficient. On this machine's RTX 5090 that is FlashAttention either way.
"""

try:
    from diffusers.models.modeling_utils import ModelMixin

    def _enable_xformers_noop(self, attention_op=None):
        return None

    ModelMixin.enable_xformers_memory_efficient_attention = _enable_xformers_noop
except Exception:
    # diffusers absent or restructured -- nothing to patch, and this module must
    # never be able to break an unrelated interpreter start.
    pass
