import megatron.training.arguments  # noqa: F401
from megatron.core.models.gpt import gpt_layer_specs


def test_local_gpt_layer_spec_supports_layernorm():
    spec = gpt_layer_specs.get_gpt_layer_local_spec(normalization="LayerNorm")

    assert (
        spec.submodules.input_layernorm
        is gpt_layer_specs.LNImpl
    )
