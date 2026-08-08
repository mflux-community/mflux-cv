import glob
import os

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common_models.gpt_oss.gpt_oss import (
    Model as GptOssModel,
    ModelArgs as GptOssModelArgs,
)
from mflux.models.lens.model.text_encoder.lens_gpt_oss_encoder import (
    LensGptOssEncoder,
    capture_hidden_states,
)
from mflux.models.lens.model.text_encoder.lens_prompt_template import (
    LENS_SELECTED_LAYERS,
    LENS_TXT_OFFSET,
    render_lens_chat,
)

_CACHED_SNAPSHOTS = glob.glob(
    os.path.expanduser(
        "/Volumes/1TB-WD750-1/ai-models-overflow/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/*"
    )
)


@pytest.mark.fast
class TestLensTemplate:
    def test_frozen_reference_strings(self):
        rendered = render_lens_chat("a cat")
        # The reference pipeline freezes these; touching them silently changes
        # every embedding, so pin them.
        assert "Current date: 2026-05-23" in rendered
        assert "Need to generate one image according to the description." in rendered
        assert rendered.endswith("<|start|>assistant<|channel|>final<|message|>")
        assert "<|start|>user<|message|>a cat<|end|>" in rendered

    def test_offset_is_larger_than_any_reasonable_prefix_change(self):
        assert LENS_TXT_OFFSET == 97
        assert LENS_SELECTED_LAYERS == (5, 11, 17, 23)


@pytest.mark.fast
class TestCaptureHiddenStates:
    def _tiny_model(self, layers=6):
        args = GptOssModelArgs(
            num_hidden_layers=layers,
            hidden_size=64,
            head_dim=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=64,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=128,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"] * (layers // 2),
        )
        return GptOssModel(args)

    def test_captures_after_selected_blocks(self):
        model = self._tiny_model(layers=6)
        ids = mx.array([[1, 2, 3, 4, 5]])
        captured = capture_hidden_states(model, ids, selected=(1, 3, 5))
        assert len(captured) == 3
        for h in captured:
            assert h.shape == (1, 5, 64)
        # Different depths must produce different states
        assert not mx.array_equal(captured[0], captured[1])
        assert not mx.array_equal(captured[1], captured[2])

    def test_capture_matches_manual_forward(self):
        model = self._tiny_model(layers=4)
        ids = mx.array([[7, 8, 9]])
        captured = capture_hidden_states(model, ids, selected=(3,))
        # Manually run all 4 layers and compare the final residual stream
        from mflux.models.common_models.gpt_oss.base_utils import create_attention_mask

        inner = model.model
        x = inner.embed_tokens(ids)
        full_mask = create_attention_mask(x, None)
        swa_mask = create_attention_mask(x, None, window_size=inner.window_size)
        for layer, layer_type in zip(inner.layers, inner.layer_types):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            x = layer(x, mask, None)
        assert mx.array_equal(captured[0], x)


@pytest.mark.fast
@pytest.mark.skipif(not _CACHED_SNAPSHOTS, reason="gpt-oss checkpoint not cached locally")
class TestTemplateTokenization:
    def test_prefix_tokenizes_to_exactly_txt_offset(self):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(_CACHED_SNAPSHOTS[0], "tokenizer.json"))
        rendered = render_lens_chat("PROBE")
        marker = "<|start|>user<|message|>"
        prefix = rendered[: rendered.index(marker) + len(marker)]
        n = len(tok.encode(prefix, add_special_tokens=False).ids)
        assert n == LENS_TXT_OFFSET


@pytest.mark.slow
@pytest.mark.skipif(not _CACHED_SNAPSHOTS, reason="gpt-oss checkpoint not cached locally")
class TestLensEncoderParity:
    def test_features_match_battery_022_reference(self):
        ref_path = os.path.join(os.path.dirname(__file__), "resources", "lens", "lens_exp1_features.npz")
        if not os.path.exists(ref_path):
            pytest.skip("reference features not present")
        ref = np.load(ref_path)
        encoder = LensGptOssEncoder(_CACHED_SNAPSHOTS[0])
        features = encoder.encode(str(ref["prompt"]))
        ours = np.array(features.astype(mx.float32))
        theirs = ref["features"]
        assert ours.shape == theirs.shape
        assert np.abs(ours - theirs).max() < 1e-2
