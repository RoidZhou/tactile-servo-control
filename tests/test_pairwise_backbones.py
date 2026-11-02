"""Exercise paired encoders and checkpoint compatibility without robot I/O."""

import gc
import io
import unittest

import torch
from torch import nn

from tactile_policy.model import (
    MODEL_TYPES,
    PAIRWISE_BACKBONES,
    PairwiseTactileSE3Regressor,
    create_pairwise_model,
    load_pairwise_model,
)


class ChannelMeanEncoder(nn.Module):
    """Expose exactly which samples/channels reach the shared encoder."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(6))
        self.calls = []

    def forward(self, inputs):
        self.calls.append(inputs.detach().clone())
        return inputs.mean(dim=(2, 3)) * self.scale


class PairwiseBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    @staticmethod
    def make_model(backbone):
        return create_pairwise_model(
            backbone=backbone, feature_dim=8, grid_size=8, image_size=64,
        )

    def test_factory_keeps_original_model_as_default(self):
        model = create_pairwise_model()
        self.assertIsInstance(model, PairwiseTactileSE3Regressor)
        self.assertEqual(PAIRWISE_BACKBONES, ("spatial",) + MODEL_TYPES)
        self.assertEqual(model.model_type, "pairwise_tactile_se3")
        self.assertEqual(model.model_config, {})
        self.assertEqual(model.tactile_input_kind, "paired_force")

    def test_all_backbones_train_with_one_native_tactile_pair(self):
        # B=1 is deliberate: FCN BatchNorm must see both times in one batch.
        for backbone in MODEL_TYPES:
            with self.subTest(backbone=backbone):
                model = self.make_model(backbone).train()
                inputs = torch.randn(1, 2, 6, 35, 20, requires_grad=True)
                output = model(inputs)
                self.assertEqual(tuple(output.shape), (1, 6))
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertTrue((output.abs() <= 1).all().item())
                (output - 0.25).square().mean().backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all().item())
                for module in (model.encoder, model.head):
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
                    self.assertTrue(any((g != 0).any().item() for g in gradients))
                del model, inputs, output, module, gradients
                gc.collect()

    def test_all_backbone_checkpoints_rebuild_and_preserve_predictions(self):
        inputs = torch.randn(1, 2, 6, 35, 20)
        for backbone in MODEL_TYPES:
            with self.subTest(backbone=backbone):
                model = self.make_model(backbone).eval()
                with torch.no_grad():
                    expected = model(inputs)
                checkpoint = {
                    "model_type": model.model_type,
                    "model_config": model.model_config,
                    "model_state_dict": model.state_dict(),
                }
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, map_location="cpu")
                restored = load_pairwise_model(saved, device="cpu")
                self.assertFalse(restored.training)
                self.assertEqual(restored.model_config, model.model_config)
                self.assertEqual(restored.tactile_input_kind, "paired_force")
                with torch.no_grad():
                    actual = restored(inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                del model, checkpoint, buffer, saved, restored, expected, actual
                gc.collect()

    def test_legacy_checkpoint_without_model_config_still_loads_strictly(self):
        original = PairwiseTactileSE3Regressor().eval()
        checkpoint = {
            "model_type": "pairwise_tactile_se3",
            "model_state_dict": original.state_dict(),
        }
        restored = load_pairwise_model(checkpoint)
        self.assertIsInstance(restored, PairwiseTactileSE3Regressor)
        self.assertEqual(set(restored.state_dict()), set(original.state_dict()))
        inputs = torch.randn(1, 2, 6, 35, 20)
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)

    def test_reference_current_and_channel_order_use_one_shared_encoder(self):
        model = create_pairwise_model("fcn", feature_dim=6, grid_size=8, image_size=64)
        encoder = ChannelMeanEncoder()
        model.encoder = encoder
        model.head = nn.Identity()
        values = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
        inputs = values[..., None, None].expand(2, 2, 6, 35, 20)
        result = model(inputs)
        self.assertEqual(len(encoder.calls), 1)
        torch.testing.assert_close(encoder.calls[0], inputs.reshape(4, 6, 35, 20))
        torch.testing.assert_close(result, values.reshape(2, 12))
        result.sum().backward()
        torch.testing.assert_close(encoder.scale.grad, values.sum(dim=(0, 1)))

    def test_backbones_reject_invalid_input_layouts(self):
        for backbone in ("spatial", "fcn"):
            model = self.make_model(backbone).eval()
            for shape in ((2, 6, 35, 20), (1, 1, 6, 35, 20), (1, 2, 3, 35, 20)):
                with self.subTest(backbone=backbone, shape=shape):
                    with self.assertRaisesRegex(ValueError, "Expected"):
                        model(torch.zeros(shape))

    def test_invalid_backbone_configuration_fails_before_training(self):
        invalid = (
            {"backbone": "unknown"},
            {"backbone": "fcn", "feature_dim": 0},
            {"backbone": "fcn", "feature_dim": True},
            {"backbone": "fcn", "grid_size": -1},
            {"backbone": "fcn", "image_size": 64.0},
            {"backbone": "vit", "image_size": 65},
            {"backbone": "simple_cnn", "image_size": 37},
            {"backbone": "nature_cnn", "image_size": 35},
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    create_pairwise_model(**config)

    def test_checkpoint_requires_complete_config_and_matching_weights(self):
        model = self.make_model("fcn")
        config = dict(model.model_config)
        invalid_configs = (None, {}, {"backbone": "fcn"},
                           dict(config, backbone="spatial"), dict(config, unused=1))
        for invalid in invalid_configs:
            with self.subTest(config=invalid):
                with self.assertRaisesRegex(ValueError, "model_config"):
                    load_pairwise_model({
                        "model_type": "pairwise_backbone_se3", "model_config": invalid,
                        "model_state_dict": model.state_dict(),
                    })
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_pairwise_model({"model_type": "unknown"})
        weights = dict(model.state_dict())
        weights.pop(next(iter(weights)))
        with self.assertRaises(RuntimeError):
            load_pairwise_model({
                "model_type": model.model_type, "model_config": config,
                "model_state_dict": weights,
            })


if __name__ == "__main__":
    unittest.main()
