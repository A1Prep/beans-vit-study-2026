"""CPU-only checks for SigLIP pooled features, frozen pooling, and provenance.

Tiny randomly initialized fixtures verify code behavior, not model accuracy.
No pretrained model download, dataset inference, or GPU training is performed.
"""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import followup_experiment as exp

torch.set_num_threads(2)


class ToySiglipVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, model_type="siglip_vision_model")
        self.vision_model = nn.Module()
        self.vision_model.embedding = nn.Linear(3, 4)
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.vision_model.post_layernorm = nn.LayerNorm(4)
        self.vision_model.head = nn.Linear(4, 4)
        self.vision_model.use_head = True

    def forward(self, pixel_values, return_dict=True):
        hidden = self.vision_model.embedding(pixel_values.mean((2, 3)))
        for layer in self.vision_model.encoder.layers:
            hidden = hidden + torch.tanh(layer(hidden))
        hidden = self.vision_model.post_layernorm(hidden)
        return SimpleNamespace(last_hidden_state=torch.stack([hidden + 50, hidden - 10], dim=1),
                               pooler_output=self.vision_model.head(hidden))


def tiny_siglip_source(folder: Path, missing_key=None):
    from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipTextConfig, SiglipVisionConfig
    vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                               num_attention_heads=2, image_size=224, patch_size=16)
    text = SiglipTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8)
    full = SiglipModel(SiglipConfig(vision_config=vision.to_dict(), text_config=text.to_dict()))
    state = dict(full.state_dict())
    if missing_key:
        del state[missing_key]
    full.save_pretrained(folder, state_dict=state, safe_serialization=True)
    processor = SiglipImageProcessor(size={"height": 224, "width": 224},
                                    resample=Image.Resampling.BILINEAR,
                                    image_mean=[0.5] * 3, image_std=[0.5] * 3)
    processor.save_pretrained(folder)
    return full, processor


class FollowupTests(unittest.TestCase):
    def test_siglip_uses_pooler_output_not_first_patch(self):
        torch.manual_seed(7)
        model = exp.BeansClassifier(ToySiglipVision(), "frozen", 1_000_017).eval()
        with torch.no_grad():
            model.classifier.weight.copy_(torch.eye(3, 4))
            model.classifier.bias.zero_()
            pixels = torch.randn(2, 3, 8, 8)
            output = model.backbone(pixels)
            actual = model(pixels)
            pooled = model.classifier(output.pooler_output)
            wrong_first_patch = model.classifier(output.last_hidden_state[:, 0, :])
        torch.testing.assert_close(actual, pooled)
        self.assertFalse(torch.allclose(actual, wrong_first_patch))

    def test_frozen_pooler_transmits_partial_gradients_without_weight_updates(self):
        torch.manual_seed(11)
        backbone = ToySiglipVision()
        models = [exp.BeansClassifier(copy.deepcopy(backbone), arm, 1_000_017)
                  for arm in ["frozen", "partial"]]
        self.assertTrue(torch.equal(models[0].classifier.weight, models[1].classifier.weight))
        self.assertTrue(torch.equal(models[0].classifier.bias, models[1].classifier.bias))
        pixels, labels = torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 2])
        allowed = ("backbone.vision_model.encoder.layers.2.", "backbone.vision_model.encoder.layers.3.",
                   "backbone.vision_model.post_layernorm.")
        for model in models:
            with self.subTest(arm=model.arm):
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                model.train()
                self.assertFalse(model.backbone.training)
                optimizer = exp.make_optimizer(model, 1e-2, 1e-3, .01)
                nn.functional.cross_entropy(model(pixels), labels).backward()
                for name, parameter in model.named_parameters():
                    expected = name.startswith("classifier.") or (
                        model.arm == "partial" and name.startswith(allowed))
                    self.assertEqual(parameter.requires_grad, expected, name)
                    self.assertEqual(parameter.grad is not None, expected, name)
                optimizer.step()
                changed = {name for name, parameter in model.named_parameters()
                           if not torch.equal(parameter, before[name])}
                self.assertIn("classifier.weight", changed)
                self.assertFalse(any(name.startswith("backbone.vision_model.head.") for name in changed))
                self.assertFalse(any(name.startswith("backbone.vision_model.embedding.") for name in changed))
                if model.arm == "partial":
                    self.assertTrue(any(name.startswith(allowed) for name in changed))
                    self.assertEqual(model.unfrozen_modules, ["vision_model.encoder.layers.2",
                                     "vision_model.encoder.layers.3", "vision_model.post_layernorm"])
                else:
                    self.assertFalse(any(name.startswith("backbone.") for name in changed))

    def test_native_siglip_pixels_match_processor_and_dino_stays_bicubic(self):
        from transformers import SiglipImageProcessor
        processor = SiglipImageProcessor(size={"height": 224, "width": 224},
                                        resample=Image.Resampling.BILINEAR,
                                        image_mean=[.5] * 3, image_std=[.5] * 3)
        specification = exp.native_preprocessing(processor, "siglip_vision_model")
        self.assertEqual(specification["resample"], 2)
        self.assertEqual(specification["size"], 224)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "non_square.png"
            pixels = np.random.default_rng(17).integers(0, 256, size=(31, 43, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            image.save(path)
            rows = [{"image_id": "test/image.png", "resolved_path": str(path),
                     "label": 0, "label_name": "angular_leaf_spot"}]
            actual = exp.BeansImages(rows, specification)[0]["pixel_values"]
            native = processor(images=image, return_tensors="pt").pixel_values[0]
            torch.testing.assert_close(actual, native, rtol=0, atol=1e-6)
            dino = exp.native_preprocessing(SimpleNamespace(image_mean=[.485, .456, .406],
                                                           image_std=[.229, .224, .225]), "dinov2")
            self.assertEqual(dino["resample"], 3)
            original_pixels = np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255
            expected = torch.from_numpy(((original_pixels - np.array(dino["mean"], dtype=np.float32))
                                        / np.array(dino["std"], dtype=np.float32)).transpose(2, 0, 1).copy())
            torch.testing.assert_close(exp.BeansImages(rows, dino)[0]["pixel_values"], expected, rtol=0, atol=0)
        processor.resample = Image.Resampling.BICUBIC
        with self.assertRaisesRegex(ValueError, "native SigLIP2"):
            exp.native_preprocessing(processor, "siglip_vision_model")

    def test_loading_validation_rejects_missing_mismatched_or_unexpected_vision(self):
        clean = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        permitted = {**clean, "unexpected_keys": ["text_model.embeddings.token_embedding.weight", "logit_scale", "logit_bias"]}
        self.assertTrue(exp.validate_pretrained_loading_info(permitted, "siglip_vision_model")["vision_weights_fully_loaded"])
        for key, value in [("missing_keys", ["vision_model.head.probe"]),
                           ("unexpected_keys", ["vision_model.wrong.weight"]),
                           ("mismatched_keys", [("vision_model.head.probe", [1, 1, 4], [1, 1, 8])]),
                           ("error_msgs", ["loader failed"])]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Pretrained vision loading"):
                exp.validate_pretrained_loading_info({**clean, key: value}, "siglip_vision_model")
        with self.assertRaises(ValueError):
            exp.validate_pretrained_loading_info(permitted, "dinov2")

    def test_local_full_siglip_checkpoint_loads_all_vision_weights_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            original, _ = tiny_siglip_source(folder)
            manifest = folder / "manifest.csv"
            manifest.write_text("test fixture\n")
            args = exp.parser().parse_args(["pilot", "--manifest", str(manifest), "--output-dir", str(folder / "out"),
                                           "--local-model-path", str(folder), "--model-id", "google/siglip2-base-patch16-224",
                                           "--revision", "test-fixture"])
            model, preprocessing = exp.load_pretrained(args, "partial", 17)
            self.assertEqual(model.backbone.config.model_type, "siglip_vision_model")
            for name, tensor in model.backbone.state_dict().items():
                self.assertTrue(torch.equal(tensor, original.state_dict()[name]), name)
            info = model.pretrained_loading_info
            self.assertEqual(info["missing_keys"], [])
            self.assertEqual(info["mismatched_keys"], [])
            self.assertIn("logit_scale", info["allowed_unused_text_or_logit_keys"])
            metadata = exp.run_metadata(args, model, preprocessing, "partial", 17, torch.device("cpu"))
            self.assertTrue(metadata["pretrained_loading_info"]["vision_weights_fully_loaded"])
            self.assertEqual(metadata["training_code_sha256"], exp.file_sha256(Path(exp.__file__)))
            self.assertIn("pooler_output", metadata["pooling"])

    def test_missing_real_pretrained_pooler_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tiny_siglip_source(folder, missing_key="vision_model.head.probe")
            args = SimpleNamespace(local_model_path=str(folder), model_id="google/siglip2-base-patch16-224", revision=None)
            with self.assertRaisesRegex(ValueError, "Pretrained vision loading"):
                exp.load_pretrained(args, "frozen", 17)

    def test_tiny_vision_only_siglip_and_dino_checkpoints_roundtrip(self):
        from transformers import Dinov2Config, Dinov2Model, SiglipVisionConfig, SiglipVisionModel
        backbones = [SiglipVisionModel(SiglipVisionConfig(hidden_size=24, intermediate_size=48,
                     num_hidden_layers=4, num_attention_heads=3, image_size=16, patch_size=8)),
                     Dinov2Model(Dinov2Config(hidden_size=24, num_hidden_layers=4, num_attention_heads=3,
                     intermediate_size=48, image_size=16, patch_size=8))]
        for backbone in backbones:
            with self.subTest(model_type=backbone.config.model_type), tempfile.TemporaryDirectory() as temporary:
                model = exp.BeansClassifier(backbone, "partial", 1_000_017).eval()
                pixels = torch.randn(2, 3, 16, 16)
                optimizer = exp.make_optimizer(model, 1e-3, 1e-5, .01)
                nn.functional.cross_entropy(model(pixels), torch.tensor([0, 1])).backward()
                optimizer.step()
                with torch.inference_mode():
                    expected = model(pixels)
                    if backbone.config.model_type == "dinov2":
                        torch.testing.assert_close(expected, model.classifier(backbone(pixels).last_hidden_state[:, 0, :]))
                path = Path(temporary) / "tiny.pt"
                exp.save_checkpoint(path, model, {"arm": "partial", "seed": 17})
                restored, _ = exp.load_checkpoint(path, torch.device("cpu"))
                with torch.inference_mode():
                    observed = restored(pixels)
                torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(restored.backbone.config.model_type, backbone.config.model_type)
                self.assertEqual(restored.unfrozen_modules, model.unfrozen_modules)

    def test_test_protocol_is_checked_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            lock = folder / "lock.json"
            lock.write_text("{}")
            args = SimpleNamespace(split="test", confirm_test_locked=True, protocol_lock=str(lock),
                                   manifest=str(folder / "manifest.csv"), checkpoint=str(folder / "best.pt"))
            with patch("validate_protocol.assert_locked", side_effect=ValueError("tampered provenance")) as validator, \
                    patch.object(exp, "load_checkpoint") as loader:
                with self.assertRaisesRegex(ValueError, "tampered provenance"):
                    exp.evaluate(args, [], torch.device("cpu"))
                loader.assert_not_called()
                self.assertEqual(validator.call_args.kwargs["training_code"], Path(exp.__file__))


if __name__ == "__main__":
    unittest.main()
