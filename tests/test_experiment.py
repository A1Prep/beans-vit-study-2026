"""Scientific invariants: matched inputs, controlled updates, and selection rules."""
import copy
import csv
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import experiment as exp

torch.set_num_threads(2)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Linear(3, 4)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.layernorm = nn.LayerNorm(4)

    def forward(self, pixel_values, return_dict=True):
        features = self.embedding(pixel_values.mean((2, 3)))
        for block in self.encoder.layer:
            features = features + torch.tanh(block(features))
        return SimpleNamespace(last_hidden_state=self.layernorm(features).unsqueeze(1))


class ControlledExperimentTests(unittest.TestCase):
    def test_partial_update_changes_only_final_blocks_norm_and_head(self):
        torch.manual_seed(4)
        backbone = ToyBackbone()
        frozen = exp.BeansClassifier(copy.deepcopy(backbone), "frozen", 999)
        partial = exp.BeansClassifier(copy.deepcopy(backbone), "partial", 999)
        self.assertTrue(torch.equal(frozen.classifier.weight, partial.classifier.weight))
        self.assertTrue(torch.equal(frozen.classifier.bias, partial.classifier.bias))
        inputs = torch.randn(3, 3, 8, 8)
        labels = torch.tensor([0, 1, 2])
        for arm, model in [("frozen", frozen), ("partial", partial)]:
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            model.train()
            self.assertFalse(model.backbone.training)
            self.assertTrue(model.classifier.training)
            optimizer = exp.make_optimizer(model, 1e-2, 1e-3, 0.01)
            nn.functional.cross_entropy(model(inputs), labels).backward()
            for name, parameter in model.named_parameters():
                expected = name.startswith("classifier.") or (
                    arm == "partial" and any(name.startswith(prefix) for prefix in
                    ("backbone.encoder.layer.2.", "backbone.encoder.layer.3.", "backbone.layernorm.")))
                self.assertEqual(parameter.requires_grad, expected, name)
                self.assertEqual(parameter.grad is not None, expected, name)
            optimizer.step()
            changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
            self.assertIn("classifier.weight", changed)
            self.assertFalse(any(name.startswith("backbone.embedding.") for name in changed))
            self.assertFalse(any(name.startswith("backbone.encoder.layer.0.") for name in changed))
            if arm == "frozen":
                self.assertFalse(any(name.startswith("backbone.") for name in changed))
            else:
                self.assertTrue(any(name.startswith("backbone.encoder.layer.3.") for name in changed))

    def test_augmented_pixels_and_shuffle_ignore_unrelated_rng_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            array = np.arange(31 * 31 * 3, dtype=np.uint32).reshape(31, 31, 3) % 256
            Image.fromarray(array.astype(np.uint8)).save(path)
            rows = [{"image_id": str(i), "label": i % 3, "label_name": exp.CLASS_NAMES[i % 3],
                     "resolved_path": str(path)} for i in range(11)]
            preprocessing = {"size": 28, "mean": [.5] * 3, "std": [.5] * 3}
            first = exp.BeansImages(rows, preprocessing, True, 17, 2)
            pixels = first[0]["pixel_values"].clone()
            order = [item for batch in exp.make_loader(first, 3, 0, torch.device("cpu"))
                     for item in batch["image_id"]]
            torch.rand(1000)
            np.random.rand(1000)
            [random.random() for _ in range(1000)]
            second = exp.BeansImages(rows, preprocessing, True, 17, 2)
            self.assertTrue(torch.equal(pixels, second[0]["pixel_values"]))
            next_order = [item for batch in exp.make_loader(second, 3, 0, torch.device("cpu"))
                          for item in batch["image_id"]]
            self.assertEqual(order, next_order)
            third = exp.BeansImages(rows, preprocessing, True, 17, 3)
            self.assertFalse(torch.equal(pixels, third[0]["pixel_values"]))

    def test_loss_ties_can_save_without_extending_patience(self):
        selection = exp.CheckpointSelection(patience=3)
        self.assertEqual(selection.update(1, .8, .6), (True, False))
        self.assertEqual(selection.update(2, .8, .5), (True, False))
        self.assertEqual(selection.update(3, .8, .5), (False, False))
        self.assertEqual(selection.update(4, .79, .2), (False, True))
        self.assertEqual(selection.best_epoch, 2)
        self.assertEqual(selection.stale, 3)
        self.assertEqual(selection.update(5, .81, .6), (True, False))
        self.assertEqual(selection.stale, 0)

    def test_metrics_include_missing_predictions_without_nan(self):
        result = exp.metrics_from_predictions([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["macro_f1"], (.5 + .8 + 2 / 3) / 3)
        missing = exp.metrics_from_predictions([0, 1, 2], [0, 0, 0])
        self.assertTrue(np.isfinite(missing["macro_f1"]))
        self.assertEqual(missing["per_class"][2]["f1"], 0)

    def test_manifest_exclusions_and_label_mapping_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            fields = ["image_id", "split", "label", "label_name", "path", "include"]
            rows = [["a", "train", 0, "angular_leaf_spot", "a.png", True],
                    ["b", "train", 1, "bean_rust", "b.png", False]]
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            included = exp.read_manifest(path, Path(temporary))
            self.assertEqual([row["image_id"] for row in included], ["a"])
            rows[0][3] = "healthy"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                exp.read_manifest(path, Path(temporary))

    def test_test_evaluation_requires_explicit_protocol_gate(self):
        args = SimpleNamespace(split="test", confirm_test_locked=False)
        with self.assertRaisesRegex(ValueError, "confirm-test-locked"):
            exp.evaluate(args, [], torch.device("cpu"))
        args.confirm_test_locked = True
        args.protocol_lock = None
        with self.assertRaisesRegex(ValueError, "protocol-lock"):
            exp.evaluate(args, [], torch.device("cpu"))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
    def test_real_tiny_dinov2_checkpoint_reloads_without_model_download(self):
        from transformers import Dinov2Config, Dinov2Model
        config = Dinov2Config(hidden_size=24, num_hidden_layers=4, num_attention_heads=3,
                             intermediate_size=48, image_size=28, patch_size=14)
        backbone = Dinov2Model(config)
        model = exp.BeansClassifier(backbone, "partial", 1_000_017).eval()
        pixels = torch.randn(2, 3, 28, 28)
        optimizer = exp.make_optimizer(model, 1e-3, 1e-5, .01)
        nn.functional.cross_entropy(model(pixels), torch.tensor([0, 1])).backward()
        optimizer.step()
        with torch.inference_mode():
            expected = model(pixels)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.pt"
            exp.save_checkpoint(path, model, {"arm": "partial", "seed": 17})
            restored, metadata = exp.load_checkpoint(path, torch.device("cpu"))
            with torch.inference_mode():
                observed = restored(pixels)
            torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
            self.assertEqual(metadata["arm"], "partial")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
