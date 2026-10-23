"""Reject changed artifacts and unselected runs before held-out evaluation."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validate_protocol import ProtocolValidationError, assert_locked, sha256


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "src" / "experiment.py"
        self.source.parent.mkdir()
        self.source.write_text("# locked training source\n")
        self.analysis = self.source.with_name("analyse_results.py")
        self.analysis.write_text("# locked analysis source\n")
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_text("image_id,split,label\na,test,0\n")
        self.checkpoint = self.root / "runs" / "frozen" / "seed_17" / "best.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"synthetic checkpoint bytes; never unpickled")
        self.config_path = self.checkpoint.with_name("config.json")
        self.config = {"arm": "frozen", "seed": 17, "model_id": "facebook/dinov2-small",
                       "requested_revision": "pinned-test-revision", "batch_size": 8,
                       "manifest_sha256": sha256(self.manifest),
                       "training_code_sha256": sha256(self.source)}
        self.config_path.write_text(json.dumps(self.config))
        self.summary = self.checkpoint.with_name("summary.json")
        self.summary.write_text(json.dumps({"best_epoch": 4}))
        self.lock_path = self.root / "protocol.lock.json"
        self.local_lock = {"training_code_sha256": sha256(self.source),
                           "manifest_sha256": sha256(self.manifest),
                           "model": "facebook/dinov2-small", "revision": "pinned-test-revision",
                           "seeds": [17, 42, 2026],
                           "runs": [{"arm": "frozen", "seed": 17,
                                     "checkpoint_sha256": sha256(self.checkpoint),
                                     "config_sha256": sha256(self.config_path)}]}
        self.lock_path.write_text(json.dumps(self.local_lock))

    def tearDown(self):
        self.directory.cleanup()

    def verify(self):
        return assert_locked(self.manifest, self.checkpoint, self.lock_path, self.source)

    def write_notebook_lock(self):
        lock = {"manifest_sha256": sha256(self.manifest),
                "settings": {"model_id": "facebook/dinov2-small",
                             "model_revision": "pinned-test-revision", "seeds": [17], "batch_size": 8},
                "source_sha256": {"experiment.py": sha256(self.source),
                                  "analyse_results.py": sha256(self.analysis)},
                "selected_checkpoints": [{"arm": "frozen", "seed": 17,
                    "path": "runs/frozen/seed_17/best.pt", "sha256": sha256(self.checkpoint),
                    "config_sha256": sha256(self.config_path), "summary_sha256": sha256(self.summary)}]}
        self.lock_path.write_text(json.dumps(lock))

    def test_local_schema_accepts_intact_artifacts(self):
        verified = self.verify()
        self.assertEqual(verified["status"], "provenance_verified")
        self.assertEqual(verified["schema"], "local_runs")
        self.assertEqual(verified["seed"], 17)

    def test_notebook_schema_checks_other_sources_and_summary(self):
        self.write_notebook_lock()
        verified = self.verify()
        self.assertEqual(verified["schema"], "notebook_checkpoints")
        self.assertEqual(verified["summary_sha256"], sha256(self.summary))
        self.assertIn("analyse_results.py", verified["source_hashes_verified"])

    def test_modified_checkpoint_is_rejected(self):
        self.checkpoint.write_bytes(self.checkpoint.read_bytes() + b"changed")
        with self.assertRaisesRegex(ProtocolValidationError, "selected checkpoint SHA256"):
            self.verify()

    def test_modified_manifest_is_rejected(self):
        self.manifest.write_text("image_id,split,label\na,test,2\n")
        with self.assertRaisesRegex(ProtocolValidationError, "manifest SHA256"):
            self.verify()

    def test_modified_training_source_is_rejected(self):
        self.source.write_text("# different experiment\n")
        with self.assertRaisesRegex(ProtocolValidationError, "training code SHA256"):
            self.verify()

    def test_modified_run_config_is_rejected(self):
        self.config["batch_size"] = 16
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ProtocolValidationError, "run config SHA256"):
            self.verify()

    def test_unselected_run_cannot_reuse_a_selected_checkpoint(self):
        self.config["seed"] = 42
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ProtocolValidationError, "must occur exactly once"):
            self.verify()

    def test_empty_checkpoint_selection_is_rejected(self):
        self.local_lock["runs"] = []
        self.lock_path.write_text(json.dumps(self.local_lock))
        with self.assertRaisesRegex(ProtocolValidationError, "nonempty"):
            self.verify()

    def test_ambiguous_duplicate_run_selection_is_rejected(self):
        self.local_lock["runs"] *= 2
        self.lock_path.write_text(json.dumps(self.local_lock))
        with self.assertRaisesRegex(ProtocolValidationError, "exactly once"):
            self.verify()

    def test_modified_recorded_analysis_source_is_rejected(self):
        self.write_notebook_lock()
        self.analysis.write_text("# changed analysis\n")
        with self.assertRaisesRegex(ProtocolValidationError, "source analyse_results.py SHA256"):
            self.verify()

    def test_modified_recorded_summary_is_rejected(self):
        self.write_notebook_lock()
        self.summary.write_text(json.dumps({"best_epoch": 8}))
        with self.assertRaisesRegex(ProtocolValidationError, "run summary SHA256"):
            self.verify()

    def test_intact_relocated_archive_is_accepted(self):
        self.write_notebook_lock()
        relocated = self.root / "relocated" / "best.pt"
        relocated.parent.mkdir()
        for original in (self.checkpoint, self.config_path, self.summary):
            shutil.copy2(original, relocated.parent / original.name)
        verified = assert_locked(self.manifest, relocated, self.lock_path, self.source)
        self.assertEqual(verified["checkpoint_sha256"], sha256(self.checkpoint))


if __name__ == "__main__":
    unittest.main()
