"""Verify evaluation inputs against a saved, content-addressed protocol lock.

Run immediately before test evaluation. This checks consistency with the supplied
lock; it cannot authenticate its creation time or prove absence of earlier test
access. No checkpoint code or tensor object is deserialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


class ProtocolValidationError(ValueError):
    """The supplied artifacts are not the ones recorded by the protocol lock."""


def sha256(path: Path) -> str:
    if not path.is_file():
        raise ProtocolValidationError(f"Required file is missing: {path}")
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolValidationError(f"Required file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolValidationError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"Expected a JSON object: {path}")
    return value


def expected_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ProtocolValidationError(f"Missing or malformed SHA256 for {label}")
    return value.lower()


def check_hash(path: Path, expected: Any, label: str) -> str:
    wanted = expected_hash(expected, label)
    actual = sha256(path)
    if actual != wanted:
        raise ProtocolValidationError(f"{label} SHA256 differs from the locked artifact: {path}")
    return actual


def assert_locked(manifest: str | Path, checkpoint: str | Path,
                  protocol_lock: str | Path,
                  training_code: str | Path | None = None) -> dict[str, Any]:
    """Return verified provenance or raise ProtocolValidationError.

    Supported lock schemas are the local study's ``runs`` list and the public
    notebook's ``selected_checkpoints`` list. Membership is established using
    checkpoint bytes AND the arm/seed recorded in its hash-verified config.
    Relative path fields are descriptive so intact archives can be relocated.
    """
    manifest, checkpoint, protocol_lock = map(Path, (manifest, checkpoint, protocol_lock))
    training_code = Path(training_code) if training_code is not None else Path(__file__).with_name("experiment.py")
    lock = read_json(protocol_lock)
    config_path = checkpoint.parent / "config.json"
    config = read_json(config_path)

    manifest_hash = check_hash(manifest, lock.get("manifest_sha256"), "manifest")
    if expected_hash(config.get("manifest_sha256"), "run config manifest") != manifest_hash:
        raise ProtocolValidationError("Run config refers to a different manifest")

    recorded_sources = lock.get("source_sha256")
    source_hashes: dict[str, str] = {}
    if recorded_sources is not None:
        if not isinstance(recorded_sources, dict) or "experiment.py" not in recorded_sources:
            raise ProtocolValidationError("source_sha256 must contain experiment.py")
        for name, wanted in recorded_sources.items():
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                raise ProtocolValidationError("Source hash keys must be simple file names")
            source_path = training_code if name == "experiment.py" else training_code.parent / name
            source_hashes[name] = check_hash(source_path, wanted, f"source {name}")
        training_hash = source_hashes["experiment.py"]
        if "training_code_sha256" in lock:
            if expected_hash(lock["training_code_sha256"], "training code") != training_hash:
                raise ProtocolValidationError("The lock contains inconsistent training-code hashes")
    else:
        training_hash = check_hash(training_code, lock.get("training_code_sha256"), "training code")
        source_hashes["experiment.py"] = training_hash
    if expected_hash(config.get("training_code_sha256"), "run config training code") != training_hash:
        raise ProtocolValidationError("Run config refers to different training code")

    has_local = "runs" in lock
    has_notebook = "selected_checkpoints" in lock
    if has_local == has_notebook:
        raise ProtocolValidationError("Expected exactly one lock schema: runs or selected_checkpoints")
    entries = lock["runs"] if has_local else lock["selected_checkpoints"]
    if not isinstance(entries, list) or not entries or not all(isinstance(item, dict) for item in entries):
        raise ProtocolValidationError("The protocol must contain a nonempty checkpoint selection list")

    arm, seed = config.get("arm"), config.get("seed")
    if arm not in {"frozen", "partial"} or not isinstance(seed, int) or isinstance(seed, bool):
        raise ProtocolValidationError("Run config has an invalid arm or seed")
    matches = [entry for entry in entries if entry.get("arm") == arm and entry.get("seed") == seed]
    if len(matches) != 1:
        raise ProtocolValidationError(f"Checkpoint run {arm}, seed {seed} must occur exactly once in the lock")
    selected = matches[0]
    checkpoint_key = "checkpoint_sha256" if has_local else "sha256"
    checkpoint_hash = check_hash(checkpoint, selected.get(checkpoint_key), "selected checkpoint")
    config_hash = check_hash(config_path, selected.get("config_sha256"), "run config")

    summary_hash = None
    if "summary_sha256" in selected:
        summary_hash = check_hash(checkpoint.parent / "summary.json", selected["summary_sha256"], "run summary")

    settings = lock.get("settings", {})
    if not isinstance(settings, dict):
        raise ProtocolValidationError("settings must be a JSON object")
    wanted_model = settings.get("model_id", lock.get("model"))
    wanted_revision = settings.get("model_revision", lock.get("revision"))
    if wanted_model is not None and config.get("model_id") != wanted_model:
        raise ProtocolValidationError("Model identity differs from the protocol")
    if wanted_revision is not None and config.get("requested_revision") != wanted_revision:
        raise ProtocolValidationError("Model revision differs from the protocol")
    allowed_seeds = settings.get("seeds", lock.get("seeds"))
    if allowed_seeds is not None and (not isinstance(allowed_seeds, list) or seed not in allowed_seeds):
        raise ProtocolValidationError("Run seed is not among the planned seeds")
    for config_key, setting_key in (("batch_size", "batch_size"), ("max_epochs", "epochs"),
                                    ("patience", "patience"), ("head_lr", "head_lr"),
                                    ("backbone_lr", "backbone_lr"), ("weight_decay", "weight_decay")):
        if setting_key in settings and config.get(config_key) != settings[setting_key]:
            raise ProtocolValidationError(f"Run setting differs from the protocol: {config_key}")

    return {"status": "provenance_verified", "schema": "local_runs" if has_local else "notebook_checkpoints",
            "arm": arm, "seed": seed, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
            "summary_sha256": summary_hash, "manifest_sha256": manifest_hash,
            "training_code_sha256": training_hash, "source_hashes_verified": source_hashes,
            "protocol_lock_sha256": sha256(protocol_lock),
            "scope": "Artifact consistency with this lock; not independent proof of lock chronology or prior test access"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--training-code", type=Path, default=Path(__file__).with_name("experiment.py"))
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file")
    args = parser.parse_args()
    try:
        result = assert_locked(args.manifest, args.checkpoint, args.protocol_lock, args.training_code)
    except ProtocolValidationError as error:
        parser.exit(2, f"PROTOCOL VALIDATION FAILED: {error}\n")
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
