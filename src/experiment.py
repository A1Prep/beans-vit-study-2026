"""Controlled Beans experiments: frozen DINO features versus partial fine-tuning.

Training never evaluates the test split. Use the separate ``evaluate`` command
only after the development protocol has been locked. Image augmentation and
sample order are deterministic functions of seed/epoch/image ID, independent of
the model's random-number consumption. Requires torch, transformers, Pillow,
and numpy; no torchvision or remote executable model code is used.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    text = "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn rather than fail on a platform-specific nondeterministic operation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_manifest(path: Path, data_root: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    required = {"image_id", "split", "label", "label_name", "path", "include"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"Manifest must contain rows and columns {sorted(required)}")
    included = []
    identifiers = set()
    for row in records:
        if str(row["include"]).strip().lower() not in {"true", "1", "yes"}:
            continue
        row = dict(row)
        row["label"] = int(row["label"])
        if row["label"] not in range(3) or row["label_name"] != CLASS_NAMES[row["label"]]:
            raise ValueError(f"Unexpected label mapping: {row}")
        if row["image_id"] in identifiers:
            raise ValueError(f"Repeated included image_id: {row['image_id']}")
        identifiers.add(row["image_id"])
        image_path = Path(row["path"])
        resolved = image_path if image_path.is_absolute() else data_root / image_path
        # Archived manifests may retain an original machine's absolute path.
        # image_id is the stable split/class/file path for a relocated dataset.
        if not resolved.is_file() and (data_root / row["image_id"]).is_file():
            resolved = data_root / row["image_id"]
        row["resolved_path"] = str(resolved)
        included.append(row)
    return sorted(included, key=lambda row: row["image_id"])


class BeansImages(Dataset):
    def __init__(self, records: list[dict[str, Any]], preprocessing: dict[str, Any],
                 training: bool = False, seed: int = 17, epoch: int = 0,
                 corruption: str = "clean"):
        self.records = records
        self.preprocessing = preprocessing
        self.training = training
        self.seed = seed
        self.epoch = epoch
        self.corruption = corruption
        if training and corruption != "clean":
            raise ValueError("Corruptions are evaluation-only")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        with Image.open(row["resolved_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        size = int(self.preprocessing["size"])
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        if self.training:
            rng = random.Random(stable_seed("augmentation", self.seed, self.epoch, row["image_id"]))
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
            if rng.random() < 0.5:
                image = ImageOps.flip(image)
            angle = rng.uniform(-15.0, 15.0)
            fill = tuple(round(float(x) * 255) for x in self.preprocessing["mean"])
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)
        if self.corruption == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif self.corruption == "brightness":
            image = ImageEnhance.Brightness(image).enhance(0.7)
        elif self.corruption != "clean":
            raise ValueError(f"Unknown corruption {self.corruption}")
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        mean = np.asarray(self.preprocessing["mean"], dtype=np.float32)
        std = np.asarray(self.preprocessing["std"], dtype=np.float32)
        tensor = torch.from_numpy(((array - mean) / std).transpose(2, 0, 1).copy())
        return {"pixel_values": tensor, "labels": row["label"],
                "image_id": row["image_id"], "label_name": row["label_name"]}


def make_loader(dataset: BeansImages, batch_size: int, workers: int,
                device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(stable_seed("sample_order", dataset.seed, dataset.epoch) % (2**63 - 1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.training,
                      generator=generator, num_workers=workers,
                      pin_memory=device.type == "cuda", drop_last=False,
                      persistent_workers=False)


def transformer_blocks(backbone: nn.Module) -> tuple[nn.ModuleList, str]:
    for name in ("encoder.layer", "encoder.layers", "layer", "layers", "blocks"):
        module: Any = backbone
        for part in name.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) >= 2:
            return module, name
    raise ValueError("Cannot locate transformer blocks; inspect the backbone before training")


def final_normalization(backbone: nn.Module) -> tuple[nn.Module, str]:
    for name in ("layernorm", "norm", "final_layer_norm", "ln_f"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module):
            return module, name
    raise ValueError("Cannot locate the backbone final normalization")


class BeansClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, arm: str, head_seed: int):
        super().__init__()
        if arm not in {"frozen", "partial"}:
            raise ValueError("Arm must be frozen or partial")
        self.backbone = backbone
        self.arm = arm
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.unfrozen_modules: list[str] = []
        if arm == "partial":
            blocks, block_name = transformer_blocks(backbone)
            for index in range(len(blocks) - 2, len(blocks)):
                for parameter in blocks[index].parameters():
                    parameter.requires_grad_(True)
                self.unfrozen_modules.append(f"{block_name}.{index}")
            norm, name = final_normalization(backbone)
            for parameter in norm.parameters():
                parameter.requires_grad_(True)
            self.unfrozen_modules.append(name)
        # Head initialisation does not depend on the loading code's RNG usage.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.classifier = nn.Linear(int(backbone.config.hidden_size), 3)

    def train(self, mode: bool = True) -> "BeansClassifier":
        super().train(mode)
        # Gradient tracking is independent of eval mode. This disables stochastic
        # backbone layers consistently in both arms while allowing partial updates.
        self.backbone.eval()
        self.classifier.train(mode)
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.arm == "frozen" else contextlib.nullcontext()
        with context:
            output = self.backbone(pixel_values=pixel_values, return_dict=True)
            features = output.last_hidden_state[:, 0, :]
        return self.classifier(features)


def load_pretrained(args: argparse.Namespace, arm: str, seed: int) -> tuple[BeansClassifier, dict[str, Any]]:
    from transformers import AutoImageProcessor, AutoModel
    source = args.local_model_path or args.model_id
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if args.local_model_path:
        kwargs["local_files_only"] = True
    elif args.revision:
        kwargs["revision"] = args.revision
    processor = AutoImageProcessor.from_pretrained(source, **kwargs)
    backbone = AutoModel.from_pretrained(source, **kwargs)
    model_type = str(backbone.config.model_type)
    if model_type not in {"dinov2", "dinov2_with_registers", "dinov3_vit"}:
        raise ValueError(f"This controlled experiment supports DINO ViTs; got {model_type}")
    preprocessing = {"size": 224, "mean": list(processor.image_mean),
                     "std": list(processor.image_std),
                     "resize": "PIL bicubic direct square resize", "crop": "none",
                     "augmentation": "horizontal/vertical flip p=.5 each; rotation uniform[-15,15]; mean-colour fill"}
    model = BeansClassifier(backbone, arm=arm, head_seed=seed + 1_000_000)
    return model, preprocessing


def make_optimizer(model: BeansClassifier, head_lr: float, backbone_lr: float,
                   weight_decay: float) -> torch.optim.Optimizer:
    groups = [{"params": list(model.classifier.parameters()), "lr": head_lr, "name": "head"}]
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def metrics_from_predictions(labels: list[int] | np.ndarray,
                             predictions: list[int] | np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot evaluate an empty split")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    true_positive = np.diag(matrix).astype(float)
    precision = np.divide(true_positive, matrix.sum(axis=0), out=np.zeros(3), where=matrix.sum(axis=0) != 0)
    recall = np.divide(true_positive, matrix.sum(axis=1), out=np.zeros(3), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) != 0)
    return {"n": int(labels.size), "accuracy": float((labels == predictions).mean()),
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
            "per_class": [{"label": i, "name": CLASS_NAMES[i], "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]),
                           "support": int(matrix[i].sum())} for i in range(3)]}


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model: BeansClassifier, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scaler: Any, device: torch.device, amp: bool,
                max_batches: int | None = None, verify_gradients: bool = False) -> dict[str, Any]:
    model.train()
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    gradient_report = None
    synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp):
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if verify_gradients and batch_index == 0:
            missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            forbidden = [name for name, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            nonfinite = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if missing or forbidden or nonfinite:
                raise AssertionError({"missing_gradients": missing, "frozen_gradients": forbidden,
                                      "nonfinite_gradients": nonfinite})
            gradient_report = {"all_trainable_have_gradients": True,
                               "no_frozen_gradients": True, "all_gradients_finite": True}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        size = target.shape[0]
        total_loss += float(loss.detach()) * size
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    synchronize(device)
    result = metrics_from_predictions(labels, predictions)
    result.update(loss=total_loss / len(labels), seconds=time.perf_counter() - started)
    if gradient_report:
        result["gradient_checks"] = gradient_report
    return result


@torch.inference_mode()
def predict(model: BeansClassifier, loader: DataLoader, device: torch.device,
            amp: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows = []
    total_loss = 0.0
    synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["labels"].to(device, non_blocking=True)
        with amp_context(device, amp):
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, target, reduction="sum")
        probabilities = logits.float().softmax(dim=1).cpu().numpy()
        total_loss += float(loss)
        for index, probability in enumerate(probabilities):
            label = int(batch["labels"][index])
            prediction = int(probability.argmax())
            rows.append({"image_id": batch["image_id"][index], "label": label,
                         "label_name": CLASS_NAMES[label], "predicted_label": prediction,
                         "predicted_name": CLASS_NAMES[prediction],
                         "confidence": float(probability[prediction]),
                         **{f"prob_{i}": float(probability[i]) for i in range(3)}})
    synchronize(device)
    result = metrics_from_predictions([r["label"] for r in rows], [r["predicted_label"] for r in rows])
    result.update(loss=total_loss / len(rows), seconds=time.perf_counter() - started)
    return result, rows


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    rows = [{"epoch": entry["epoch"], "train_loss": entry["train"]["loss"],
             "train_accuracy": entry["train"]["accuracy"],
             "val_loss": entry["validation"]["loss"],
             "val_accuracy": entry["validation"]["accuracy"],
             "val_macro_f1": entry["validation"]["macro_f1"],
             "epoch_seconds": entry["train"]["seconds"] + entry["validation"]["seconds"],
             "train_seconds": entry["train"]["seconds"],
             "validation_seconds": entry["validation"]["seconds"],
             "saved_best": entry["saved_best"], "patience_counter": entry["patience_counter"]}
            for entry in history]
    write_predictions(path, rows)


def save_checkpoint(path: Path, model: BeansClassifier, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"format_version": 1,
                "backbone_config": model.backbone.config.to_dict(),
                "metadata": metadata,
                "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[BeansClassifier, dict[str, Any]]:
    from transformers import AutoConfig, AutoModel
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config_data = dict(saved["backbone_config"])
    model_type = config_data.pop("model_type")
    config = AutoConfig.for_model(model_type, **config_data)
    backbone = AutoModel.from_config(config, trust_remote_code=False)
    metadata = saved["metadata"]
    model = BeansClassifier(backbone, metadata["arm"], int(metadata["seed"]) + 1_000_000)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    return model, metadata


class CheckpointSelection:
    """Primary-metric patience is separate from loss tie-breaking."""
    def __init__(self, patience: int):
        self.patience = patience
        self.best_f1 = -math.inf
        self.best_loss = math.inf
        self.best_epoch = 0
        self.stale = 0

    def update(self, epoch: int, f1: float, loss: float) -> tuple[bool, bool]:
        if not math.isfinite(f1) or not math.isfinite(loss):
            raise FloatingPointError("Validation metric is non-finite")
        strict_improvement = f1 > self.best_f1
        better_checkpoint = strict_improvement or (f1 == self.best_f1 and loss < self.best_loss)
        self.stale = 0 if strict_improvement else self.stale + 1
        if better_checkpoint:
            self.best_f1, self.best_loss, self.best_epoch = f1, loss, epoch
        return better_checkpoint, self.stale >= self.patience


def runtime_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch environment has no working CUDA device")
    return torch.device(name)


def environment(device: torch.device) -> dict[str, Any]:
    versions = {}
    for package in ("torch", "transformers", "numpy", "Pillow", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    result = {"device": str(device), "packages": versions,
              "torch_cuda_version": torch.version.cuda,
              "torch_threads": torch.get_num_threads()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory)
    return result


def parameter_metadata(model: BeansClassifier) -> dict[str, Any]:
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
            "unfrozen_modules": model.unfrozen_modules}


def run_metadata(args: argparse.Namespace, model: BeansClassifier,
                 preprocessing: dict[str, Any], arm: str, seed: int,
                 device: torch.device) -> dict[str, Any]:
    head_digest = hashlib.sha256()
    for name, parameter in model.classifier.named_parameters():
        head_digest.update(name.encode("utf-8"))
        head_digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return {"arm": arm, "seed": seed, "head_initialization_seed": seed + 1_000_000,
            "initial_head_sha256": head_digest.hexdigest(),
            "model_id": args.model_id, "requested_revision": args.revision,
            "resolved_revision": getattr(model.backbone.config, "_commit_hash", None) or args.revision,
            "local_model_path": args.local_model_path,
            "manifest_sha256": file_sha256(Path(args.manifest)),
            "training_code_sha256": file_sha256(Path(__file__)),
            "preprocessing": preprocessing, "pooling": "final last_hidden_state CLS token",
            "optimizer": "AdamW", "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay, "gradient_clip_norm": 1.0,
            "scheduler": "constant", "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience,
            "patience_rule": "strict validation macro-F1 improvement only",
            "checkpoint_rule": "higher validation macro-F1; lower loss on exact tie; earlier epoch",
            "amp": device.type == "cuda" and not args.no_amp,
            "amp_initial_scale": 1024.0,
            "backbone_mode": "eval in both arms; partial gradients enabled",
            "deterministic_algorithms": "enabled with warn_only=True; cross-device bitwise equality is not promised",
            "classes": CLASS_NAMES, **parameter_metadata(model), "environment": environment(device)}


def split_records(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    result = [row for row in records if row["split"] == split]
    if not result:
        raise ValueError(f"No included records for split={split}")
    return result


def train_run(args: argparse.Namespace, records: list[dict[str, Any]], arm: str,
              seed: int, device: torch.device) -> dict[str, Any]:
    output = Path(args.output_dir) / arm / f"seed_{seed}"
    if (output / "config.json").exists() and not args.overwrite:
        raise FileExistsError(f"Run already exists: {output}; use another output directory")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    started = time.perf_counter()
    model, preprocessing = load_pretrained(args, arm, seed)
    model.to(device)
    metadata = run_metadata(args, model, preprocessing, arm, seed, device)
    train_records = split_records(records, "train")
    validation_records = split_records(records, "validation")
    metadata["split_counts"] = {"train": len(train_records), "validation": len(validation_records)}
    write_json(output / "config.json", metadata)
    optimizer = make_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
    amp = metadata["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    selection = CheckpointSelection(args.patience)
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    validation_loader = make_loader(BeansImages(validation_records, preprocessing), args.batch_size,
                                    args.workers, device)
    for epoch in range(1, args.epochs + 1):
        dataset = BeansImages(train_records, preprocessing, training=True, seed=seed, epoch=epoch)
        loader = make_loader(dataset, args.batch_size, args.workers, device)
        training = train_epoch(model, loader, optimizer, scaler, device, amp,
                               verify_gradients=(epoch == 1))
        validation, rows = predict(model, validation_loader, device, amp)
        save_best, stop = selection.update(epoch, validation["macro_f1"], validation["loss"])
        entry = {"epoch": epoch, "train": training, "validation": validation,
                 "saved_best": save_best, "patience_counter": selection.stale}
        history.append(entry)
        if save_best:
            best_metadata = {**metadata, "epoch": epoch, "validation": validation}
            save_checkpoint(output / "best.pt", model, best_metadata)
            write_predictions(output / "validation_predictions.csv", rows)
        write_json(output / "history.json", history)
        write_history(output / "history.csv", history)
        print(json.dumps({"event": "epoch", "arm": arm, "seed": seed, "epoch": epoch,
                          "train_loss": training["loss"], "validation_loss": validation["loss"],
                          "validation_macro_f1": validation["macro_f1"],
                          "train_seconds": training["seconds"], "best_epoch": selection.best_epoch}), flush=True)
        if stop:
            break
    summary = {"arm": arm, "seed": seed, "epochs_completed": len(history),
               "best_epoch": selection.best_epoch, "best_validation_macro_f1": selection.best_f1,
               "best_val_macro_f1": selection.best_f1,
               "best_validation_loss": selection.best_loss, "wall_seconds": time.perf_counter() - started,
               "training_seconds": sum(row["train"]["seconds"] for row in history),
               "validation_seconds": sum(row["validation"]["seconds"] for row in history),
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "total_params": sum(p.numel() for p in model.parameters()),
               **parameter_metadata(model), "test_evaluated": False}
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "run_complete", **{k: v for k, v in summary.items() if k != "trainable_names"}}), flush=True)
    del model, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def pilot(args: argparse.Namespace, records: list[dict[str, Any]], device: torch.device) -> None:
    output = Path(args.output_dir) / "pilot"
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for arm in ("frozen", "partial"):
        set_seed(args.seed)
        model, preprocessing = load_pretrained(args, arm, args.seed)
        model.to(device)
        metadata = run_metadata(args, model, preprocessing, arm, args.seed, device)
        before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        dataset = BeansImages(split_records(records, "train"), preprocessing, True, args.seed, 1)
        loader = make_loader(dataset, args.batch_size, args.workers, device)
        optimizer = make_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
        amp = metadata["amp"]
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = train_epoch(model, loader, optimizer, scaler, device, amp,
                             max_batches=None if args.benchmark_epoch else args.pilot_batches,
                             verify_gradients=True)
        frozen_changed = [name for name, p in model.named_parameters()
                          if not p.requires_grad and not torch.equal(before[name], p.detach().cpu())]
        trainable_changed = [name for name, p in model.named_parameters()
                             if p.requires_grad and not torch.equal(before[name], p.detach().cpu())]
        if frozen_changed or not any(name.startswith("classifier.") for name in trainable_changed):
            raise AssertionError({"frozen_changed": frozen_changed, "trainable_changed": trainable_changed})
        if arm == "partial" and not any(name.startswith("backbone.") for name in trainable_changed):
            raise AssertionError("Partial backbone parameters did not update")
        model.eval()
        inputs = dataset[0]["pixel_values"].unsqueeze(0).to(device)
        with torch.inference_mode():
            expected = model(inputs).detach().cpu()
        checkpoint = output / f"{arm}_roundtrip.pt"
        save_checkpoint(checkpoint, model, metadata)
        del before, optimizer, scaler, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        restored, _ = load_checkpoint(checkpoint, device)
        with torch.inference_mode():
            observed = restored(inputs).detach().cpu()
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        report = {"arm": arm, "status": "passed", "epoch_benchmarked": args.benchmark_epoch,
                  "training": result, "frozen_parameters_unchanged": True,
                  "trainable_parameters_updated": len(trainable_changed),
                  "checkpoint_max_absolute_difference": float((observed - expected).abs().max()),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "metadata": metadata}
        reports.append(report)
        write_json(output / "pilot_report.json", reports)
        print(json.dumps({"event": "pilot", "arm": arm, "status": "passed", "seconds": result["seconds"],
                          "n": result["n"], "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"]}), flush=True)
        del restored
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate(args: argparse.Namespace, records: list[dict[str, Any]], device: torch.device) -> None:
    if args.split == "test" and not args.confirm_test_locked:
        raise ValueError("Test evaluation requires --confirm-test-locked after development decisions are final")
    if args.split == "test" and (not args.protocol_lock or not Path(args.protocol_lock).is_file()):
        raise ValueError("Test evaluation also requires --protocol-lock pointing to the saved protocol.lock.json")
    checkpoint = Path(args.checkpoint)
    model, metadata = load_checkpoint(checkpoint, device)
    actual_manifest = file_sha256(Path(args.manifest))
    if actual_manifest != metadata["manifest_sha256"]:
        raise ValueError("Manifest differs from the training manifest; resolve the audit provenance first")
    dataset = BeansImages(split_records(records, args.split), metadata["preprocessing"], corruption=args.corruption)
    loader = make_loader(dataset, args.batch_size, args.workers, device)
    metrics, rows = predict(model, loader, device, amp=metadata.get("amp", False) and device.type == "cuda")
    metrics.update(arm=metadata["arm"], seed=metadata["seed"], split=args.split,
                   corruption=args.corruption, checkpoint_epoch=metadata["epoch"],
                   checkpoint_sha256=file_sha256(checkpoint), manifest_sha256=actual_manifest)
    if args.protocol_lock:
        metrics["protocol_lock_sha256"] = file_sha256(Path(args.protocol_lock))
    for row in rows:
        row.update(arm=metadata["arm"], seed=metadata["seed"], split=args.split, corruption=args.corruption)
        row["prediction"] = row["predicted_label"]
    output = Path(args.output_dir)
    stem = args.split + (f"_{args.corruption}" if args.corruption != "clean" else "")
    write_json(output / f"metrics_{stem}.json", metrics)
    write_predictions(output / f"predictions_{stem}.csv", rows)
    print(json.dumps({"event": "evaluation", "output_stem": stem, **metrics}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=["pilot", "train", "evaluate"])
    result.add_argument("--manifest", required=True)
    result.add_argument("--data-root", default=".")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--model-id", default="facebook/dinov2-small")
    result.add_argument("--revision", default=None)
    result.add_argument("--local-model-path", default=None)
    result.add_argument("--device", default="auto")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--seeds", default=None, help="Comma-separated paired seeds for train")
    result.add_argument("--arm", choices=["frozen", "partial", "both"], default="both")
    result.add_argument("--epochs", type=int, default=10)
    result.add_argument("--patience", type=int, default=3)
    result.add_argument("--head-lr", type=float, default=1e-3)
    result.add_argument("--backbone-lr", type=float, default=1e-5)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--pilot-batches", type=int, default=2)
    result.add_argument("--benchmark-epoch", action="store_true")
    result.add_argument("--checkpoint")
    result.add_argument("--split", choices=["validation", "test"], default="validation")
    result.add_argument("--corruption", choices=["clean", "blur", "brightness"], default="clean")
    result.add_argument("--confirm-test-locked", action="store_true")
    result.add_argument("--protocol-lock", help="Saved final development protocol; mandatory for test evaluation")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.pilot_batches < 1:
        raise ValueError("Epochs, patience, batch size, and pilot batches must be positive")
    torch.set_num_threads(args.threads)
    # Required by deterministic CUDA matrix operations; set before first CUDA use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = runtime_device(args.device)
    records = read_manifest(Path(args.manifest), Path(args.data_root))
    if args.command == "pilot":
        pilot(args, records, device)
    elif args.command == "train":
        seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else [args.seed]
        arms = ("frozen", "partial") if args.arm == "both" else (args.arm,)
        for seed in seeds:
            for arm in arms:
                train_run(args, records, arm, seed, device)
    else:
        if not args.checkpoint:
            raise ValueError("evaluate requires --checkpoint")
        evaluate(args, records, device)


if __name__ == "__main__":
    main()
