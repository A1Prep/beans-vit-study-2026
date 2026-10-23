"""Acquire the published Beans split and audit images before any model training.

Dependencies: requests, Pillow, numpy. No Hugging Face account is needed.
The image manifest uses absolute paths; rerun with --data-root after moving data.
Near-duplicate candidates are screened, not automatically treated as duplicates.
Human/visual review decisions may be supplied in review_decisions.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPOSITORY = "AI-Lab-Makerere/beans"
REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
LABELS = {"angular_leaf_spot": 0, "bean_rust": 1, "healthy": 2}
DCT_LOW = 2 * np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32)[None, :] + 1) / 64)
EXPECTED = {"train": 1034, "validation": 133, "test": 128}
ARCHIVE_SHA256 = {
    "train": "284fe8456ce20687f4367ae7ad94a64577e7f9fde2c2c6b1c74340ab5dc82715",
    "validation": "90b7aa1c26d91d9afff07a30bbc67a5ea34f1f1397f068d8675be09d7d0c602d",
    "test": "ca67b15d960d1e2fd9d23fd8498ce86818ead90755c630a43baf19ec4af09312",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def download(url, path, expected_hash):
    if path.exists() and sha256_file(path) == expected_hash:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    if sha256_file(temporary) != expected_hash:
        raise RuntimeError(f"Archive checksum mismatch: {path.name}")
    temporary.replace(path)


def perceptual_hash(image):
    # Standard 64-bit pHash: 32x32 luminance, 2D DCT, median of 8x8 low frequencies.
    array = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = DCT_LOW @ array @ DCT_LOW.T
    bits = (low > np.median(low)).reshape(-1)
    return int("".join("1" if bit else "0" for bit in bits), 2)


class UnionFind:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        self.parent[self.find(right)] = self.find(left)

    def groups(self):
        output = defaultdict(list)
        for value in self.parent:
            output[self.find(value)].append(value)
        return [sorted(group) for group in output.values() if len(group) > 1]


def font(size=16):
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def thumbnail(path, size):
    with Image.open(path) as original:
        return ImageOps.contain(ImageOps.exif_transpose(original).convert("RGB"), size)


def make_contact_sheets(candidates, lookup, output):
    # Six image pairs per page; image identifiers stay visible for review.
    pages = []
    for start in range(0, len(candidates), 6):
        batch = candidates[start:start + 6]
        sheet = Image.new("RGB", (1100, 90 + 300 * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), "Beans: pHash candidate pairs (not confirmed duplicates)", font=font(23), fill="black")
        for n, pair in enumerate(batch):
            y = 70 + n * 300
            draw.text((15, y), f"{pair['pair_id']} | Hamming distance {pair['distance']}", font=font(17), fill="black")
            for side, key in enumerate(("image_id_a", "image_id_b")):
                item = lookup[pair[key]]
                x = 20 + side * 540
                image = thumbnail(item["path"], (290, 245))
                sheet.paste(image, (x, y + 30))
                draw.text((x + 300, y + 35), item["split"], font=font(17), fill="black")
                draw.text((x + 300, y + 63), item["label_name"], font=font(14), fill="black")
                filename = Path(item["path"]).name
                for line, offset in zip((filename[:25], filename[25:50]), (92, 113)):
                    draw.text((x + 300, y + offset), line, font=font(14), fill="black")
        path = output / f"near_duplicate_candidates_{start // 6 + 1:02d}.png"
        sheet.save(path)
        pages.append(path.name)
    return pages


def make_sample_figure(rows, output):
    image = Image.new("RGB", (1050, 940), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 14), "Beans dataset: training examples", font=font(28), fill="#163329")
    draw.text((24, 50), "First three included training identifiers per class; published splits retained", font=font(17), fill="#404040")
    for row_num, name in enumerate(LABELS):
        samples = [r for r in rows if r["split"] == "train" and r["label_name"] == name and r["include"] == 1][:3]
        for column, sample in enumerate(samples):
            x, y = 24 + column * 345, 90 + row_num * 280
            thumb = thumbnail(sample["path"], (260, 225))
            image.paste(thumb, (x, y))
            draw.text((x, y + 230), name.replace("_", " "), font=font(18), fill="#163329")
            draw.text((x, y + 254), Path(sample["path"]).name, font=font(13), fill="#444444")
    image.save(output / "dataset_examples.png")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project.parents[1] / "work/data/beans")
    parser.add_argument("--output-dir", type=Path, default=project / "results/data_audit")
    parser.add_argument("--phash-threshold", type=int, default=12)
    parser.add_argument("--review-decisions", type=Path)
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    (root / "archives").mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources, rows, errors = [], [], []
    for split in EXPECTED:
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/data/{split}.zip"
        archive = root / "archives" / f"{split}.zip"
        print(f"Acquiring {split} archive", flush=True)
        download(url, archive, ARCHIVE_SHA256[split])
        sources.append({"split": split, "url": url, "sha256": sha256_file(archive), "bytes": archive.stat().st_size})
        with zipfile.ZipFile(archive) as zipped:
            names = sorted(x for x in zipped.namelist() if Path(x).suffix.lower() in {".jpg", ".jpeg", ".png"} and "__MACOSX" not in x)
            if len(names) != EXPECTED[split]:
                raise RuntimeError(f"Unexpected image count for {split}: {len(names)}")
            for name in names:
                components = Path(name).parts
                matches = [x for x in components if x in LABELS]
                if len(matches) != 1:
                    raise RuntimeError(f"Cannot map class from archive member: {name}")
                label_name = matches[0]
                image_id = f"{split}/{label_name}/{Path(name).name}"
                destination = root / "images" / image_id
                destination.parent.mkdir(parents=True, exist_ok=True)
                contents = zipped.read(name)
                if not destination.exists() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(contents).digest():
                    destination.write_bytes(contents)
                row = {"image_id": image_id, "split": split, "label": LABELS[label_name], "label_name": label_name,
                       "path": str(destination.resolve()), "include": 1, "reason": "", "file_sha256": hashlib.sha256(contents).hexdigest()}
                try:
                    with Image.open(io.BytesIO(contents)) as raw:
                        raw.load()
                        row.update(width=raw.width, height=raw.height, mode=raw.mode)
                        rgb = ImageOps.exif_transpose(raw).convert("RGB")
                        row["pixel_sha256"] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                        row["phash"] = f"{perceptual_hash(rgb):016x}"
                except Exception as error:
                    row.update(include=0, reason="corrupt_image", pixel_sha256="", phash="")
                    errors.append({"image_id": image_id, "error": str(error)})
                rows.append(row)
    rows.sort(key=lambda r: r["image_id"])
    lookup = {r["image_id"]: r for r in rows}
    union = UnionFind(lookup)
    exact_groups = []
    for field in ("file_sha256", "pixel_sha256"):
        buckets = defaultdict(list)
        for row in rows:
            if row.get(field):
                buckets[row[field]].append(row["image_id"])
        for digest, ids in buckets.items():
            if len(ids) > 1:
                exact_groups.append({"type": field, "hash": digest, "image_ids": sorted(ids)})
                for identifier in ids[1:]:
                    union.union(ids[0], identifier)
    candidates = []
    valid = [r for r in rows if r["phash"]]
    for i, left in enumerate(valid):
        for right in valid[i + 1:]:
            if union.find(left["image_id"]) == union.find(right["image_id"]):
                continue
            distance = (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count()
            if distance <= args.phash_threshold:
                candidates.append({"image_id_a": left["image_id"], "image_id_b": right["image_id"], "distance": distance,
                                   "split_a": left["split"], "split_b": right["split"], "cross_split": int(left["split"] != right["split"]),
                                   "label_a": left["label_name"], "label_b": right["label_name"]})
    candidates.sort(key=lambda r: (r["distance"], r["image_id_a"], r["image_id_b"]))
    for index, pair in enumerate(candidates, 1):
        pair["pair_id"] = f"pair_{index:04d}"
    review_path = args.review_decisions or output / "review_decisions.csv"
    old_reviews = []
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            old_reviews = list(csv.DictReader(stream))
    old_lookup = {(r["image_id_a"], r["image_id_b"]): r for r in old_reviews}
    reviews = []
    for pair in candidates:
        review = {"pair_id": pair["pair_id"], "image_id_a": pair["image_id_a"], "image_id_b": pair["image_id_b"], "decision": "pending", "notes": ""}
        old = old_lookup.get((pair["image_id_a"], pair["image_id_b"]))
        if old:
            review.update(decision=old["decision"], notes=old["notes"])
        if review["decision"] not in {"pending", "distinct", "uncertain", "confirmed_duplicate"}:
            raise ValueError(f"Invalid review decision: {review}")
        if review["decision"] == "confirmed_duplicate":
            union.union(review["image_id_a"], review["image_id_b"])
        reviews.append(review)
    save_csv(review_path, reviews, ["pair_id", "image_id_a", "image_id_b", "decision", "notes"])
    groups = union.groups()
    for ids in groups:
        labels = {lookup[x]["label"] for x in ids}
        if len(labels) > 1:
            for identifier in ids:
                lookup[identifier].update(include=0, reason="unresolved_duplicate_label_conflict")
            continue
        # Protect test first, then validation. Within a split keep smallest identifier.
        priority = {"test": 0, "validation": 1, "train": 2}
        keep = min(ids, key=lambda x: (priority[lookup[x]["split"]], x))
        for identifier in ids:
            if identifier != keep:
                lookup[identifier].update(include=0, reason=f"confirmed_duplicate_of:{keep}")
    manifest_fields = ["image_id", "split", "label", "label_name", "path", "include", "reason"]
    save_csv(output / "manifest.csv", rows, manifest_fields)
    save_csv(output / "image_metadata.csv", rows, manifest_fields + ["width", "height", "mode", "file_sha256", "pixel_sha256", "phash"])
    save_csv(output / "near_duplicate_candidates.csv", candidates, ["pair_id", "image_id_a", "image_id_b", "distance", "split_a", "split_b", "cross_split", "label_a", "label_b"])
    counts = [{"split": split, "label": label, "label_name": name,
               "original_count": sum(r["split"] == split and r["label"] == label for r in rows),
               "included_count": sum(r["split"] == split and r["label"] == label and r["include"] == 1 for r in rows)}
              for split in EXPECTED for name, label in LABELS.items()]
    save_csv(output / "class_counts.csv", counts, ["split", "label", "label_name", "original_count", "included_count"])
    save_json(output / "source_manifest.json", {"repository": REPOSITORY, "revision": REVISION, "archives": sources,
              "original_project": "https://github.com/AI-Lab-Makerere/ibean", "license": "MIT"})
    save_json(output / "exact_duplicate_groups.json", exact_groups)
    save_json(output / "confirmed_duplicate_groups.json", groups)
    pages = make_contact_sheets(candidates, lookup, output)
    make_sample_figure(rows, output)
    review_status = Counter(r["decision"] for r in reviews)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "dataset_revision": REVISION,
               "expected_counts": EXPECTED, "original_counts": dict(Counter(r["split"] for r in rows)),
               "included_counts": dict(Counter(r["split"] for r in rows if r["include"] == 1)),
               "excluded_images": [{k: r[k] for k in manifest_fields if k != "path"} for r in rows if r["include"] == 0],
               "class_counts": counts, "corrupt_images": errors,
               "dimension_counts": dict(Counter(f"{r.get('width')}x{r.get('height')}" for r in rows)),
               "mode_counts": dict(Counter(r.get("mode", "unknown") for r in rows)),
               "exact_hash_groups": exact_groups, "confirmed_duplicate_groups": groups,
               "phash_method": "32x32 luminance; DCT-II low 8x8; median threshold; 64-bit Hamming distance",
               "phash_threshold": args.phash_threshold, "near_duplicate_candidate_pairs": len(candidates),
               "cross_split_candidate_pairs": sum(p["cross_split"] for p in candidates),
               "within_test_candidate_pairs": sum(p["split_a"] == p["split_b"] == "test" for p in candidates),
               "visual_review_status": dict(review_status), "candidate_contact_sheets": pages,
               "limitations": ["pHash screening cannot rule out all near-duplicates or photographs of the same plant.",
                               "Farm, plant, person and collection-session identifiers are unavailable in this release.",
                               "Unknown overlap with model pretraining data is not measured.",
                               "Source repository reports 1296 images; the pinned distributed splits contain 1295."]}
    save_json(output / "audit_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["original_counts", "included_counts", "corrupt_images", "near_duplicate_candidate_pairs", "cross_split_candidate_pairs", "within_test_candidate_pairs", "visual_review_status"]}, indent=2), flush=True)
    print(f"Manifest: {output / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
