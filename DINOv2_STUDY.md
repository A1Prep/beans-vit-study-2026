> Historical study completed 23 October 2026. See [current comparison](README.md) for the newer-model extension.

# Bean-leaf disease classification with a pretrained vision transformer

An applied computer-vision experiment comparing **frozen DINOv2-Small features** with **partial fine-tuning of the final two transformer blocks and final normalization** on the published Makerere Beans dataset. Both approaches train the same three-class linear classifier using matched image preparation, augmentation, initialization and seed-specific sample order.

The research question is whether partial fine-tuning improves macro-F1 on this fixed split, how uncertain the difference is, and what additional training cost it requires. A frozen-model win or an inconclusive difference is a valid finding. This is a research proof of concept; it is not a validated agricultural diagnosis service.

## Evidence and current status

Start with the [technical report PDF](report/archive/dinov2_study_2026-10-23/technical_report.pdf), [editable report](report/archive/dinov2_study_2026-10-23/technical_report.md), [executed results notebook](results_review.ipynb), [presentation notes](report/archive/dinov2_study_2026-10-23/presentation_notes.md), or [project review and encountered issues](PROJECT_REVIEW.md).

See [results/README.md](results/README.md) for the execution status and available evidence. The notebook is an executable workflow; unexecuted cells and proposed settings are not experimental results. Only saved run summaries, predictions and analyses establish completed experiments.

The actual local study completed all six runs and final evaluation on 23 October 2026. Mean test macro-F1 was 90.71% for the frozen backbone and 97.16% for partial fine-tuning across the three paired seeds. The [saved results and limitations](results/README.md) explain seed variation, reference-run uncertainty and measured cost. Open the executed [results review notebook](results_review.ipynb) to inspect those metrics and figures without retraining.

A subsequent [overfitting audit](results/README.md#overfitting-audit) checked all six selected checkpoints. Partial fine-tuning had a mean clean training–test macro-F1 gap of 1.42 points. Some late epochs deteriorated; validation-selected checkpoints limited that deterioration. The audit is diagnostic and does not establish absence of overfitting.

Read the [technical report](report/archive/dinov2_study_2026-10-23/technical_report.md) for the research question, methods, findings, ethical implications and references.

The completed data audit found 1,295 readable, 500×500 RGB images, no exact file/pixel duplicates, and 14 pHash candidate pairs at distance ≤12. AI-assisted visual inspection classified those 14 pairs as distinct; no images were removed. This screen cannot rule out every near-duplicate or establish independence of plants, farms or collection sessions. The audit and per-pair decisions are included under `results/data_audit/`.

| Split | Angular leaf spot | Bean rust | Healthy | Total |
|---|---:|---:|---:|---:|
| Training | 345 | 348 | 341 | 1,034 |
| Validation | 44 | 45 | 44 | 133 |
| Test | 43 | 43 | 42 | 128 |

## Pinned inputs

- Dataset: [`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans), revision `27aa014ce09b193e1a6f58112d4a66e0eddb69c5`.
- Backbone: [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small), revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- The acquisition script verifies each original split ZIP against its pinned SHA256. It preserves the published split and records image-level hashes and identifiers.
- Use the general pretrained backbone, never a checkpoint already fine-tuned on Beans. Unknown overlap with foundation-model pretraining data remains unmeasured.

DINOv2-Small is an established 2023 ViT with accessible weights and a manageable parameter count. It is not described as the newest model in 2026 or as state of the art on Beans. DINOv3 was considered in the design, but its access conditions introduce a dependency; this reproducible implementation fixes DINOv2 in both arms. Changing backbones creates a different experiment and must be documented before test access.

## Run the notebook in Colab

1. Open `beans_vit.ipynb` in Google Colab and select a GPU runtime. GPU availability is not guaranteed.
2. Have the complete project ZIP available. If Colab opened only the notebook, its setup cell lets you upload the project ZIP and finds the source folder. A checked-out repository works directly.
3. Run setup and dependency cells. The Colab setup installs matching `torchvision==0.21.0` and `torchaudio==2.6.0` alongside the pinned PyTorch, preventing optional-import conflicts with Colab's preinstalled versions. If the installed packages differ from already imported versions, restart the runtime and run from the top.
4. Follow the cells in order: runtime check → data audit → pilot and checkpoint round trip → both training approaches → protocol lock → held-out test evaluation → analysis.
5. Save the generated result bundle before closing Colab. Colab runtime files are temporary; model checkpoints can be saved separately if needed.

Notebook reproductions use a new directory under `results/reproduction/` and do not overwrite the submitted results. Raw data and model caches stay under a separate `work/` directory. The notebook stops without a GPU unless CPU execution is explicitly enabled in its configuration; it does not silently replace the study with a different experiment.

The Colab workflow has been statically checked; it has not been executed on Colab. Local GPU verification is documented with the study results. The separate `results_review.ipynb` was executed successfully to read and display the saved actual results; that does not rerun training. See [notebook validation evidence](results/notebook_validation.json).

## Local environment

Use Python 3.11 in an isolated environment. The reference package versions are in `requirements.txt`.

```text
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. For an NVIDIA installation compatible with CUDA 12.4, the reference PyTorch wheel can be installed before the remaining requirements:

```text
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Other platforms should select a suitable PyTorch installation using the [official installer](https://pytorch.org/get-started/locally/). A CPU installation can run checks but may be substantially slower for the full experiment. The runtime and package versions are recorded with every training run.

## Command-line workflow

Run these commands from the project folder. The distinction between data roots is intentional: preparation receives the directory containing `archives/` and `images/`; training and analysis receive its `images/` subdirectory.

```text
python src/prepare_data.py --data-root work/data/beans --output-dir results/reproduction/example/data_audit --review-decisions results/data_audit/review_decisions.csv
python src/experiment.py pilot --manifest results/reproduction/example/data_audit/manifest.csv --data-root work/data/beans/images --output-dir results/reproduction/example --model-id facebook/dinov2-small --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --batch-size 8 --benchmark-epoch
python src/experiment.py train --manifest results/reproduction/example/data_audit/manifest.csv --data-root work/data/beans/images --output-dir results/reproduction/example/runs --model-id facebook/dinov2-small --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --batch-size 8 --arm both --seeds 17,42,2026 --epochs 10 --patience 3
```

Use the notebook's protocol-lock cell after development is complete. It checks that all planned run summaries and selected checkpoints exist, verifies settings against run configurations, then records their hashes, the data manifest, code and seeds. Run the standalone lock validator immediately before each test evaluation. It checks artifact consistency; it does not independently authenticate chronology or absence of prior test access:

```text
python src/validate_protocol.py --manifest results/reproduction/example/data_audit/manifest.csv --checkpoint results/reproduction/example/runs/frozen/seed_17/best.pt --protocol-lock results/reproduction/example/protocol.lock.json --training-code src/experiment.py
python src/experiment.py evaluate --manifest results/reproduction/example/data_audit/manifest.csv --data-root work/data/beans/images --output-dir results/reproduction/example/runs/frozen/seed_17 --checkpoint results/reproduction/example/runs/frozen/seed_17/best.pt --split test --confirm-test-locked --protocol-lock results/reproduction/example/protocol.lock.json --batch-size 8
```

Repeat evaluation for every planned arm and seed. The notebook automates that loop. Then:

```text
python src/analyse_results.py --runs results/reproduction/example/runs --manifest results/reproduction/example/data_audit/manifest.csv --data-root work/data/beans/images --out results/reproduction/example/analysis
python src/audit_overfitting.py --runs results/reproduction/example/runs --manifest results/reproduction/example/data_audit/manifest.csv --data-root work/data/beans/images --protocol-lock results/reproduction/example/protocol.lock.json --out results/reproduction/example/overfitting
python -m unittest discover -s tests -v
```

Existing completed/partial training directories are protected against accidental overwrite. Use a new run directory for a deliberate fresh experiment. Do not tune choices after looking at test results.

## Experimental controls and interpretation

- Label mapping: angular leaf spot = 0; bean rust = 1; healthy = 2.
- Both arms use 224×224 RGB inputs, checkpoint-specific normalization, the final CLS token and an identically initialized linear classifier for each paired seed.
- Training augmentation: horizontal and vertical flips with probability 0.5 each, plus rotation between −15° and +15°. Validation and test preparation are deterministic. Neither arm receives an augmentation advantage.
- Executed study settings: AdamW; head learning rate 0.001; partial-backbone learning rate 0.00001; weight decay 0.01; batch size 8; maximum 10 epochs; patience 3. Both measured full-epoch pilots passed before the full runs; all three paired seeds were completed.
- Checkpoint selection: highest validation macro-F1, then lower validation loss, then earlier epoch. Only a strict macro-F1 increase resets early-stopping patience.
- Executed paired seeds: 17, 42, 2026. Seed 17 is the prespecified detailed-analysis run. Fresh reproductions should document any reduction in repeats and must not invent missing runs.
- Completed epochs in that seed order were 10, 5, 10 for the frozen backbone and 5, 8, 10 for partial fine-tuning. Total training time therefore reflects different stopping points; [run metrics](results/run_metrics.csv) also record time per completed epoch.
- Primary metric: macro-F1. Also report accuracy, class-level precision/recall/F1, confusion matrices, a majority-class baseline, errors and measured computational cost.
- The reference-run macro-F1 difference uses 2,000 paired class-stratified bootstrap resamples with seed 20261020. This estimates test-sample uncertainty separately from training-seed variation. Unknown plant/farm dependence is not corrected by that interval.
- Prespecified sensitivity checks apply Gaussian blur (radius 1) and brightness ×0.7 after resizing, equally to both seed-17 models. The notebook previews validation examples before locking and evaluates transformed test images afterward. These are paired synthetic checks, not independent additional test samples or field validation.
- Softmax scores are confidence scores, not calibrated probabilities. The small 128-image test set and limited geographic/disease coverage constrain conclusions.

## Repository contents

| Path | Purpose |
|---|---|
| `beans_vit.ipynb` | Complete portable notebook workflow |
| `results_review.ipynb` | Read-only display of actual saved study evidence |
| `report/archive/dinov2_study_2026-10-23/technical_report.md` | Technical report with actual results and cited sources |
| `src/prepare_data.py` | Pinned download, integrity checks, dataset audit and manifest |
| `src/experiment.py` | Pilot, matched training, checkpointing and separate test evaluation |
| `src/validate_protocol.py` | Checkpoint membership and artifact-hash validation before evaluation |
| `src/analyse_results.py` | Metrics, paired uncertainty and figures from saved predictions |
| `src/audit_overfitting.py` | Post-hoc clean train/validation scoring and all-seed history diagnostics of unchanged checkpoints |
| `results/overfitting/` | Executed diagnostics, per-image predictions, split gaps and learning curves |
| `tests/` | Focused checks of experiment logic |
| `results/data_audit/` | Source provenance, counts, hashes, visual duplicate screening and sample figure |
| `results/` | Actual study evidence; status documented in its README |
| `ASSISTANCE.md` | AI-assistance disclosure and outstanding student verification |
| `THIRD_PARTY_NOTICES.md` | Original dataset copyright/license notice and model attribution |

Raw datasets, pretrained/trained weights, caches, environments and secrets are excluded from Git. Regenerate them with the scripts. Image paths in archived manifests may refer to the original machine; `--data-root` lets training and analysis resolve the stable `image_id` beneath a relocated `images/` directory. Do not rewrite a locked manifest after training, because its checksum is part of run provenance.

Dataset image excerpts in audit and result figures retain the attribution and original MIT notice in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are downloaded separately; the repository does not distribute them.

## Sources and responsible use

- [Makerere AI Lab: original bean disease dataset](https://github.com/AI-Lab-Makerere/ibean). Field smartphone images and expert annotations; dataset repository specifies the MIT license. The original README reports 1,296 images, while the pinned distributed splits contain 1,295; this project uses the audited distributed count.
- [Oquab et al., DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- [DINOv2-Small model card](https://huggingface.co/facebook/dinov2-small), with Apache-2.0 model licensing.
- [Dosovitskiy et al., An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929), for the ViT architecture.
- [Google Colab resource limitations](https://research.google.com/colaboratory/faq.html).

Discuss missed disease, false alarms, geographic coverage, nuisance cues such as backgrounds, unknown diseases, access to technology and the need for human expertise. Do not infer demographic fairness or practical diagnostic readiness from this benchmark. Cite and disclose any external code or AI assistance according to the course's rules.

This project was developed with extensive Codex AI assistance, including experimental planning, code generation, AI-assisted duplicate screening and separate agent reviews. Local execution produces real saved outputs; AI review does not replace student understanding or lecturer assessment. See [ASSISTANCE.md](ASSISTANCE.md) for the disclosure and required student verification.
