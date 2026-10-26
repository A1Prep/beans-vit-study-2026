# Bean-leaf classification: SigLIP 2 versus DINOv2

Completed locally on 24 October 2026: 12 fresh training runs, both frozen-feature and partial fine-tuning configurations, three paired seeds, test evaluation, sample predictions and all-run overfitting diagnostics. The selected recent model is Google's publicly accessible **SigLIP 2 B/16–224**, compared against DINOv2-Small. Its official weights were downloaded without authentication and their published SHA-256 verified.

Partial SigLIP 2 achieved **96.90%** mean test macro-F1, compared with **96.90%** for partial DINOv2 (+0.00 percentage points). These are actual executions, not estimated scores. The enlarged study is exploratory because the original DINOv2 test results were known before it began.

The partially fine-tuned models are practically tied (difference below 0.01 percentage point). SigLIP 2 improves the frozen-feature mean by 6.18 points over frozen DINOv2. Partial adaptation adds no meaningful mean-score gain over frozen SigLIP 2 in these runs; this is a descriptive result, not a fresh test-based model-selection guarantee.

## Start here

- [Technical report (PDF)](report/technical_report.pdf) and [editable report](report/technical_report.md), within the 2,500-word limit.
- [Executed comparison review notebook](model_comparison_review.ipynb): reads the saved evidence and figures without retraining.
- [Complete reproduction notebook](model_comparison.ipynb): a self-contained notebook that downloads data and pinned models, runs pilots/training/evaluation/analysis, and exports results. It includes its required source files and needs no private GitHub login or second project ZIP. See the [step-by-step Colab guide](COLAB_GUIDE.md). Hosted Colab execution has **not** been verified.
- [Executed full notebook](model_comparison_executed_local.ipynb): the unchanged reproduction notebook completed locally with all 10 code cells, 12 new training runs, 20 test evaluations, 24 clean training/validation evaluations, analysis and export; zero errors. [Verification evidence](validation/full_notebook_local/README.md) is separate from the original results used in the report.
- [Comparison evidence and interpretation](results/model_comparison/README.md), [independent numerical verification](results/model_comparison/independent_review.json), and [rubric coverage](PROJECT_REVIEW.md).
- [Five-minute presentation (editable PowerPoint)](report/presentation/bean_leaf_presentation.pptx): seven main slides and seven backup slides, with embedded speaker notes and citations. [Timed script and question preparation](report/presentation_notes.md) match the report.
- [AI-assistance disclosure](ASSISTANCE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Measured results

| Model | Adaptation | Accuracy (%) | Macro-F1 (%) | Median epoch (s) | Peak allocated GPU (MiB) |
|---|---|---:|---:|---:|---:|
| DINOv2-S/14 (2023) | frozen | 90.62 ± 2.07 | 90.71 ± 2.05 | 11.94 | 182.9 |
| DINOv2-S/14 (2023) | partial | 96.88 ± 1.56 | 96.90 ± 1.55 | 12.62 | 274.4 |
| SigLIP 2-B/16 (2025) | frozen | 96.88 ± 1.35 | 96.90 ± 1.34 | 12.99 | 459.7 |
| SigLIP 2-B/16 (2025) | partial | 96.88 ± 0.78 | 96.90 ± 0.78 | 16.08 | 688.8 |

Accuracy and macro-F1 are mean ± sample SD across seeds 17, 42 and 2026. Epoch medians pool all completed epochs, so seeds with more epochs receive more weight. GPU values are the maximum framework-allocated memory over three seeds, not whole-device occupancy. Runtime is descriptive of this laptop, not an isolated hardware benchmark. A SigLIP 2 frozen epoch took 809.10 seconds while its other epochs took roughly 13 seconds; a partial-training epoch later took 184.36 seconds. The causes are unknown. [The timing amendment](results/model_comparison/timing_analysis_amendment.json) was recorded before new test scoring. All raw times, means, totals, ranges and IQRs are retained.

The prespecified seed-17 partial comparison was -0.78 points, with descriptive paired bootstrap 95% interval [-3.13, 1.57]. This interval describes that reference pair, not the three-seed mean. The test split contains only 128 images, with unknown plant/farm dependence.

## Model and dataset rationale

[SigLIP 2 (2025)](https://arxiv.org/abs/2502.14786) provides a stronger recent-model rationale than using only established DINOv2 (2023). Its paper reports leading same-size comparisons, including 78.2% ImageNet zero-shot accuracy for B/16–224 versus 76.2% for original SigLIP. Those are published ImageNet results, not this project's Beans results or proof of the best model in October 2026. [Selection research](results/model_comparison/model_selection.json) records primary sources, access checks, the alternative Perception Encoder and the reasons for the selected implementation.

The [Makerere Beans dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans) supplies 1,034 training, 133 validation and 128 test field photographs from Uganda. Its three classes are angular leaf spot, bean rust and healthy. Data revision, archive checksums, class counts, duplicate screening and exclusions are recorded in [the original unchanged audit](results/data_audit/). No corrupt images or exact duplicates were found. AI-assisted screening retained all 14 perceptual-hash candidate pairs; plant/farm independence and pretraining overlap remain unverified.

Each image tower has a new three-class classifier. Frozen arms update only that classifier. Partial arms also update the final two transformer blocks and final normalisation. SigLIP 2 uses its pretrained attention pooler, whose weights stay frozen while gradients pass through it. DINOv2 uses its class-token representation. Native normalisation/interpolation differ, and the models have different capacities and pretraining; the comparison evaluates these concrete systems rather than isolating model age.

## Reproduce and inspect

Python 3.11 is the reference environment. Install a CUDA-enabled PyTorch 2.6 build suitable for the hardware, then `pip install -r requirements.txt`. Use the reproduction notebook from the project directory. It creates a fresh output folder, downloads the pinned dataset/model revisions, and records the regenerated manifest's hash in a separate reproduction plan. It never overwrites the submitted evidence. Recorded package versions and hardware are in every run configuration. Retraining is seeded but CUDA attention can be nondeterministic.

Colab is optional and is not required by the assignment. For an optional Colab run, select a T4 GPU and runtime version 2026.07 (Python 3.12), upload the self-contained notebook and choose Run all. Its pinned experiment packages use an isolated environment, avoiding replacement of libraries already loaded by Colab. Six [bootstrap checks and a Windows environment smoke test](results/model_comparison/notebook_bootstrap_validation.json) passed; these do not constitute a hosted Colab test. The [Colab execution record](results/model_comparison/colab_execution_status.json) states the actual scope. Rebuild the embedded source bundle after changing its canonical files with `python tools/build_comparison_notebooks.py`; the builder preserves the executed reader by default.

The complete notebook was subsequently executed from first to last code cell on the local RTX 3050 Laptop GPU. It downloaded both official model revisions afresh, completed all phases and produced an export whose contents and checksums were verified. The local run reused the pinned project environment; it does not verify the separate Colab installation branch. The new reproduction has its own measurements and recorded long session delays. The report continues to describe the original comparison under `results/model_comparison/`; the notebook verification is under `validation/full_notebook_local/`.

For a command-line reproduction, follow the same preparation and plan-generation cells and invoke `src/run_model_comparison.py --help`. The runner supports `pilot`, `train`, `evaluate`, `analyse` and `all` phases. It checks matching inputs on resume, requires full pilots, finishes all 12 training runs before test scoring, verifies checkpoint locks and refuses to overwrite incomplete runs. Checkpoints and raw image downloads are excluded from Git; they are regenerated by the workflow. Fixed-checkpoint scoring and clean train/validation diagnostics use `src/followup_experiment.py` and `src/audit_model_comparison.py`; combined analysis reads saved predictions with `src/analyse_model_comparison.py`.

Run `python -m unittest discover -s tests -v` for implementation, protocol, metric and analysis checks. Saved results include per-image probabilities, per-class precision/recall/F1, confusion matrices, learning curves, generalisation gaps, blur/darkening sensitivity and sample errors.

## Interpretation and submission

The selected partial checkpoints have mean clean train–test macro-F1 gaps of 1.27 points (DINOv2) and 2.01 points (SigLIP 2). Late validation deterioration is reported alongside selected-checkpoint results; small gaps do not prove absence of overfitting. No external dataset or farm-level holdout was tested, and confidence is uncalibrated. The project is not a deployed agricultural diagnosis system.

The [original six-run DINOv2 study](DINOv2_STUDY.md), its source, predictions and locks are preserved. Its report is [archived separately](report/archive/dinov2_study_2026-10-23/technical_report.pdf). The current report uses only the fresh comparison results for its main model comparison.

The GitHub repository is private. Provide assessor access or use the permitted Moodle/ZIP route before the deadline; no Moodle submission has been made. Review the actual code, references and assistance disclosure and be ready to explain the work. AI reviews are evidence checks, not a guarantee of marks.
