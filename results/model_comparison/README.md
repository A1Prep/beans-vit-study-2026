# Completed exploratory model comparison

All 12 planned runs completed on 24 October 2026, using an RTX 3050 Laptop GPU with 4 GB memory. Both official pretrained models were acquired and used. New test inference occurred only after all 12 runs completed and checkpoints were locked. The original test results were already known, so this is not a fresh confirmatory test.

| Model | Adaptation | Accuracy (%) | Macro-F1 (%) | Median epoch (s) | Peak allocated GPU (MiB) |
|---|---|---:|---:|---:|---:|
| DINOv2-S/14 (2023) | frozen | 90.62 ± 2.07 | 90.71 ± 2.05 | 11.94 | 182.9 |
| DINOv2-S/14 (2023) | partial | 96.88 ± 1.56 | 96.90 ± 1.55 | 12.62 | 274.4 |
| SigLIP 2-B/16 (2025) | frozen | 96.88 ± 1.35 | 96.90 ± 1.34 | 12.99 | 459.7 |
| SigLIP 2-B/16 (2025) | partial | 96.88 ± 0.78 | 96.90 ± 0.78 | 16.08 | 688.8 |

The full [analysis](comparison_analysis.json) and [independent review](independent_review.json) support these values. The review independently recomputed 12 clean test evaluations, eight additional synthetic conditions, 24 clean training/validation scores, class metrics, confusion matrices, generalisation gaps and the reference bootstrap. Original source and run artifacts were checked unchanged. AI reviewers are not independent human assessors.

## Evidence map

| File or folder | What it records |
|---|---|
| `model_selection.json`, `siglip2_acquisition.json` | Primary research, accessible official model identity and completed verified download |
| `study_plan.json`, `execution_identity.json` | Fixed settings and source/manifest/plan hashes |
| `pretrained_input_verification.json` | Exact local weights, configuration and processor hashes |
| `dinov2/pilot_report.json`, `siglip2/pilot_report.json` | Full-epoch pilots, actual parameter counts, gradients, frozen-weight checks and checkpoint reloads |
| `training_completion.json`, `evaluation_completion.json`, `completion.json` | Executed stages and counts |
| `<model>/protocol.lock.json`, `<model>/runs/` | Selected checkpoint hashes, run settings, histories, validation and test predictions |
| `<model>/overfitting/` | Clean train/validation predictions, selected-checkpoint gaps and all-seed history diagnostics |
| `combined_run_metrics.csv`, `combined_summary.csv` | All 12 scores and four configuration summaries, including raw timing totals/means and robust descriptive summaries |
| `paired_model_differences.csv` | Same-seed newer-model minus DINOv2 differences |
| `combined_per_class_metrics.csv` | Precision, recall, F1 and support for every class and run |
| `figures/` | PNG/PDF comparison, cost, confusion, learning curves, gaps and sample predictions |
| `independent_review.json`, `implementation_tests.txt` | Independent numerical audit and implementation checks |
| `notebook_validation.json` | Static full-workflow validation and actual saved-results reader execution status |

## Interpretation limits

The partially fine-tuned models are practically tied (difference below 0.01 percentage point). SigLIP 2 improves the frozen-feature mean by 6.18 points over frozen DINOv2. Partial adaptation adds no meaningful mean-score gain over frozen SigLIP 2 in these runs; this is a descriptive result, not a fresh test-based model-selection guarantee.

The primary paired reference difference is -0.78 points, 95% bootstrap interval [-3.13, 1.57], for partial seed 17 only. Three-seed mean difference is +0.00 points. These summaries describe different uncertainty sources and must not be conflated.

Model capacity, pretraining collection, pooling and native preprocessing differ. Learning rates were fixed, not individually optimised. The test set is small and reused, with unknown plant/farm dependence. No observed metric establishes external performance, calibrated diagnostic confidence or fairness across farmers. Blur and darkening are limited synthetic checks.

Timing medians/IQRs pool all completed epochs without excluding the unusually long recorded epoch. [The descriptive amendment](timing_analysis_amendment.json) explains why the median was added before test scoring. All raw durations remain available; no cause for the delay was established. Training time excludes acquisition, validation, checkpoint operations and pretraining. Framework-allocated GPU memory is not total device occupancy.

Use [the main README](../../README.md) and [full reproduction notebook](../../model_comparison.ipynb) to rerun the experiment. Actual runs were local; full Colab execution remains untested. Raw weights/checkpoints/images are omitted from Git and ZIP.
