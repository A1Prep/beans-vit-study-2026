# Execution evidence

**Completed locally on 23 October 2026:** data acquisition and audit, both full-epoch GPU pilots, six training runs, locked test evaluation, synthetic sensitivity checks, and statistical analysis. The full Colab workflow has not been executed.

The backbone was `facebook/dinov2-small` at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. All experiments used the original audited Makerere Beans split: 1,034 train, 133 validation and 128 test images. The hardware was an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM, PyTorch 2.6.0+cu124, Python 3.11 and transformers 4.57.1. Exact installed versions are in `environment-lock.txt`.

| Method | Paired runs | Test accuracy, mean ± sample SD | Test macro-F1, mean ± sample SD | Mean training time |
|---|---:|---:|---:|---:|
| Frozen backbone | 3 | 90.63% ± 2.07 pp | 90.71% ± 2.05 pp | 100.57 seconds |
| Partial fine-tuning | 3 | 97.14% ± 1.19 pp | 97.16% ± 1.18 pp | 101.85 seconds |

Seeds were 17, 42 and 2026. Partial fine-tuning improved macro-F1 in each pair, by 8.52, 6.97 and 3.86 percentage points. For the prespecified reference seed 17, the paired class-stratified 95% bootstrap interval for the difference is +3.15 to +14.10 percentage points. This is an interval for that reference pair, not the three-seed mean. Unknown plant/farm dependence and limited test coverage remain limitations. Timing is training-loop time; the number of completed epochs differs between runs.

`protocol.lock.json` records the settings, source/data hashes and selected checkpoints before final test calls. `runs/{frozen,partial}/seed_{17,42,2026}/` contains configuration, histories, summaries, validation predictions, final test predictions and evaluation metrics. The standalone validator was applied after the original execution and all six runs passed; evaluation-record hashes match the lock. The public notebook additionally invokes it before each test evaluation. See `final_verification.json` for this distinction and independent metric recomputation.

Seven implementation tests and twelve protocol tamper/portability tests passed. The pilots confirmed finite intended gradients, unchanged frozen parameters, matched initial classifier weights and exact save/reload predictions. `analysis_verification.json` records separate synthetic checks of bootstrap mathematics. CUDA attention reported possible nondeterminism, so bitwise-identical retraining is not promised.

`analysis.json`, `comparison.csv`, `run_metrics.csv`, `paired_seed_differences.csv`, `per_class_metrics.csv` and `stress_metrics.csv` are the machine-readable results. `figures/` contains PNG and PDF figures. Audit source hashes and AI-assisted duplicate-screening evidence are under `data_audit/`. Checkpoints and raw data are excluded from Git and can be regenerated using the notebook.

## Overfitting audit

A post-hoc diagnostic evaluated all six unchanged selected checkpoints on the complete training and validation splits with identical clean preprocessing and evaluation mode. Original test scores were reused; no training, checkpoint selection or new test evaluation occurred. All clean validation predictions matched the original saved predictions, and the original protocol, checkpoint and training-source hashes passed validation.

| Method | Mean clean training F1 | Mean validation F1 | Mean test F1 | Training minus test |
|---|---:|---:|---:|---:|
| Frozen | 94.27% | 94.01% | 90.71% | 3.55 pp |
| Partial fine-tuning | 98.59% | 98.51% | 97.16% | 1.42 pp |

These are macro-F1 averages over the three seeds. For partial fine-tuning, individual training–test gaps were -0.09, 2.32 and 2.04 points. A small average gap is reassuring within this split, but does not prove absence of overfitting. Training and validation are reused data, validation guided checkpoint selection, and the test contains only 128 images with unknown farm/plant independence.

The histories show late deterioration. For partial seed 42, validation loss rose from 0.0598 at selected epoch 5 to 0.1348 at epoch 8 while online augmented training loss fell from 0.0913 to 0.0707. All three subsequent validation losses exceeded the selected epoch's loss. This is consistent with overfitting or instability; selecting the earlier checkpoint prevented using the deteriorated final state. Other runs also fluctuated, so it would be incorrect to claim that overfitting never occurred. Loss curves compare online augmented training with clean validation; the new fixed-checkpoint audit supplies a more comparable split evaluation.

See [checkpoint gaps](overfitting/checkpoint_gaps.csv), [history diagnostics](overfitting/history_diagnostics.csv), [all-seed curves](overfitting/all_seed_learning_curves.png), [split comparison](overfitting/checkpoint_generalisation.png) and [provenance](overfitting/audit.json). No hyperparameters were changed in response to these test-aware diagnostics. An independent dataset would be required for a new confirmatory experiment after such changes.

`reproduction/` is reserved for fresh runs and is ignored by Git. The original held-out test has now been evaluated: further choices informed by these results must be described as exploratory, not as a fresh untouched-test experiment.
