# Current project review — 24 October 2026

The accessible newer-model comparison has completed end to end. Google's SigLIP 2 (2025) and DINOv2 (2023) each ran frozen and partial configurations with three seeds, followed by evaluation, sample outputs and all-run overfitting checks. The paper-backed recent-model rationale is stronger than the original DINOv2-only study. This does not guarantee the lecturer's grade or establish the newest/best model in October 2026.

| Criterion | Weight | Current evidence |
|---|---:|---|
| Dataset justification and preprocessing | 15% | Report section 2; pinned field-image dataset; class counts, corruption/hash/duplicate audit; section 4 explains native preprocessing and matched augmentation |
| Model selection and implementation | 25% | Report sections 3–4; recent public SigLIP 2, benchmark-based rationale, actual download/hash verification, both genuine fine-tuning and frozen baselines; four successful full-epoch pilots |
| Evaluation and performance analysis | 20% | Report sections 5–6; 12 runs, metrics and class errors, paired bootstrap, cost, all-run overfitting checks, learning curves, confusion matrices and sample predictions |
| Ethical and societal considerations | 20% | Report section 7; error consequences, missing diseases, confidence, geographic/subgroup limits, access, attribution, privacy and human review |
| Structure, clarity and referencing | 20% | Updated report below 2,500 words with seven primary references, actual experimental evidence and transparent limitations |

Partial SigLIP 2 mean test macro-F1 was 96.90% versus 96.90% for DINOv2. Clean training–test gaps were 2.01 and 1.27 points respectively. Late validation deterioration is retained and discussed; checkpoint selection used validation only. These diagnostics do not prove no overfitting.

A separate AI reviewer independently recomputed predictions and intervals rather than treating a language-model rating as scientific evidence. The [verification record](results/model_comparison/independent_review.json) passed. Original DINOv2 source/results remain unchanged and its report is archived. The enlarged study explicitly discloses previously known test results.

## Issues encountered and resolved or disclosed

- DINOv3 required gated access. SigLIP 2 official public weights were downloaded without authentication and the full publisher checksum matched.
- The earlier system Python installation was unsuitable; an isolated CUDA environment on D: executed the experiments. Its versions are recorded.
- A model-download interruption and a transient local process/session failure were followed by a successful verified download and fresh successful pilots.
- SigLIP 2 recorded unusually long frozen and partial epochs of 809.10 and 184.36 seconds. Their causes are unknown. Raw times were kept and median/IQR summaries added before new test inference.
- CUDA attention can be nondeterministic; seeded reproduction does not promise bitwise-identical results.
- The complete comparison notebook now passed an unchanged local execution: 10 code cells, 12 fresh training runs, 20 test evaluations, 24 clean training/validation evaluations, analysis, figures and export, with zero errors. The [execution record](validation/full_notebook_local/execution_verification.json) binds it to the supplied notebook. Its new results are separate from the report's original comparison.
- A live-kernel package replacement problem was fixed by isolating Colab dependencies. An embedded verified source bundle removes private-repository access and a second ZIP upload. Six bootstrap checks and a Windows virtualenv smoke test passed alongside the 35 experiment/protocol/analysis tests.
- Hosted Colab remains untested because the available browser session was signed out. Colab is optional, not a rubric requirement. The local complete notebook run does not verify Colab's Linux installation or browser download branches. See the [run guide](COLAB_GUIDE.md).

## Final deliverables

The [technical report](report/technical_report.pdf) remains within 2,500 words, including figure text, and distinguishes the original experiment from the successful later notebook verification. The [five-minute PowerPoint](report/presentation/bean_leaf_presentation.pptx) has seven main slides and seven backup slides, editable charts/tables, actual dataset photographs and embedded speaker notes. Backup material includes the majority baseline, per-class precision/recall/F1 and synthetic blur/darkening checks. The [timed script](report/presentation_notes.md) covers the result, overfitting, errors and responsible use, with additional question preparation. Code, executed notebooks, visualizations and sample predictions are supplied through the repository and ZIP. No Moodle submission has been made.

## Remaining student actions

Review and understand the final report/code, check course AI-assistance requirements, practise the presentation, and provide assessor access to the private GitHub repository or submit the ZIP via the permitted route. Nothing was submitted to Moodle. These student and submission steps are distinct from the completed technical experiment.
