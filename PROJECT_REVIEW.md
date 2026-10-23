# Completed proof of concept

The project was built and executed on 23 October 2026. It is feasible on the available RTX 3050 laptop GPU (4 GB). The six training runs took about 10.1 minutes of measured training-loop time in total, excluding setup, validation and checkpoint operations. The result supports continuing with this topic: partial fine-tuning improved mean test macro-F1 from 90.71% to 97.16%, with a positive difference in all three paired seeds.

Start with the [technical report](report/technical_report.pdf), [saved-results notebook](results_review.ipynb), and [presentation notes](report/presentation_notes.md). The [reproduction notebook](beans_vit.ipynb) downloads the actual data and runs the complete experiment. Full Colab execution has not been tested; the Python workflow was tested locally.

## Coverage against the rubric

| Criterion | Weight | Evidence |
|---|---:|---|
| Dataset justification and preprocessing | 15% | Report sections 2–3; pinned source archives; class counts; image/hash/duplicate audit; model-specific normalization and matched augmentation |
| Model selection and implementation | 25% | Report section 4; pretrained DINOv2-Small; frozen and genuine partial fine-tuning; parameter/gradient checks; checkpoint round trips |
| Evaluation and performance analysis | 20% | Six runs; macro-F1/accuracy/per-class metrics; paired bootstrap; learning curves, errors, runtime and synthetic sensitivity checks |
| Ethical and societal considerations | 20% | Report section 7; missed disease and false alarms; limited geographic and disease coverage; uncertain independence, fairness limits, access, attribution and human review |
| Report structure, clarity and referencing | 20% | Six-page PDF and editable source; seven primary references; all reported values linked to saved experiment evidence |

Separate AI reviewers checked methods, feasibility and rubric coverage. A final reviewer using a different model independently recomputed all ten evaluation outputs and the bootstrap interval and found no material discrepancy. This is review evidence, not a guarantee of marks. DINOv2 is an established 2023 foundation ViT; the lecturer's interpretation of “state of the art” remains an assessment judgment.

## Issues actually encountered

- DINOv3 weights returned an access error because they are gated. The accessible DINOv2-Small fallback was used consistently in both arms.
- The existing Python installation was CPU-only and had a broken SciPy import. An isolated CUDA environment resolved these problems.
- Installing the GPU environment on C: ran out of disk space. The working environment and trained checkpoints were placed on D:.
- CUDA attention reported possible nondeterminism. Seeds and sample ordering are recorded, but bitwise-identical retraining is not promised.
- Automatic approval review blocked deletion of the failed temporary environment in the task's `work/venv` folder, without a more specific reason. That incomplete environment remains outside the deliverable and is not used.

## Refinement and submission

Read the report and code, verify the source references and assistance disclosure, and practise explaining the controlled comparison and its limits. The current test set has been evaluated; later changes informed by its scores must be described as exploratory. A fresh claim of generalisation would need independent evaluation data.

The GitHub repository is private for review. It must be made accessible to the assessor or the project ZIP uploaded through the permitted submission route before the deadline. Nothing has been submitted to Moodle. See [ASSISTANCE.md](ASSISTANCE.md) for the actual use of AI in developing this draft.
