# Complete local notebook verification

On 24 October 2026, the unchanged `model_comparison.ipynb` was executed from
its first to last code cell on the local RTX 3050 Laptop GPU. All 10 code cells
completed with zero errors, including fresh dataset/model acquisition, four
full-epoch pilots, 12 new training runs, 12 clean test evaluations, eight
synthetic test evaluations, 24 clean training/validation evaluations, analysis,
figures and ZIP export.

- [Executed notebook](../../model_comparison_executed_local.ipynb)
- [Execution verification and hashes](execution_verification.json)
- [New reproduction evidence](evidence/)
- [Original report](../../report/technical_report.pdf)

The executed notebook's cell sources exactly match the clean template. Its
downloaded DINOv2 and SigLIP 2 weight hashes match the previously verified official
weights. Export CRC and every exported file were checked. Checkpoints and raw
datasets remain local and are excluded from this lightweight delivery.

These are additional reproduction measurements. They do not replace the original
comparison in `results/model_comparison/` or the report. Seeds do not guarantee
bitwise-identical CUDA results. Long session delays occurred during execution;
all raw epoch times remain in the histories and summaries. This was a functional
notebook test, not a controlled timing benchmark.

For clarity, the new three-seed partial-fine-tuning mean macro-F1 was 97.67% for
DINOv2 and 96.90% for SigLIP 2. The original report's corresponding means remain
96.90% and 96.90%. The additional run verifies reproducibility of the workflow;
it does not establish a reliable model advantage on this small test set.

The notebook reused the local pinned Python 3.11/CUDA environment. Hosted Google
Colab and its separate isolated-environment installation/browser-download branches
were not executed. The bootstrap checks and Windows virtualenv smoke test are
documented separately. Colab is not an assignment requirement.

Codex ran this verification and prepared its evidence with AI-reviewer assistance.
The student's review, understanding and submission remain separate from technical
execution. No automated check guarantees a mark.
