# Run the complete comparison in Google Colab

**Colab is optional. The assignment does not require it.** The project can be
run on a suitable local GPU, as used for the reported experiments.

Use **model_comparison.ipynb** for training and evaluation. The separate
**model_comparison_review.ipynb** only displays the saved study results.

## Start a fresh run

1. Sign in at [Google Colab](https://colab.research.google.com/).
2. Choose **Upload notebook** and select `model_comparison.ipynb` from this project.
3. Open **Runtime → Change runtime type**. Select a **T4 GPU** and runtime version
   **2026.07**, which supplies Python 3.12. Save the settings. If T4 is unavailable,
   wait for GPU access; a CPU run is an explicit, substantially slower alternative.
4. Choose **Runtime → Run all**. Keep the notebook open and read any Colab prompt.
   The notebook includes its required project source files, so it does not need
   access to the private GitHub repository, a second project ZIP, a Hugging Face
   token, or a Google Drive mount.
5. Let installation, dataset checking, both model downloads, four pilots, all
   12 training runs, evaluation and analysis finish. The notebook prints the new
   run ID and output location. Download and retain the results ZIP offered by the
   final code cell.
6. Use **File → Download → Download .ipynb** to save the executed notebook with
   its outputs. Keep both this file and the evidence ZIP.

The embedded bundle contains the files needed to run the experiment. Links in
the notebook to the original report and saved-results reader refer to files in
the complete project download; those documents are supplied separately from the
standalone training notebook.

The pinned experiment packages run in a separate environment. They do not replace
the libraries already loaded by Colab's notebook. Python 3.11 and 3.12 are the
supported versions for this environment; use the runtime selector if the default
Python version is newer. Google's [runtime version documentation](https://research.google.com/colaboratory/runtime-version-faq.html)
lists the available versions and their packages.

## What successful completion looks like

- Both model pilots pass for frozen and partial adaptation.
- Training reports **12 completed runs**: two models × two arms × three seeds.
- Both checkpoint locks are created before new test scoring.
- Analysis displays the four model/configuration summaries, paired comparisons,
  confusion matrices, example predictions and overfitting diagnostics.
- The final study completion record says `completed` with `run_count: 12`.
- No code cell has an error. The final evidence ZIP and executed notebook are saved.

A successful first cell or pilot alone does not establish end-to-end success.
Small differences from the report's scores are possible because GPU operations
are not guaranteed to reproduce bit for bit. New runs have separate folders;
the submitted report still describes the original recorded laptop experiments.

## If a run stops

**Unsupported Python:** select runtime version 2026.07 and rerun from the top.

**No GPU:** check the runtime's hardware selection. Google does not guarantee
GPU availability. Do not buy a plan solely to complete this project without
considering the already executed local results and other available hardware.

**Network interruption during installation or downloading:** rerun the failed
setup/download cell after connectivity recovers. Read the actual error before
changing model revisions or package versions.

**Kernel restart with runtime files still present:** set `RUN_ID` in the resource
cell to the previously printed ID, then run from the top. Matching completed
runs can be reused. The workflow intentionally refuses to overwrite an incomplete
training run. If a training run was interrupted, use a new run ID for a fresh
complete reproduction and retain the partial output for diagnosis.

**Runtime deleted or disconnected permanently:** its temporary files may be
lost. Start a fresh run; a previous runtime's local path is not a backup.
Colab explains its [resource and lifetime limits](https://research.google.com/colaboratory/faq.html).

## Verification status

See [notebook validation](results/model_comparison/notebook_validation.json)
for recorded execution evidence. The [complete notebook executed locally](model_comparison_executed_local.ipynb)
with all 10 code cells, 12 new training runs, all evaluation/analysis stages and
export, with zero errors. Its [verification record](validation/full_notebook_local/execution_verification.json)
also checks the downloaded model hashes and export contents.

Hosted Colab has not been tested: the browser session was signed out. The local
run reused the reference environment and does not verify Colab's separate Linux
environment installation or browser-download branches. Colab remains optional.

## Submission files

The required deliverables are `report/technical_report.pdf`, the project code
(including this notebook) and the figures/sample predictions already included
under `results/model_comparison/`. The current report is below 2,500 words.
Use the private GitHub repository with assessor access, or the assignment's
permitted Moodle/ZIP route. No Moodle submission has been made.

The [rubric review](PROJECT_REVIEW.md) maps all five criteria to concrete evidence.
No automated or AI review can guarantee full marks. Review the report, code and
AI-assistance disclosure yourself and practise explaining the experiment.
