# Five-minute presentation: SigLIP 2 versus DINOv2

Presentation: 27 October 2026, 6:30 pm. Use slides 1–7 for the five-minute talk. Slides 8–12 are backup material for questions. The main results match the original 12-run experiment in the report; slide 11 explains the separate notebook verification.

[Editable PowerPoint](presentation/bean_leaf_presentation.pptx)

The timings below total five minutes and are rehearsal targets, not recorded speaking times. Speak in your own words. Speaker notes and source citations are also embedded in the PowerPoint.

## Main talk

### Slide 1: Bean-leaf disease classification (0:00–0:15)

This project asks whether a newer vision model improves bean-leaf disease classification. I compared SigLIP 2 with DINOv2, looking at accuracy, the benefit of fine-tuning, and the limits of the results.

### Slide 2: Three classes in field photographs (0:15–1:00)

The task has three labels: angular leaf spot, bean rust and healthy. These are real smartphone photographs from Ugandan fields, so the backgrounds, hands and lighting vary. That makes the task useful for testing transfer learning with relatively little labelled data. The audited release contains 1,295 images. I kept the published training, validation and test split. There were no corrupt files or exact duplicates. However, there are no plant or farm identifiers, so the split cannot guarantee that every test image comes from an independent farm.

### Slide 3: Two encoders, two adaptation settings (1:00–1:50)

Both models start from public pretrained weights. SigLIP 2 is a 2025 vision-language model, but this experiment uses only its image encoder. DINOv2 is the smaller 2023 comparator. Each gets a new three-class classifier. In the frozen setting, only that classifier learns. In the partial setting, the final two transformer blocks and final normalisation also learn. Both models use 224-pixel images, model-specific normalisation, and matching flips and rotations. I ran three seeds for each setting, giving twelve runs, with early stopping based on validation macro-F1. The settings compare these systems, rather than each model’s best possible performance.

### Slide 4: SigLIP 2 helped most with frozen features (1:50–2:40)

Macro-F1 gives each class equal weight. The frozen results are where the models differ most: DINOv2 reaches 90.71 percent, while SigLIP 2 reaches 96.90 percent, a gain of 6.18 percentage points. After partial fine-tuning, both average 96.90 percent. That is a practical tie in these runs, not proof that the models are equivalent. DINOv2 also has a smaller memory footprint. The test set has only 128 images, and earlier DINOv2 test results were already known, so this is an exploratory comparison. The reference paired confidence interval includes zero. The backup table contains the seed variation and timing details.

### Slide 5: Later epochs showed overfitting or instability (2:40–3:25)

I also checked overfitting. This example is the partial SigLIP run with seed seventeen. Training loss generally falls, but validation loss jumps near the end. The selected checkpoint is epoch six, before that deterioration, because validation macro-F1 determines selection. Across selected checkpoints, the average clean training-to-test F1 gap is 1.27 points for DINOv2 and 2.01 for SigLIP. Those small gaps are useful diagnostics, but they do not prove an absence of overfitting. Validation already influenced checkpoint selection, and the project has no independent farm-level evaluation. The full results retain every seed’s learning history.

### Slide 6: A confident prediction can still be wrong (3:25–4:20)

These two images show why a high average score is not enough. Both labels are angular leaf spot, but SigLIP predicts rust. One wrong prediction has 98.65 percent confidence. Confusing disease types could lead to an unsuitable intervention. Missing disease could delay advice, while false alarms could add unnecessary cost. The dataset also covers only three labels in one collection, with no evidence of equal performance across farms, devices or regions. Other damage still gets forced into those labels. A future application would need independent field testing, a validated referral policy and agricultural expert review. It should not prescribe treatment automatically.

### Slide 7: What the comparison supports (4:20–5:00)

The main finding is that SigLIP 2 supplies stronger frozen features. Partial fine-tuning brings DINOv2 to essentially the same average score in the original study, using a smaller model and less allocated memory. This supports a useful comparison, but not a deployment claim. The next step would be validation on independently collected farms, followed by tuning that uses only validation data. Codex assisted the research, code, analysis and drafting. The report, executed notebook and saved predictions document the actual experiments and their limitations.

## Backup slides

- Slide 8: results and uncertainty.
- Slide 9: where the reference models failed.
- Slide 10: preprocessing and training settings.
- Slide 11: reproducibility and remaining limits.
- Slide 12: sources.

## Results to have ready

| Configuration | Accuracy (%) | Macro-F1 (%) | Mean raw training time (s) |
|---|---:|---:|---:|
| DINOv2 frozen | 90.62 ± 2.07 | 90.71 ± 2.05 | 102.8 |
| DINOv2 partial | 96.88 ± 1.56 | 96.90 ± 1.55 | 96.2 |
| SigLIP 2 frozen | 96.88 ± 1.35 | 96.90 ± 1.34 | 387.5 |
| SigLIP 2 partial | 96.88 ± 0.78 | 96.90 ± 0.78 | 195.9 |

Values are means ± sample SD across three seeds. Training time excludes validation/loading/checkpoint overhead. Every observed epoch is retained. Lower seed SD does not establish population-level reliability. The partial SigLIP mean is 0.00110 percentage points above partial DINOv2: practically tied, not mathematically identical.

## Likely questions

**Does SigLIP 2 meet the advanced-model requirement?** It is a recent 2025 vision-language family with leading results among the same-size baselines in its published comparison. This project actually acquired and trained its image tower. That is a defensible contemporary-model rationale, not a claim that it is the latest model in October 2026 or that these Beans results are state of the art.

**Why did the newer model not clearly win?** Stronger frozen representations do not guarantee greater benefits from updating the last two blocks. The dataset is small, and learning rates/adaptation depth were not optimised separately. Model size, pretraining, pooling and preprocessing differ, so the experiment cannot identify one cause.

**Which model would you choose?** Within this resource-limited experiment, partial DINOv2 achieved essentially the same accuracy with fewer total parameters and less allocated memory. This is an exploratory conclusion, not a deployment decision. Frozen SigLIP is also useful: it reached similar performance while updating only 2,307 classifier parameters, although its full image tower still requires computation and memory.

**Did you train from scratch or use text prompts?** Neither. Pretrained image towers were reused with supervised classifiers. The SigLIP text tower and zero-shot prompts were not used. Frozen versus partial describes which image-model parameters were updated.

**Why not use the first SigLIP token?** It is an image patch, not a class token. The pretrained attention-pooling head supplies the intended image representation. DINOv2 has a class token, so its readout differs.

**Was this a fair comparison?** The same audited split, seeds, augmentation decisions, head learning rate, stopping rule and final-two-block adaptation policy were used. Head initialisation was paired within each backbone. Preprocessing and model dimensions differ. This compares concrete systems under fixed settings, not each model's best achievable accuracy or a single causal factor.

**Was the test set untouched?** No. Original DINOv2 results were already known, and test images had been viewed for integrity screening. The new plan was fixed before SigLIP test inference; all 12 training runs completed and checkpoints were locked before new evaluation. This preserves the follow-up procedure without undoing prior exposure. The study is explicitly exploratory.

**Did the models overfit?** Partial DINOv2 clean train/validation/test macro-F1 averaged 98.17/98.51/96.90%, versus SigLIP 98.91/98.26/96.90%. Training–test gaps were 1.27 and 2.01 points. Some later epochs deteriorated: SigLIP seed 17 validation loss rose from 0.0618 at selected epoch 6 to 0.3746 at epoch 9 while training loss fell; seed 42 rose from 0.0183 at epoch 4 to 0.2101 at epoch 7. Earlier checkpoints were retained. This is consistent with overfitting or instability. Small selected-checkpoint gaps cannot prove its absence, because validation informed selection and there is no independent farm-level evaluation. No settings were tuned in response to this audit.

**Why does frozen SigLIP score higher on test than training?** Its clean means were 94.42% training, 95.75% validation and 96.90% test. Different small samples have different difficulty. A negative training–test gap does not prove external generalisation or absence of leakage; test has only 128 images.

**Why are raw timing means different from medians?** One frozen SigLIP epoch took 809.10 seconds instead of roughly 13 seconds; a partial SigLIP epoch took 184.36 seconds. Causes are unknown. A documented amendment before new test evaluation added medians/IQRs while retaining every raw total, mean and range. Slow observations were not deleted or attributed to power/sleep events. Different stopping epochs also affect totals. These are measured durations, not controlled hardware benchmarks.

**What do the error counts mean?** For seed-17 partial models, both had 0/86 diseased images labelled healthy. SigLIP had 0/42 healthy images labelled diseased; DINOv2 had 1/42. SigLIP still confused three leaf-spot images with rust. Wrong disease subtype matters even when disease-versus-healthy counts look favourable. These counts cannot establish safety or justify treatment prescriptions.

**Did synthetic tests prove reliability?** No. Reference partial SigLIP macro-F1 was 97.67% clean, 96.07% under blur and 97.67% under darkening. DINOv2 scored 98.45%, 97.67% and 98.45%. These are one-severity changes to the same images, not independent farms, devices or seasons.

**What is macro-F1, and why a bootstrap?** Precision measures correctness among a class's predictions; recall measures the fraction of its true examples found. F1 is `2PR/(P+R)`; macro-F1 averages the classes equally. Seeds expose training variation. The paired bootstrap resamples matching test images within classes for the seed-17 partial-model comparison. It does not estimate the three-seed mean's uncertainty or correct unknown farm dependence.

**What are the main ethical limits?** Unfamiliar diseases or other damage are forced into three labels; confidence is uncalibrated. False reassurance or unnecessary intervention could harm farmers. Balanced class results do not establish fairness across regions, farmers or devices; published SigLIP fairness tests concern other benchmarks. Future use needs independent sampling, expert oversight, referral and appropriate treatment of farm-identifying data.

**Was the entire workflow tested in Colab?** The unchanged full notebook completed all 10 code cells and 12 fresh runs locally on an RTX 3050 Laptop GPU, with zero errors. Its separate partial-model means were 97.67% for DINOv2 and 96.90% for SigLIP 2. These verification outputs do not replace the original report comparison. Hosted Colab and its isolated installation branch remain untested. Colab is optional, and seeded CUDA reruns are not guaranteed to match bit for bit.

**How was AI used?** Codex assisted research, implementation, duplicate-image screening, analysis, writing and review. The experiments were executed and prediction files are available for checking. Student understanding and final review remain necessary. Do not describe assisted work as independently completed without assistance.

## Before presenting

- Distinguish mean accuracy, reference-seed results, seed SD and the reference bootstrap.
- Locate the study plan, checkpoint locks, timing amendment, histories and predictions.
- Use the current `results/model_comparison` figures, avoiding earlier DINOv2-only results.
- Verify course-specific duration, submission access and AI-disclosure requirements.
