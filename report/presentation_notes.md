# Presentation notes: SigLIP 2 versus DINOv2

Presentation: 27 October 2026, 6:30 pm. These notes describe the fresh 12-run comparison executed on 24 October, not the earlier six-run DINOv2 study. They accompany the report and are outside its word count. Review the code and explanations yourself and disclose AI assistance according to course requirements.

## Five-minute explanation

1. **Question.** Does the recent SigLIP 2 vision-language encoder improve bean-leaf classification over DINOv2, and what does adaptation cost? Both frozen features and partial fine-tuning were tested, with three matched seeds per configuration. The main comparison is partial SigLIP 2 versus partial DINOv2.
2. **Dataset.** Makerere Beans has 1,295 images: 1,034 training, 133 validation and 128 test. Expert source labels are angular leaf spot, bean rust and healthy. Images came from Ugandan fields. There were no corrupt images or exact duplicates; AI-assisted inspection judged 14 perceptual-hash candidate pairs distinct. Missing plant/farm identities and unknown pretraining overlap remain limitations.
3. **Recent model.** SigLIP 2 was published in 2025. At B/16–224 scale, its paper reports 78.2% ImageNet zero-shot accuracy against original SigLIP's 76.2%, leading the same-size baselines in that table. Google's public weights were downloaded and the image tower used. SigLIP's attention-pooled 768-value representation differs from DINOv2's 384-value class token. Each received a new three-class head. Partial adaptation updated the final two blocks and final normalisation; SigLIP's pretrained pooling remained frozen. All four configurations ran on a 4 GB GPU.
4. **Main result.** Show `results/model_comparison/figures/performance_cost.png`. Mean macro-F1 was 90.71% for frozen DINOv2 and 96.90% for partial DINOv2, frozen SigLIP and partial SigLIP. Partial models differed by less than 0.01 percentage points. SigLIP supplied stronger frozen features, but its partial fine-tuning produced no meaningful average improvement under these settings. It would be misleading to say the newer model won.
5. **Cost and uncertainty.** Partial DINOv2/SigLIP median observed epoch times were 12.62/16.08 seconds, with peak allocated memory 288/722 MB. Raw means are retained because some epochs were unusually long for unknown reasons. Partial SigLIP-minus-DINO differences were −0.78, +1.55 and −0.77 points. The seed-17 paired bootstrap interval was [−3.13, +1.57] points, including zero. It concerns that pair, not the three-seed mean. The reused test set makes this exploratory.
6. **Failure analysis.** Show `partial_confusion.png` and `siglip_sample_predictions.png`. Seed-17 partial SigLIP called three leaf-spot images rust; one wrong prediction had 98.65% confidence. Partial DINOv2 made two errors. Neither labelled a diseased image healthy, but the small sample cannot establish safety. Partial DINOv2 offers comparable mean accuracy with a smaller model and memory requirement. Independent farm-level validation is necessary.

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

**Was the entire workflow tested in Colab?** The experiments ran locally on an RTX 3050 Laptop GPU. A portable notebook and separate results-review notebook are supplied. A full Colab rerun has not been completed, and bitwise identical CUDA results are not promised.

**How was AI used?** Codex assisted research, implementation, duplicate-image screening, analysis, writing and review. The experiments were executed and prediction files are available for checking. Student understanding and final review remain necessary. Do not describe assisted work as independently completed without assistance.

## Before presenting

- Distinguish mean accuracy, reference-seed results, seed SD and the reference bootstrap.
- Locate the study plan, checkpoint locks, timing amendment, histories and predictions.
- Use the current `results/model_comparison` figures, avoiding earlier DINOv2-only results.
- Verify course-specific duration, submission access and AI-disclosure requirements.
