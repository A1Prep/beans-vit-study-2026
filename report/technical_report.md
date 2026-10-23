# Bean-Leaf Disease Classification with a Pretrained Vision Transformer: Frozen Features versus Partial Fine-Tuning

Experiments executed: 23 October 2026.

## 1. Introduction

Image-based recognition could support agricultural assessment, but a classifier must be evaluated beyond its overall accuracy. The practical question here is whether a small labelled dataset benefits from adapting an existing vision model. Vision transformers represent an image as a sequence of embedded patches and use attention to exchange information across them [1]. Pretraining enables these representations to be reused for a new classification task.

This study asks: on the published Beans split, does updating the final layers of a pretrained transformer improve macro-F1 over a frozen backbone, how reliable is the difference, and what additional training cost does it require? The contribution is a controlled comparison of two adaptation strategies, with reproducible evaluation and analysis of failure consequences. The experiment evaluates recognition of three dataset categories; it does not establish a field-ready diagnostic system.

## 2. Dataset and audit

Beans contains smartphone photographs collected in Ugandan fields by Makerere AI Lab with the National Crops Resources Research Institute. Agricultural experts assigned three labels: angular leaf spot, bean rust and healthy [3]. The small labelled sample motivates transfer learning, while field backgrounds provide a more relevant setting than isolated leaf cutouts.

The pinned Hugging Face release contains 1,034 training, 133 validation and 128 test images [4]. The original repository reports 1,296 images; the distributed splits actually loaded here total 1,295. Table 1 reports the retained class counts. All images were 500 × 500 RGB, with no corrupt files. File and decoded-pixel hashes found no exact duplicates. A 64-bit perceptual-hash screen at Hamming distance ≤12 produced 14 candidate pairs, including seven across splits. Recorded AI-assisted visual duplicate screening judged them distinct; no images were excluded. Test images were accessed for this integrity audit before modelling, but not for training or checkpoint selection. This screening was not independent expert verification of disease labels.

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| Angular leaf spot | 345 | 44 | 43 |
| Bean rust | 348 | 45 | 43 |
| Healthy | 341 | 44 | 42 |
| Total | 1,034 | 133 | 128 |

Table 1. Audited split counts; no exclusions were required.

The original split was preserved. The manifest, audit decisions and revisions accompany the code. Perceptual hashing cannot exclude every related photograph. No plant or farm identifiers were available, so independence at those levels and geographic generalisation remain unverified. Potential overlap with the backbone's pretraining collection was not measured.

## 3. Pre-processing

Images were converted to RGB and resized in full to 224 × 224 using bicubic interpolation. Their square aspect ratio was preserved. Values were divided by 255 and normalised using checkpoint channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225) [6]. The official processor resizes to 256 and centre-crops to 224; the full-image resize here deliberately retains peripheral image regions while reducing fine detail.

Training applied horizontal and vertical flips, each with probability 0.5, followed by rotation sampled uniformly from −15° to 15°, with mean-colour border fill. These bounded changes vary orientation while avoiding deliberate colour alteration or aggressive cropping of disease evidence. Validation and test preparation was deterministic. Augmentation and sample order were deterministic functions of seed, epoch and image identifier, giving both training arms matched inputs. Frozen features were computed from the augmented images during training, avoiding a comparison against an unaugmented cache.

## 4. Model and training

The backbone was facebook/dinov2-small, a ViT-S/14 pretrained through DINOv2 self-supervision and distillation on the curated LVD-142M collection [2,5]. Its configuration has 12 transformer blocks, 384-dimensional embeddings and six attention heads. At 224 × 224, 14 × 14 patches produce 256 patch tokens plus a class token [5,6]. The final 384-dimensional class token feeds a new linear three-class classifier. The exact model and dataset revisions are recorded in the accompanying metadata.

DINOv2 offers documented transferable representations and a manageable model size. It is an established vision foundation model, not the newest 2026 architecture or a claimed state-of-the-art Beans result. Accessible weights made this a practical choice after the preferred DINOv3 checkpoint required gated access. Meta recommends starting with frozen features [5], motivating a test of whether adaptation benefits this small dataset.

Arm A trained only the classifier. Arm B additionally trained blocks 10–11 and the final normalisation layer, leaving earlier layers fixed. Both used the same backbone weights, paired classifier initialisation, data, augmentation and head learning rate. The backbone stayed in evaluation mode in both arms to disable stochastic layers; gradient tracking remained enabled for trainable parameters in Arm B. Trainable counts were 1,155 and 3,552,387 (22,057,731 total in each model).

Cross-entropy was optimised with AdamW: head learning rate 0.001, backbone rate 0.00001, weight decay 0.01, batch size 8, gradient clipping at norm 1, and mixed FP16 precision. Learning rates were constant. Training allowed 10 epochs with patience 3 on strict validation macro-F1 increases. Checkpoint ties used lower validation loss, then earlier epoch. Three paired seeds (17, 42 and 2026) completed both arms; no hyperparameter search was performed. Settings were fixed before final test evaluation. A full-epoch pilot verified finite gradients, unchanged frozen weights, updates to intended parameters and matching pilot outputs after checkpoint reload. Seven implementation tests passed. Runtime was measured on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), using PyTorch 2.6.0 with CUDA 12.4. Configurations, histories, revisions and per-image predictions support reproduction. Random seeds and data ordering were controlled, but CUDA attention issued a nondeterminism warning; bitwise reproduction is not guaranteed. Codex assisted with implementation, analysis, drafting and review; the reported experiments were executed.

## 5. Evaluation

The primary metric was macro-F1, the unweighted mean of the three class F1 scores, giving each class equal influence. Accuracy, class-specific precision, recall, F1 and support, and confusion matrices provide complementary views. A majority-class classifier, selected from training labels only, supplies a simple baseline. Both arms were evaluated on the same audited test images after all model-selection decisions were fixed.

For repeated training, the mean and standard deviation across paired seeds describe optimisation variability. This is distinct from uncertainty caused by the small test sample. For the prespecified reference seed 17, a paired class-stratified bootstrap used 2,000 resamples (random seed 20261020) to estimate a 95% percentile interval for the partial-minus-frozen macro-F1 difference. Each resample used the same image indices for both arms. This interval concerns that reference pair, not the three-seed mean. The interval is conditional on the observed class composition and an independence assumption; it cannot account for unknown plant or farm dependence. A difference was interpreted alongside its size, uncertainty and training cost rather than accuracy alone.

## 6. Results and interpretation

Table 2 shows improvements for partial fine-tuning under the fixed configurations. The mean macro-F1 gain was 6.45 percentage points; both arms substantially exceeded the training-majority baseline, which always predicted bean rust.

| Approach | Accuracy (%) | Macro-F1 (%) | Mean training time (s) |
|---|---:|---:|---:|
| Majority baseline | 33.59 | 16.76 | — |
| Frozen features | 90.63 ± 2.07 | 90.71 ± 2.05 | 100.6 |
| Partial fine-tuning | 97.14 ± 1.19 | 97.16 ± 1.18 | 101.8 |

Table 2. Means ± sample standard deviations across three seeds; the deterministic baseline has one result. Training time excludes validation, loading and checkpoint overhead.

| Paired seed | Frozen F1 (%) | Partial F1 (%) | Difference (points) |
|---|---:|---:|---:|
| 17 | 89.93 | 98.45 | +8.52 |
| 42 | 89.16 | 96.14 | +6.97 |
| 2026 | 93.04 | 96.90 | +3.86 |

Table 3. Every paired seed favoured partial fine-tuning. The reference difference was +8.52 points, with bootstrap interval [3.15, 14.10]. This supports a positive difference for that pair under the stated sampling assumptions; it does not establish superiority on other datasets.

Partial adaptation reached high validation F1 sooner (Figure 1): the reference checkpoints were epochs 3 and 9 for partial and frozen training. Partial validation F1 tied at epochs 2 and 3; lower loss selected epoch 3. Partial training loss fell from 0.351 to 0.084, while validation loss fluctuated from 0.063 at epoch 3 to 0.286 at epoch 4 and 0.099 at epoch 5. This indicates validation instability rather than a sustained overfitting trend. Augmented online training loss and deterministic validation loss are not directly equivalent evaluations.

<!--FIGURE:learning_curves-->

Figure 1. Training/validation loss and validation macro-F1 for reference seed 17. Selection used validation macro-F1, then lower loss.

The reference partial model made two errors: one angular leaf spot and one healthy image were classified as bean rust (Figure 2). Its class F1 scores were 98.82%, 97.73% and 98.80%, respectively; rust precision was 95.56% despite 100% recall. Frozen class F1 scores were 89.66%, 87.64% and 92.50%, making rust the weakest class in both. Full precision, recall and support tables accompany the code. Figure 3 includes a wrong rust prediction with a 99.5% softmax score, illustrating why confidence alone cannot justify a diagnosis.

<!--FIGURE:confusion_matrices-->

Figure 2. Test confusion matrices for seed 17: rows are true classes, columns predicted classes; entries are counts.

<!--FIGURE:sample_predictions-->

Figure 3. Seed-17 partial predictions: up to four highest-confidence errors, then lowest-confidence correct cases to complete six examples. Scores are uncalibrated.

Similar total training times reflect different stopping epochs. Frozen runs completed 25 epochs altogether, versus 23 for partial training. Pooled training time per epoch increased from 12.07 to 13.28 seconds, despite the much larger trainable parameter count. Thus similar elapsed training cost does not imply equal computational work per epoch.

Supplementary synthetic checks applied Gaussian blur (radius 1 at 224 pixels) or brightness ×0.7 to the same reference test images. Partial macro-F1 was 97.67% under blur and 98.45% under darkening, versus 98.45% clean; frozen scores rose from 89.93% to 92.25% under either change. These non-monotonic responses at one severity do not establish field robustness.

Only 128 test images were available; one error changes accuracy by 0.78 points. Three seeds, fixed hyperparameters, uncertain image independence and no external validation limit the conclusion. The comparison concerns these configurations, not each strategy's best achievable performance.

## 7. Ethical and societal considerations

The potential benefit is more accessible assistance in recognising visible leaf symptoms, but mistakes have unequal consequences. The partial reference run labelled 0/86 diseased leaves healthy and 1/42 healthy leaves diseased, compared with 1/86 and 5/42 for frozen features. Zero observed missed cases does not establish zero risk. Missed disease could delay expert assessment; false alarms could cause unnecessary expense or inappropriate intervention. These are plausible consequences, not outcomes measured by this experiment. Any future tool should support agricultural experts, communicate limitations and avoid automatically prescribing treatment.

The dataset covers one crop and two disease categories within a restricted collection. Other diseases, pests or nutrient problems would still be forced into one of the three classes. A high softmax score is therefore not proof of correctness or a calibrated diagnostic probability. Before practical use, the system would need prospective data covering other regions, seasons, devices and disease presentations, alongside an independently validated rejection or referral policy. Earlier plant-disease research reported large performance losses on images acquired differently from its training data [7]; that finding motivates external validation but does not predict the size of any Beans-specific decline.

Balanced class scores cannot establish fairness across farmers. No suitable demographic or farm identifiers were available for those tests. Meta also reports socioeconomic and geographic biases in DINOv2 on separate benchmarks [5]; this is a reason to investigate representation, not evidence that a particular Beans error was caused by those biases. Unequal access to suitable phones, connectivity and agronomic support should inform any later deployment design.

Attribution must recognise Makerere AI Lab, the agricultural annotators and the model authors. Reusing pretrained weights and limiting adaptation avoids reproducing the full pretraining workload, although the present runtime does not measure the model's entire environmental footprint. The report therefore gives measured experiment cost without inventing carbon savings. Public availability of data also does not itself establish blanket consent for unrelated future uses.

## 8. Conclusion

Partial adaptation improved mean test macro-F1 from 90.71% to 97.16%, winning all three paired comparisons. It is the preferred configuration within this experiment, with similar total training time but higher per-epoch cost. The result supports adapting pretrained representations for this dataset; it does not establish deployable diagnosis. Next steps are validation on independently collected farms and a validation-only comparison of additional fine-tuning depths, with calibrated uncertainty and expert referral evaluated before practical use.

## References

[1] Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. [Source](https://arxiv.org/abs/2010.11929).

[2] Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193. [Source](https://arxiv.org/abs/2304.07193).

[3] Makerere AI Lab (2020). Bean disease dataset (ibean). [Source](https://github.com/AI-Lab-Makerere/ibean).

[4] Makerere AI Lab (n.d.). Beans dataset card and release metadata. Hugging Face. Accessed 23 October 2026. [Source](https://huggingface.co/datasets/AI-Lab-Makerere/beans/blob/main/README.md).

[5] Meta AI (n.d.). DINOv2-S/B/L/g model card. Accessed 23 October 2026. [Source](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).

[6] Meta AI / Hugging Face (n.d.). DINOv2-Small checkpoint and image-processor configurations. Accessed 23 October 2026. [Source](https://huggingface.co/facebook/dinov2-small/blob/main/config.json). [Image processor](https://huggingface.co/facebook/dinov2-small/blob/main/preprocessor_config.json).

[7] Mohanty, S. P., Hughes, D. P., and Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, 1419. [Source](https://doi.org/10.3389/fpls.2016.01419).
