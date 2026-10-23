# Presentation notes

Presentation: 27 October 2026, 6:30 pm. These notes accompany the technical report and are outside its word count. Review the code, figures and explanations yourself before presenting; do not describe AI-assisted work as independently completed by you.

## Five-minute explanation

1. **Question and motivation.** Can adapting a pretrained vision transformer improve bean-leaf classification over using its frozen representations? This is a controlled comparison of two configurations, not a claim to build a deployable plant-disease diagnosis system.
2. **Data.** Makerere's Beans release has 1,295 smartphone images from Uganda: 1,034 training, 133 validation and 128 test. Classes are angular leaf spot, bean rust and healthy. No exact duplicates or corrupt images were found. AI-assisted inspection judged 14 perceptual-hash candidate pairs distinct. Test images were accessed for integrity screening before modelling. There are no farm or plant identities, so related-image leakage cannot be ruled out fully.
3. **Model and comparison.** DINOv2-Small is a pretrained ViT-S/14. A 224-pixel image becomes 256 image-patch tokens plus a class token. The 384-value class representation feeds a three-output linear head. Frozen training updates 1,155 head parameters; partial training updates 3,552,387 parameters in the head, final two transformer blocks and final normalisation. The two arms share starting weights, paired head initialisation and augmented inputs. Three seeds were run.
4. **Results.** Mean test macro-F1 was 90.71% ± 2.05 for frozen and 97.16% ± 1.18 for partial adaptation: a 6.45-percentage-point difference. Partial won all three pairs. For the prespecified seed 17, the difference was 8.52 points with a paired bootstrap interval of [3.15, 14.10]. This interval is for that reference pair, not the three-seed mean. Mean training time was 100.6 versus 101.8 seconds, but stopping epochs differed; partial was more expensive per epoch (13.28 versus 12.07 seconds).
5. **Failure analysis and conclusion.** Show the learning curves, confusion matrices and sample outputs. Partial seed 17 made two errors, both predictions of rust: one angular leaf spot and one healthy image. One incorrect prediction had a softmax score of 0.995. Partial adaptation is preferable within the tested settings, but external validation, broader sampling and expert referral remain necessary before practical use.

## Likely questions

**Why choose this dataset?** It is feasible on limited hardware, contains field images, has an existing split and expert source annotations, and allows task-specific discussion of agricultural consequences. Its small size and narrow coverage are also clear limitations.

**Why DINOv2 rather than the newest model?** The intended DINOv3 weights required gated access. DINOv2 offered accessible weights, documented transferable representations and a manageable size. It is an advanced, established foundation ViT from 2023, not the newest 2026 architecture or a claimed state-of-the-art Beans result.

**Did you train the transformer from scratch?** No. Both models reused pretrained weights. One trained a new classification head; the other also adapted the final two blocks and final normalisation layer.

**Why not train every layer?** Partial adaptation limits memory and optimisation demands on a small dataset. Full fine-tuning was not tested, so the project cannot say it would perform worse.

**What is macro-F1?** Precision is the fraction of a class's predictions that are correct; recall is the fraction of that class's true examples found. F1 is their harmonic mean, `2PR/(P+R)`. Macro-F1 averages the three class F1 values equally. Accuracy is the fraction of all images classified correctly.

**What prevented test-set tuning?** Training used training images, and checkpoint selection used validation macro-F1, breaking ties by lower validation loss. The model configurations, reference seed and final protocol were recorded before test prediction calls. Integrity auditing accessed test images earlier; the report discloses that. Artifact hashes establish consistency, not independent proof of chronology.

**Did the model overfit?** Some late epochs showed deterioration. For partial seed 42, validation loss rose from 0.0598 at selected epoch 5 to 0.1348 at epoch 8 while training loss fell; the saved checkpoint precedes that deterioration. We then evaluated all six unchanged selected checkpoints on clean training and validation images and reused the original test results. Partial mean macro-F1 was 98.59% training, 98.51% validation and 97.16% test, a 1.42-point training–test gap (frozen: 3.55 points). These modest gaps support the selected model within this split but cannot prove no overfitting or generalisation to new farms. Validation was used for model selection. The post-hoc audit did not tune settings.

**Why three seeds and a bootstrap?** Seeds expose variation from initialisation and training. The paired bootstrap resamples the same test indices for both reference models, within each class, to assess uncertainty from that fixed test sample. Neither resolves unknown farm dependence or proves performance elsewhere.

**Can zero missed disease cases establish safety?** No. The partial reference run had 0/86 diseased leaves classified healthy and 1/42 healthy leaves classified diseased. These are small, selected-dataset counts. Other diseases would still be forced into the three available labels, and a high score can be wrong.

**Did the robustness tests prove real-world reliability?** No. Blur at radius 1 and brightness multiplied by 0.7 were limited synthetic checks on the same images. Partial F1 was 97.67% under blur and 98.45% under darkening. Frozen F1 improved under both. One severity cannot establish resilience across farms, phones or seasons.

**Is the result reproducible in Colab?** The actual experiments ran locally on an RTX 3050 Laptop GPU with 4 GB memory. The repository includes the workflow, metadata and outputs, but the complete Colab workflow was not executed. CUDA attention warned of possible nondeterminism, so bitwise identical reruns are not promised.

**How was AI used?** Codex assisted with implementation, source research, duplicate-image screening, analysis, drafting and review. Experiments actually ran and metrics were recomputed from prediction files. Explain the assistance honestly and follow the course's disclosure requirements; student review and understanding are still required.

## Before presenting

- Open the report and explain each figure without reading the caption aloud.
- Locate the split manifest, model configuration, training history, prediction CSVs and analysis outputs in the repository.
- Be able to distinguish training from validation and test data, and a mean across seeds from a reference-run result.
- Verify the final submission location and any presentation-duration or AI-disclosure requirements from the course.
