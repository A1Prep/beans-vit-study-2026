# Third-party notices and attribution

## Makerere Beans dataset and image excerpts

This project uses the bean-leaf dataset from the [Makerere AI Lab original repository](https://github.com/AI-Lab-Makerere/ibean), acquired from the pinned [Hugging Face distribution](https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/27aa014ce09b193e1a6f58112d4a66e0eddb69c5). The raw dataset is downloaded separately and is not included in this repository. Audit contact sheets, sample images and result figures contain image excerpts or transformed versions of dataset images.

The following notice is reproduced verbatim from the original repository's [LICENSE file](https://raw.githubusercontent.com/AI-Lab-Makerere/ibean/master/LICENSE), retrieved 2026-10-23T13:27:29.177761+00:00. SHA256 of the retrieved file bytes: `6bd5d9a34c6a5aa3700e38f4deed272cc116f5c443d68ec461e775a6ee00b9e5`.

```text
MIT License

Copyright (c) 2020 AIR Lab Makerere University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DINOv2 pretrained model

The experiment uses `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, based on the work of Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). Its [pinned model card](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/README.md) identifies the license as Apache-2.0. See the [original DINOv2 repository](https://github.com/facebookresearch/dinov2), its [license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE), and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Pretrained weights and derived trained checkpoints are not distributed in this repository or its lightweight results bundle. The reproduction code downloads the pinned pretrained source separately. Model authorship is credited to the original researchers; project-specific classifiers and analyses are described in the report.

## Software dependencies and project code

Python packages are installed separately from the pinned requirements; their own licenses and notices continue to apply. This repository does not vendor those packages. No repository-wide license has been selected for the original student project code. The notices above preserve third-party attribution and do not apply the dataset or model license to all project files.
