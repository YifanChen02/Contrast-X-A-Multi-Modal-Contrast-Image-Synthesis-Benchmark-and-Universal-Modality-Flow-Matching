# Contrast-X

<div align="center">

## [Contrast-X: A Multi-Modal Contrast Image Synthesis Benchmark and Universal Modality Flow Matching](https://arxiv.org/abs/2601.15884v2)

[![arXiv](https://img.shields.io/badge/arXiv-2601.15884-b31b1b.svg)](https://arxiv.org/abs/2601.15884v2)
[![Dataset](https://img.shields.io/badge/Dataset-Apply%20for%20Access-green)](#dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
  - [Dataset Information](#dataset-information)
  - [Dataset Access](#dataset-access)
  - [Dataset Processing Code](#dataset-processing-code)
- [Model](#model)
  - [Model Code](#model-code)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

<div align="center">
  
<img src="https://github.com/YifanChen02/Contrast-X-A-Multi-Modal-Contrast-Image-Synthesis-Benchmark-and-Universal-Modality-Flow-Matching/blob/main/assets/contrast_tease.jpg?raw=true" width="95%" alt="Contrast-X teaser">

</div>

---

## Dataset

We provide the **Contrast-X** benchmark dataset for research purposes.

The source collections are released under CC BY 3.0 or CC BY 4.0 licenses, which permit sharing and adaptation with proper attribution. Access to the processed Contrast-X dataset is provided upon request and approval to ensure responsible research use and compliance with applicable data-use requirements.


### Dataset Information

**Detailed statistics of the Contrast-X dataset.** The benchmark contains **2,642 paired cases** across **MR** and **CT** modalities, covering **11 organs** and multiple clinical systems.

<div align="center">

<img src="https://raw.githubusercontent.com/YifanChen02/Contrast-X-A-Multi-Modal-Contrast-Image-Synthesis-Benchmark-and-Universal-Modality-Flow-Matching/main/assets/Detailed%20statistics%20of%20the%20Contrast-X%20dataset.png?v=2" width="95%">

</div>


### Dataset Access

The processed Contrast-X dataset is available upon request for research use. To ensure responsible data use, applicants are required to complete an access request form and agree to the dataset usage terms.

[Apply for Dataset Access via Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdGr4vSBYOSA2BvAW_hnAwrhnB8d5VgAfacrAjfqxA1aPZZCw/viewform?usp=publish-editor)


Approved applicants will receive access to the dataset through Google Drive.

### Dataset Processing Code

The scripts for data preprocessing, metadata generation, pair construction, and quality control will be released soon.

---

## Model

### Model Code

The model training and inference code is in [`FlowMI/`](FlowMI):

```
FlowMI/
├── autoencoder/          # slot / PoE autoencoders and their training scripts
├── contrastx_dataloader/ # Contrast-X dataset and preprocessing
├── flow/                 # flow matching model, data, training and evaluation
└── tools/                # evaluation and analysis utilities
```


---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{contrastx2026,
  title={Contrast-X: A Multi-Modal Contrast Image Synthesis Benchmark and Universal Modality Flow Matching},
  author={Chen, Yifan and Yin, Fei and Chen, Hao and Wu, Jia and Li, Chao},
  journal={arXiv preprint arXiv:2601.15884},
  year={2026}
}
```

---

## License

The code in this repository is released under the MIT License.

The processed **Contrast-X** dataset is released under the **CC BY 4.0 License**, with attribution to the original source collections. The source collections are released under CC BY 3.0 or CC BY 4.0 licenses. Users must comply with the applicable terms of the original data sources.

Dataset access is provided upon request and approval to ensure responsible research use and compliance with applicable data-use requirements.


---

## Contact

For questions about the dataset or code release, please open an issue or contact the authors.
