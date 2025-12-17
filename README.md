<div align="center">

<img src="imgs/graphfairness_logo.png" style="width: 60%;"/>

# An Open-Source Toolkit for Fair Graph Deep Learning

</div>
![GitHub Repo stars](https://img.shields.io/github/stars/ZzoomD/GraphFairness)
![GitHub forks](https://img.shields.io/github/forks/ZzoomD/GraphFairness)
![GitHub watchers](https://img.shields.io/github/watchers/ZzoomD/GraphFairness)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ZzoomD/GraphFairness)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/ZzoomD/GraphFairness)
![GitHub repo size](https://img.shields.io/github/repo-size/ZzoomD/GraphFairness)
![GitHub milestones](https://img.shields.io/github/milestones/all/ZzoomD/GraphFairness)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/ZzoomD/GraphFairness)
![GitHub all releases](https://img.shields.io/github/downloads/ZzoomD/GraphFairness/total)
![GitHub issues](https://img.shields.io/github/issues/ZzoomD/GraphFairness)
![GitHub closed issues](https://img.shields.io/github/issues-closed/ZzoomD/GraphFairness)
![GitHub](https://img.shields.io/github/license/ZzoomD/GraphFairness)

## Overview
GroupFairness is an open-source toolkit for fair graph deep learning. It provides a collection of state-of-the-art algorithms for fair graph representation learning. The toolkit is designed to be easy to train and evaluate graph deep learning models. Specifically, GroupFairness includes five modules, i.e., datasets, data loader & pre-processing, model architecture, model training, model evaluation. The overview of the toolkit is shown in the figure below.
<div align="center">
<img src="imgs/overview.png" style="width: 80%;"/>
</div>

## Supported Algorithms
GroupFairness supports the following algorithms which can be categorized into pre-processing and in-processing methods.
- **Pre-processing Methods**: pre-processing methods aim to mitigate the bias in the graph data before training the model.
- **In-processing Methods**: in-processing methods aim to improve the fairness of the model during training.

| **Methods** | **Paper Title** | **Publication** | **Method Category** |
| --- | --- | --- | --- |
| FairDrop | [FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning](https://arxiv.org/pdf/2104.14210) | IEEE Transactions on Artificial Intelligence 2021 | Pre-processing |
| EDITS | [EDITS: Modeling and Mitigating Data Bias for Graph Neural Networks](https://arxiv.org/pdf/2108.05233) | WWW 2022 | Pre-processing |
| Graphair | [Learning Fair Graph Representations via Automated Data Augmentations](https://openreview.net/pdf?id=1_OGWcP1s9w) | ICLR 2023 | Pre-processing |
| FairGNN | [Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information](https://arxiv.org/pdf/2009.01454) | WSDM 2021 | In-processing |
| NIFTY | [Towards a Unified Framework for Fair and Stable Graph Representation Learning](https://arxiv.org/pdf/2102.13186) | UAI 2021 | In-processing |
| FairVGNN | [Improving Fairness in Graph Neural Networks via Mitigating Sensitive Attribute Leakage](https://arxiv.org/pdf/2206.03426) | KDD 2022 | In-processing |
| FairSIN | [FairSIN: Achieving Fairness in Graph Neural Networks through Sensitive Information Neutralization](https://arxiv.org/pdf/2403.12474) | AAAI 2024 | In-processing |
| FairDLA | [Fairdla: Improving the Fairness-Utility Trade-Off in Graph Neural Networks Via Dual-Level Alignment](https://www.sciencedirect.com/science/article/abs/pii/S0950705125008147) | KBS 2025 | In-processing |


