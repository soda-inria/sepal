[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/inria-soda/sepal-datasets)
[![arXiv](https://img.shields.io/badge/arXiv-2507.00965-blue.svg)](https://arxiv.org/pdf/2507.00965v2)


# SEPAL: Scalable Feature Learning on Huge Knowledge Graphs for Downstream Machine Learning

![SEPAL pipeline](images/pipeline.png)

This repository contains the implementation of the paper:

> **Scalable Feature Learning on Huge Knowledge Graphs for Downstream Machine Learning**  
> Félix Lefebvre and Gaël Varoquaux  
> NeurIPS 2025  
> PDF: https://arxiv.org/pdf/2507.00965v2

## ✨ Highlights
- Scales to knowledge graphs with millions of entities
- Robust to highly skewed degree distributions
- Produces embeddings for downstream regression and classification tasks

Method details and ablations are in the paper.

## 🧪 Example
- Mini YAGO3 tutorial: `examples/mini_yago3_embeddings.ipynb`

## 📊 Datasets
- Downstream tables and Mini YAGO3: https://huggingface.co/datasets/inria-soda/sepal-datasets

## 📣 Citation
If you use SEPAL, please cite:
```bibtex
@inproceedings{lefebvre2025scalable,
  title={Scalable Feature Learning on Huge Knowledge Graphs for Downstream Machine Learning},
  author={Lefebvre, Félix and Varoquaux, Gaël},
  booktitle={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```
