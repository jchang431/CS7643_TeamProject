# CS7643_TeamProject  
## SSL Project Progress Report: FixMatch Reproduction

---

## Overview

This project aims to reproduce and study **FixMatch** for semi-supervised learning on **CIFAR-10 with 250 labeled examples**.

Our implementation is written in **PyTorch**, while the original Google Research implementation is written in **TensorFlow**.

The goal of this project is to reproduce the FixMatch training pipeline and understand how implementation details influence semi-supervised learning performance.

---

# Current Progress

## 1. Dataset Pipeline

We implemented the CIFAR-10 semi-supervised data pipeline, including:

### Class-balanced labeled split

- `num_labeled = 250`
- `25 labeled examples per class`

### Unlabeled dataset

- The remaining CIFAR-10 training images are used as unlabeled data.

### Separate dataloaders

- labeled data
- unlabeled data
- test data

For the unlabeled dataset, the dataloader returns:
