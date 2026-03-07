# CS7643_TeamProject
# SSL Project Progress Report: FixMatch Reproduction

## Overview

This project aims to reproduce and study **FixMatch** for semi-supervised learning on **CIFAR-10 with 250 labeled examples**.  
Our current implementation is written in **PyTorch**, while the original Google Research implementation is written in **TensorFlow**.

---

## Current Progress

### 1. Dataset pipeline
We have implemented the CIFAR-10 semi-supervised data pipeline, including:

- **Class-balanced labeled split**
  - `num_labeled = 250`
  - `25 labeled examples per class`
- **Unlabeled set**
  - currently uses the full CIFAR-10 training set as unlabeled data
- Separate dataloaders for:
  - labeled data
  - unlabeled data
  - test data

This follows the standard FixMatch setup.

---

### 2. Data augmentation
We implemented both **weak** and **strong** augmentations for FixMatch.

#### Weak augmentation
Used for pseudo-label generation:
- Random horizontal flip
- Random crop with reflection padding

#### Strong augmentation
Used for consistency training:
- Random horizontal flip
- Random crop with reflection padding
- `RandAugment(num_ops=2, magnitude=10)`
- Cutout-like corruption using `RandomErasing`

### Comparison with the original FixMatch code
Our augmentation design is conceptually aligned with FixMatch:
- weak augmentation for pseudo-labeling
- strong augmentation for unsupervised consistency loss

However, the original Google Research code uses a more complex augmentation framework:
- CTAugment-based augmentation pool
- Cutout added through the augmentation policy pipeline

So, our augmentation is **functionally similar**, but not an exact line-by-line reproduction.

---

### 3. Model
We implemented a **WideResNet-style classifier** in PyTorch.

Current model:
- WRN-like architecture
- depth = 28
- widen factor = 2
- BatchNorm + LeakyReLU
- global average pooling
- linear classifier

### Comparison with the original FixMatch code
The original FixMatch implementation uses a ResNet/WideResNet-style architecture in TensorFlow.

Our model is **structurally similar**, but not exactly identical in every low-level detail.  
Possible differences include:
- shortcut handling
- block implementation details
- initialization details
- exact BatchNorm behavior

So our model is a **reasonable PyTorch reimplementation**, but not an exact copy of the TensorFlow backbone.

---

### 4. FixMatch loss implementation
We implemented the core FixMatch loss:

- supervised cross-entropy on labeled data
- pseudo-label generation from **weakly augmented unlabeled images**
- confidence thresholding with `threshold = 0.95`
- cross-entropy on **strongly augmented unlabeled images**
- final loss:
  
  \[
  L = L_s + \lambda_u L_u
  \]

with:
- `lambda_u = 1.0`

### Comparison with the original FixMatch code
This part is very close to the original FixMatch algorithm.

Implemented components match the paper and original code:
- pseudo-label from weak augmentation
- threshold-based masking
- strong-augmentation consistency loss
- labeled CE + unlabeled CE

We also updated the implementation to include **interleave / de-interleave style batching** for better BatchNorm behavior, following the original code more closely.

---

### 5. Training pipeline
We implemented a PyTorch training loop with:

- SGD optimizer
- momentum = 0.9
- Nesterov = True
- weight decay = 0.0005
- EMA model
- cosine learning rate schedule
- test accuracy evaluation using the EMA model
- best model checkpoint saving

### Comparison with the original `training.py`
The original Google Research training code is much more abstract and framework-heavy.

Original TensorFlow training features:
- `Experiment` abstraction
- distribution strategy (TPU / multi-GPU support)
- TensorFlow checkpoint manager
- TensorBoard summary writing
- top-1 and top-5 metrics
- EMA and raw model evaluation
- highly modular training/evaluation loop

Our PyTorch training loop is much simpler:
- single-GPU oriented
- direct Python loop
- manual optimizer step
- manual EMA update
- test accuracy only
- manual checkpoint saving

So the training logic is **algorithmically similar**, but the engineering structure is much simpler than the original TensorFlow framework.

---

## Hyperparameter Comparison

Our current config:

```yaml
seed: 42
num_classes: 10
num_labeled: 250

batch_size: 64
mu: 7
epochs: 50

lr: 0.03
momentum: 0.9
weight_decay: 0.0005

threshold: 0.95
lambda_u: 1.0
ema_decay: 0.999

num_workers: 2
save_path: results/fixmatch_best.pt
```

---

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jchang431/CS7643_TeamProject/blob/main/notebooks/run_fixmatch_colab.ipynb)

