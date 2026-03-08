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

For the unlabeled dataset, the dataloader returns: (weak_image, strong_image, augmentation_policy)


The augmentation policy is used when **CTAugment** is enabled.

This follows the standard **FixMatch semi-supervised learning setup**.

---

## 2. Data Augmentation

We implemented both **weak** and **strong** augmentations for FixMatch.

### Weak Augmentation

Used for pseudo-label generation:

- Random horizontal flip  
- Random crop with reflection padding  

Weak augmentation generates a minimally perturbed version of the image for reliable pseudo-label prediction.

---

### Strong Augmentation

Used for consistency training.

We implemented two strong augmentation strategies.

#### RandAugment

- Random horizontal flip  
- Random crop with reflection padding  
- `RandAugment(num_ops=2, magnitude=10)`  
- Cutout-like corruption using `RandomErasing`  

RandAugment uses a **fixed augmentation policy** throughout training.

---

#### CTAugment

We also implemented **CTAugment**, which is used in the official FixMatch implementation.

CTAugment:

- dynamically samples augmentation policies  
- maintains probability distributions for augmentation strengths  
- updates augmentation policies during training  

Supported operations include:

- rotate
- translate
- shear
- brightness
- contrast
- color
- posterize
- solarize
- blur
- rescale
- cutout

Unlike RandAugment, CTAugment is **adaptive**, meaning that the augmentation-policy distribution changes during training based on model feedback.

---

### Comparison with the Original FixMatch Code

Our augmentation design follows the same **weak vs strong augmentation principle** used in FixMatch. Weak augmentation is used to generate pseudo-labels, while strong augmentation is used for consistency training.

Our implementation supports both **RandAugment** (fixed augmentation policy) and **CTAugment** (adaptive augmentation policy). In our CTAugment implementation, augmentation distributions are updated during training using model feedback through `update_rates`.

However, our implementation is still a **simplified approximation of the official Google Research implementation**. In the original FixMatch code, CTAugment policies are updated using a separate **probe augmentation mechanism**, which evaluates augmentation quality independently from the training loss. In contrast, our PyTorch implementation updates augmentation policies directly using model confidence signals observed during training. As a result, our CTAugment still learns an adaptive augmentation distribution, but its update mechanism is simpler than the official implementation.

---

## 3. Model

We implemented a **WideResNet-style classifier** in PyTorch.

### Current Model Configuration

- WRN-like architecture  
- depth = 28  
- widen factor = 2  
- BatchNorm + LeakyReLU  
- global average pooling  
- linear classifier  

---

### Comparison with the Original FixMatch Code

The original FixMatch implementation uses a WideResNet-like architecture implemented in TensorFlow.

Our model is structurally similar, but not exactly identical in every low-level detail.

Possible differences include:

- shortcut implementation  
- block construction  
- parameter initialization  
- exact BatchNorm behavior  

Therefore, our model can be considered a **reasonable PyTorch reimplementation** of the original architecture.

---

## 4. FixMatch Loss Implementation

We implemented the core FixMatch loss:

- supervised cross-entropy on labeled data  
- pseudo-label generation from **weakly augmented unlabeled images**  
- confidence thresholding with `threshold = 0.95`  
- cross-entropy on **strongly augmented unlabeled images**  

### Final Loss
L = L_s + λ_u L_u


where:

- `L_s` = supervised loss  
- `L_u` = unsupervised consistency loss  
- `lambda_u = 1.0`

---

### Comparison with the Original FixMatch Code

This part closely follows the original FixMatch algorithm.

Implemented components match both the **FixMatch paper** and the **official implementation**:

- pseudo-label generation from weak augmentations  
- confidence-based masking  
- strong augmentation consistency training  
- supervised + unsupervised cross entropy  

We also updated the implementation to include:

**interleave / de-interleave batching**

This helps stabilize **BatchNorm statistics**, following the strategy used in the official implementation.

---

## 5. Training Pipeline

We implemented a PyTorch training loop including:

- SGD optimizer  
- momentum = 0.9  
- Nesterov momentum  
- weight decay = 0.0005  
- EMA model  
- cosine learning rate schedule  
- test accuracy evaluation using the EMA model  
- best model checkpoint saving  

---

### Comparison with the Original `training.py`

The original Google Research training code is significantly more complex and framework-heavy.

Original TensorFlow training features include:

- `Experiment` abstraction  
- distributed training (TPU / multi-GPU)  
- TensorFlow checkpoint manager  
- TensorBoard logging  
- top-1 and top-5 metrics  
- EMA and raw model evaluation  
- modular training / evaluation loops  

Our PyTorch implementation is intentionally simpler:

- single-GPU training  
- direct Python training loop  
- manual optimizer updates  
- manual EMA updates  
- test accuracy evaluation only  
- manual checkpoint saving  

Thus, the training logic is **algorithmically similar**, but the engineering structure is simplified.

---

# Hyperparameter Configuration

Our current config:

```yaml
seed: 42
num_classes: 10
num_labeled: 250

batch_size: 64
mu: 7
epochs: 100

# Each training step processes:
# 64 labeled + 64*7 weak unlabeled + 64*7 strong unlabeled = 960 images

lr: 0.03
momentum: 0.9
weight_decay: 0.0005

threshold: 0.95
lambda_u: 1.0
ema_decay: 0.999

augment: ctaugment  # options: randaugment | ctaugment

# IMPORTANT:
# When using CTAugment, num_workers should be 0.
# CTAugment maintains internal state (policy rates) that is updated during training.
# If num_workers > 0, PyTorch DataLoader will create multiple dataset copies
# across worker processes, causing each worker to have its own CTAugment instance.
# In that case, policy updates (update_rates) in the main training loop will not
# correctly affect the augmentation used by the workers.
# For RandAugment this issue does not occur, so num_workers > 0 (e.g., 2 or 4)
# can be used for faster data loading.

num_workers: 0

save_path: results/fixmatch_best.pt

Our current config:

