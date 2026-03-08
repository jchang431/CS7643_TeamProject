import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.augment.ctaugment import CTAugment, apply, pil_to_cta_array, cta_array_to_pil


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class TransformFixMatch:
    def __init__(self, augment="randaugment", mean=CIFAR10_MEAN, std=CIFAR10_STD):
        self.augment = augment

        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        ])

        self.strong_rand = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandAugment(num_ops=2, magnitude=10),
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.cutout = transforms.RandomErasing(
            p=1.0,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random",
        )

        self.cta = CTAugment() if augment == "ctaugment" else None

    def __call__(self, img):
        weak = self.weak(img)

        if self.augment == "randaugment":
            strong = self.strong_rand(img)
            policy = None

        elif self.augment == "ctaugment":
            strong_base = self.weak(img)
            x = pil_to_cta_array(strong_base)
            policy = self.cta.policy(probe=False)
            x = apply(x, policy)
            strong = cta_array_to_pil(x)

        else:
            raise ValueError(f"Unknown augment: {self.augment}")

        weak = self.normalize(weak)
        strong = self.normalize(strong)
        strong = self.cutout(strong)

        return weak, strong, policy


def get_labeled_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def x_u_split(labels, num_labeled, num_classes=10, seed=42):
    """
    FixMatch-style class-balanced split.
    Example: num_labeled=250 -> 25 labels per class
    """
    labels = np.array(labels)
    assert num_labeled % num_classes == 0

    label_per_class = num_labeled // num_classes
    rng = np.random.RandomState(seed)

    labeled_idx = []
    unlabeled_idx = np.arange(len(labels))

    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        rng.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])

    labeled_idx = np.array(labeled_idx)
    rng.shuffle(labeled_idx)

    return labeled_idx, unlabeled_idx


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(
        self,
        root,
        indexs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs].tolist()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10Unlabeled(CIFAR10SSL):
    def __getitem__(self, index):
        img, _ = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            weak, strong, policy = self.transform(img)
        else:
            weak, strong, policy = img, img, None

        return weak, strong, policy


def unlabeled_collate_fn(batch):
    weaks, strongs, policies = zip(*batch)
    weaks = torch.stack(weaks, dim=0)
    strongs = torch.stack(strongs, dim=0)
    return weaks, strongs, list(policies)


def get_fixmatch_dataloaders(
    num_labeled=250,
    batch_size=64,
    mu=7,
    num_workers=2,
    seed=42,
    augment="randaugment",
):
    base_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    labels = base_dataset.targets

    labeled_idx, unlabeled_idx = x_u_split(
        labels=labels,
        num_labeled=num_labeled,
        num_classes=10,
        seed=seed,
    )

    train_labeled_dataset = CIFAR10SSL(
        root="./data",
        indexs=labeled_idx,
        train=True,
        transform=get_labeled_transform(),
        download=False,
    )

    train_unlabeled_dataset = CIFAR10Unlabeled(
        root="./data",
        indexs=unlabeled_idx,
        train=True,
        transform=TransformFixMatch(augment=augment),
        download=False,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=get_test_transform(),
        download=True,
    )

    labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=batch_size * mu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=unlabeled_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return labeled_loader, unlabeled_loader, test_loader