"""
Shared dataset utilities for FL-Pneumonia-Detection.

Works both locally (conda env) and on Kaggle/Colab by accepting
an explicit data_dir at construction time.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


class ChestXrayDataset(Dataset):
    """
    Loads chest X-ray images from a directory with NORMAL/ and PNEUMONIA/ subdirs.

    Can be constructed in two ways:
      1. From a directory:  ChestXrayDataset(data_dir="/path/to/split")
      2. From pre-built lists: ChestXrayDataset(images=[...], labels=[...])

    Labels: 0 = NORMAL, 1 = PNEUMONIA
    """

    CLASSES = ["NORMAL", "PNEUMONIA"]

    def __init__(self, data_dir=None, transform=None, images=None, labels=None):
        self.transform = transform

        if images is not None:
            # Constructed from pre-built index lists (used by PneumoniaClient)
            self.images = list(images)
            self.labels = list(labels)
        else:
            if data_dir is None:
                raise ValueError("Provide either data_dir or (images, labels).")
            self.images = []
            self.labels = []
            for label, cls in enumerate(self.CLASSES):
                cls_dir = os.path.join(data_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.images.append(os.path.join(cls_dir, fname))
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def class_counts(self):
        return {
            "NORMAL":    self.labels.count(0),
            "PNEUMONIA": self.labels.count(1),
        }


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def compute_class_weights(dataset):
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Returns a float tensor of shape [2].
    """
    counts = dataset.class_counts()
    total  = sum(counts.values())
    weights = torch.tensor(
        [total / (2 * counts["NORMAL"]),
         total / (2 * counts["PNEUMONIA"])],
        dtype=torch.float,
    )
    return weights
