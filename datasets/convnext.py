from benchopt import BaseDataset
from benchopt.config import get_data_path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from torchvision import transforms
from torchvision.datasets import Food101

import timm


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Food101Dataset(TorchDataset):
    def __init__(self, root, train=True, image_size=224, download=True):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        self._inner = Food101(
            root=str(root), split="train" if train else "test",
            download=download, transform=transform,
        )

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        img, label = self._inner[idx]
        return img, torch.tensor(label, dtype=torch.long)


class ConvNeXtV2Wrapper(nn.Module):
    def __init__(self, variant="convnextv2_tiny", num_classes=1000):
        super().__init__()
        self.backbone = timm.create_model(
            variant, pretrained=False, num_classes=num_classes,
        )

    def forward(self, images, labels):
        logits = self.backbone(images)
        loss = F.cross_entropy(logits, labels)
        return loss, logits


class Dataset(BaseDataset):
    name = "convnext"

    parameters = {
        "variant": ["convnextv2_tiny"],
        "image_size": [224],
        "num_classes": [101],
    }

    requirements = ["timm", "torchvision"]

    def get_data(self):
        data_dir = get_data_path("food101")
        dataset = Food101Dataset(
            root=data_dir, train=True,
            image_size=self.image_size, download=True,
        )
        model = ConvNeXtV2Wrapper(
            variant=self.variant, num_classes=self.num_classes,
        )
        return dict(dataset=dataset, model=model)
