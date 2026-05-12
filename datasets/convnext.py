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


class Food101Dataset:
    def __init__(self, root, train=True, image_size=224, download=True):
        pass

    def get_dataloader(self, batch_size):
        def get_batch():
            for _ in range(10000):
                x = torch.randn(batch_size, 3, 224, 224)
                y = torch.randn(batch_size, 101)
                yield x, y
        return get_batch()


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
    }

    requirements = ["timm", "torchvision"]

    def get_data(self):
        data_dir = get_data_path("food101")
        dataset = Food101Dataset(
            root=data_dir, train=True,
            image_size=self.image_size, download=True,
        )
        model = ConvNeXtV2Wrapper(
            variant=self.variant, num_classes=101,
        )
        return dict(dataset=dataset, model=model)
