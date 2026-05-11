from benchopt import BaseDataset
import torch.nn as nn
import torch


class MLPDataset:
    def __init__(self, d):
        self.d = d

    def get_dataloader(self, batch_size):
        def get_batch():
            for _ in range(10000):
                x = torch.randn(batch_size, self.d)
                y = torch.randn_like(x)
                yield x, y
        return get_batch()


class MLP(nn.Module):
    def __init__(self, d, layers=1, bias=False):
        super().__init__()
        self.d = d
        self.bias = bias
        layer_list = []
        for _ in range(layers):
            layer_list.append(nn.Linear(self.d, self.d, bias=self.bias))
            layer_list.append(nn.ReLU())
        self.model = nn.Sequential(*layer_list)
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        return loss, None


class Dataset(BaseDataset):
    name = "mlp"

    parameters = {
        'd': [400],
        'layers': [1],
    }
    requirements = ["numpy"]

    def get_data(self):
        dataset = MLPDataset(self.d)
        model = MLP(self.d, self.layers)
        return dict(dataset=dataset, model=model)
