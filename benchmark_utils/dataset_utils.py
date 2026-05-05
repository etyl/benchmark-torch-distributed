from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataloader(dataset, batch_size):

    if hasattr(dataset, "get_dataloader"):
        return dataset.get_dataloader(batch_size=batch_size)

    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    return dataloader
