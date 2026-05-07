import os

import torch.distributed as dist


def setup_distributed(device):
    """Maps SLURM variables to PyTorch DDP variables and initializes the process group."""
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # NCCL debugging environment variables
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING"

    if dist.is_initialized():
        return

    if device.startswith("cuda"):
        dist.init_process_group(backend="nccl", init_method="env://")
    elif device == "cpu":
        dist.init_process_group(backend="gloo", init_method="env://")
    else:
        raise ValueError(f"Unsupported device: {device}")
