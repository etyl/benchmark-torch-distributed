from collections import defaultdict
import time
from benchopt import BaseSolver
import os
import torch
import torch.distributed as dist

from benchmark_utils.dataset_utils import get_dataloader
from benchmark_utils.batch_size_probe import get_max_batch_size
from benchmark_utils.distributed_utils import setup_distributed


class Solver(BaseSolver):
    name = "all-reduce"

    parameters = {
        "slurm_nodes": [2]
    }

    requirements = ["pytorch:pytorch"]

    sampling_strategy = "run_once"

    def set_objective(self, dataset, model, device, local_batch_size):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.local_batch_size = local_batch_size

    def run(self, _):
        setup_distributed(self.device)
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        model = self.model.to(device=self.device)
        if self.local_batch_size == -1:
            selected_batch_size = get_max_batch_size(model, self.dataset, self.device)
        else:
            selected_batch_size = self.local_batch_size
        dataloader = get_dataloader(self.dataset, batch_size=selected_batch_size)

        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            start_run = torch.cuda.Event(enable_timing=True)
            start_com = torch.cuda.Event(enable_timing=True)
            end_run = torch.cuda.Event(enable_timing=True)
            end_com = torch.cuda.Event(enable_timing=True)

        optim = torch.optim.Adam(model.parameters(), lr=float(0.01))

        self.logs = defaultdict(list)

        for batch in dataloader:
            optim.zero_grad()

            batch = [x.to(self.device) for x in batch]
            loss, *_ = model(*batch)
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size

            optim.step()
            break

        if use_cuda:
            torch.cuda.synchronize()
            dist.barrier()
            start_run.record()
        else:
            t0_run = time.perf_counter()

        k = 0
        stop_training = False
        while not stop_training:
            for batch in dataloader:
                optim.zero_grad()

                batch = [x.to(self.device) for x in batch]
                loss, *_ = model(*batch)
                loss.backward()

                # Synchronize gradients
                if use_cuda:
                    torch.cuda.synchronize()
                    dist.barrier()
                    start_com.record()
                t0_com = time.perf_counter()

                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size

                if use_cuda:
                    end_com.record()
                    torch.cuda.synchronize()
                    self.logs["comm_time"].append(start_com.elapsed_time(end_com)/1000)
                else:
                    self.logs["comm_time"].append(time.perf_counter() - t0_com)
                self.logs["comm_time_cpu"].append(time.perf_counter() - t0_com)

                optim.step()

                k += 1
                if k > 20:
                    stop_training = True
                    break

        if use_cuda:
            end_run.record()
            torch.cuda.synchronize()
            self.logs["run_time"].append(start_run.elapsed_time(end_run)/1000)
        else:
            self.logs["run_time"].append(time.perf_counter() - t0_run)

    def get_result(self):
        return dict(
            logs=self.logs
        )
