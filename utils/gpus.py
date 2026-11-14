import torch

__all__ = ["getUsableGPUs"]


def getUsableGPUs():
    """Return the visible CUDA device with the most free memory, if any."""
    if not torch.cuda.is_available():
        return None
    return max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
