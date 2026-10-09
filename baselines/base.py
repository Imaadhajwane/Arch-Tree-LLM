# baselines/base.py
from abc import ABC, abstractmethod
from typing import Dict
import torch


def get_device() -> torch.device:
    """Central device picker. Use everywhere instead of hardcoding cpu/cuda."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


class NASMethod(ABC):

    @abstractmethod
    def search(self, budget: int) -> Dict:
        """Run search for `budget` evaluations. Return dict with eval_history, best_acc, eval_count."""

    @abstractmethod
    def get_metrics(self) -> Dict:
        """Return metrics from last search() call."""