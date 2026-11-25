"""Strict P1 best and resumable 15-epoch early stopping."""

from dataclasses import asdict, dataclass
import math


@dataclass
class BestTracker:
    patience: int = 15
    best_p1: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, epoch: int, p1: float):
        if epoch <= 0 or not math.isfinite(p1):
            raise ValueError('epoch and P1 must be valid and finite')
        improved = p1 < self.best_p1
        if improved:
            self.best_p1 = p1
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

    def state_dict(self):
        return asdict(self)
