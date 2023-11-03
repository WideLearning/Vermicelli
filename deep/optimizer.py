import torch
import itertools as it
from torch.optim import Optimizer
from collections import defaultdict


class LocalSWA(Optimizer):
    """
    Hybrid of Lookahead and SWA.
    """

    def __init__(self, optimizer, k=6, pullback_momentum="none"):
        """
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.k = k
        self.step_counter = 1
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)
        self.defaults = {}

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["sum_params"] = torch.zeros_like(p.data)
                param_state["sum_params"].copy_(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "alpha": self.alpha,
            "step_counter": self.step_counter,
            "k": self.k,
            "pullback_momentum": self.pullback_momentum,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the sum weights (which typically generalize better)"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["sum_params"])
                p.data.mul_(1 / self.step_counter)

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["sum_params"].add_(p.data)
        self.step_counter += 1

        if self.step_counter >= self.k:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["sum_params"].mul_(1 / self.step_counter)
                    p.data.copy_(param_state["sum_params"])
            self.step_counter = 1

        return loss