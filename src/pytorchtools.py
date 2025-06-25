import os
import random
from typing import Callable

import numpy as np
import torch
import torch_geometric


def seed_everything(seed: int) -> None:
    """
    Sets the seed for generating random numbers for all used libraries.

    Parameters
    ----------
    seed : int
        The seed value for random number generation.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # When running on GPU with CuDNN, these two parameters should be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # For hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Original from https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self,
                 patience: int = 7,
                 verbose: bool = False,
                 delta: float = 0,
                 path: str = 'checkpoint.pt',
                 trace_func: Callable = print):
        """
        Parameters
        ----------
        patience : int, default=7
            How long to wait after the last time validation loss improved.
        verbose : bool, default=False
            If True, prints a message for each validation loss improvement.
        delta : float, default=0
            Minimum change in the monitored quantity to qualify as an improvement.
        path : str, default='checkpoint.pt'
            Path for the checkpoint to be saved to.
        trace_func : function, default=print
            Trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: torch.nn.Module) -> int:
        """
        Checks if the validation loss has improved and updates the early stopping counter.

        Parameters
        ----------
        val_loss : float
            Current validation loss.
        model : torch.nn.Module
            The model to be saved if the validation loss improves.

        Returns
        -------
        int
            The current counter value.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.counter

    def save_checkpoint(self,
                        val_loss: float,
                        model: torch.nn.Module) -> None:
        """
        Saves the model when the validation loss decreases.

        Parameters
        ----------
        val_loss : float
            Current validation loss.
        model : torch.nn.Module
            The model to be saved.

        Returns
        -------
        None
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
