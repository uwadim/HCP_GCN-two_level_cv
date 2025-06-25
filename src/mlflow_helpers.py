"""
Helper functions for working with MLflow.

This module provides utility functions to log parameters from an OmegaConf configuration dictionary
into MLflow for experiment tracking.
"""
import mlflow  # type: ignore

from omegaconf import DictConfig, ListConfig
from typing import Any


def log_params_from_omegaconf_dict(params: DictConfig) -> None:
    """
    Logs parameters from an OmegaConf dictionary into MLflow.

    This function iterates through the provided dictionary and logs each parameter to MLflow.
    It handles nested dictionaries and lists by recursively exploring them.

    Parameters
    ----------
    params : DictConfig
        The OmegaConf dictionary containing the parameters to log.

    Returns
    -------
    None
    """
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name: str, element: DictConfig | ListConfig | Any) -> None:
    """
    Recursively explores and logs parameters from nested OmegaConf dictionaries and lists.

    This function is a helper function for `log_params_from_omegaconf_dict`. It recursively
    traverses through nested dictionaries and lists, logging each parameter to MLflow.

    Parameters
    ----------
    parent_name : str
        The name of the parent parameter.
    element : DictConfig | ListConfig | Any
        The element to explore and log.

    Returns
    -------
    None
    """
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, (DictConfig, ListConfig)):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
