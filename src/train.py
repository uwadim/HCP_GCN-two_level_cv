# coding: utf-8
""" Training procedure """
import random
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch  # type: ignore
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import sigmoid
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj

from mlflow_helpers import log_params_from_omegaconf_dict
from pytorchtools import EarlyStopping  # type: ignore


def train_one_epoch(model: torch.nn.Module,
                    device: torch.device,
                    train_loader: DataLoader,
                    criterion: torch.nn.Module,
                    optimizer: Optimizer,
                    pooling_type: str) -> float:
    """
    Trains the model for one epoch using all batches.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    device : torch.device
        The device to be used for training (e.g., 'cuda' or 'cpu').
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    pooling_type : str
        Type of pooling to be used.

    Returns
    -------
    float
        Average loss for this epoch.
    """
    running_loss = 0.0
    step = 0
    model.train()
    for data in train_loader:
        data.to(device)  # Use GPU
        optimizer.zero_grad()  # Reset gradients
        out = model(x=data.x,
                    edge_index=data.edge_index,
                    edge_weight=data.edge_weight,
                    pooling_type=pooling_type,
                    batch=data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y.float().reshape(-1, 1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += loss.item()
        step += 1

    return running_loss / step


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module,
                   device: torch.device,
                   test_loader: DataLoader,
                   criterion: torch.nn.Module,
                   pooling_type: str,
                   calc_conf_matrix: bool = False) -> Tuple[float, ...]:
    """
    Tests the model for one epoch using all batches.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be tested.
    device : torch.device
        The device to be used for testing (e.g., 'cuda' or 'cpu').
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    criterion : torch.nn.Module
        Loss function.
    pooling_type : str
        Type of pooling to be used.
    calc_conf_matrix : bool, default=False
        Whether to calculate the confusion matrix.

    Returns
    -------
    tuple[float, ...]
        Average loss, average accuracy, average AUC, and optionally confusion matrix and other metrics.
    """
    model.eval()
    running_loss = 0.0
    step = 0
    batch_auc = []
    batch_accuracy = []
    all_trues = []
    all_preds = []
    all_graph_ids = []

    for data in test_loader:
        data.to(device)  # Use GPU
        out = model(x=data.x,
                    edge_index=data.edge_index,
                    edge_weight=data.edge_weight,
                    pooling_type=pooling_type,
                    batch=data.batch)
        loss = criterion(out, data.y.float().reshape(-1, 1))
        running_loss += loss.item()
        step += 1
        pred = sigmoid(out)
        y_true = data.y.detach().to('cpu').numpy()
        y_pred = pred.detach().to('cpu').numpy()
        batch_accuracy.append(accuracy_score(y_true, y_pred.round()))
        batch_auc.append(roc_auc_score(y_true, y_pred))
        if calc_conf_matrix:
            all_trues.extend(y_true)
            all_preds.extend(y_pred)
            all_graph_ids.extend(data.graph_id)

    if calc_conf_matrix:
        conf_matrix = confusion_matrix(np.asarray(all_trues), np.asarray(all_preds).ravel().round())
        diff = np.asarray(all_trues) - np.asarray(all_preds).ravel().round()
        precision, recall, fscore, support = precision_recall_fscore_support(y_true=np.asarray(all_trues),
                                                                             y_pred=np.asarray(
                                                                                 all_preds).ravel().round(),
                                                                             average=None)
        try_df = pd.DataFrame({'true': all_trues, 'diff': diff.astype(int).tolist(), 'graph_id': all_graph_ids})
        tp_ids = try_df.query('true == 1 and diff == 0')['graph_id'].to_list()
        tn_ids = try_df.query('true == 0 and diff == 0')['graph_id'].to_list()
        fp_ids = try_df.query('diff == -1')['graph_id'].to_list()
        fn_ids = try_df.query('diff == 1')['graph_id'].to_list()

        return (running_loss / step,
                np.mean(batch_accuracy),
                np.mean(batch_auc),
                conf_matrix,
                precision[0],
                recall[0],
                fscore[0],
                support[0],
                tp_ids,
                tn_ids,
                fp_ids,
                fn_ids)
    return (running_loss / step,
            np.mean(batch_auc),
            np.mean(batch_accuracy))


def reset_model(model: torch.nn.Module) -> None:
    """
    Reinitializes the model's layers.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be reinitialized.

    Returns
    -------
    None
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_valid_model(cfg: DictConfig,
                      model: torch.nn.Module,
                      device: torch.device,
                      train_loader: DataLoader,
                      criterion: torch.nn.Module,
                      optimizer: Optimizer,
                      pooling_type: str,
                      scheduler: LRScheduler,
                      valid_loader: Optional[DataLoader] = None,
                      mlflow_object: Optional[Any] = None
                      ) -> Optional[Tuple[float, float, float]]:
    """
    Trains and validates the model for a specified number of epochs.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model and training parameters.
    model : torch.nn.Module
        The model to be trained.
    device : torch.device
        The device to be used for training (e.g., 'cuda' or 'cpu').
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    pooling_type : str
        Type of pooling to be used.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    valid_loader : torch.utils.data.DataLoader, optional, default=None
        DataLoader for the validation dataset.
    mlflow_object : Optional[Any], default=None
        MLflow object for logging metrics and parameters.

    Returns
    -------
    tuple[float, ...] | None
        Validation loss, accuracy, and AUC if valid_loader is not None, else None
    """
    print("Start training ...")
    avg_train_losses = []
    avg_valid_losses = []

    n_epochs = cfg.models.max_epochs
    early_stopping = None
    if cfg.models.model.early_stopping:
        early_stopping = EarlyStopping(**cfg.models.stoping)
    if mlflow_object is not None:
        log_params_from_omegaconf_dict(cfg)
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model=model,
                                     device=device,
                                     train_loader=train_loader,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     pooling_type=pooling_type)
        avg_train_losses.append(train_loss)

        if valid_loader is not None:
            (valid_loss,
             valid_acc,
             valid_auc) = test_one_epoch(model=model,
                                         device=device,
                                         test_loader=valid_loader,
                                         criterion=criterion,
                                         pooling_type=pooling_type,
                                         calc_conf_matrix=False)
            avg_valid_losses.append(valid_loss)
        epoch_len = len(str(n_epochs))
        if mlflow_object is not None:
            mlflow_object.log_metric('train loss', train_loss, step=epoch)
            if valid_loader is not None:
                mlflow_object.log_metric('valid loss', valid_loss, step=epoch)
                mlflow_object.log_metric('valid accuracy', valid_acc, step=epoch)
                mlflow_object.log_metric('valid roc_auc', valid_auc, step=epoch)

        if valid_loader is not None:
            print(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
                  f'train_loss: {train_loss:.5f} '
                  f'valid_loss: {valid_loss:.5f} ')
        else:
            print(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
                  f'train_loss: {train_loss:.5f} ')

        if cfg.models.model.early_stopping:
            patience_counter = early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                if mlflow_object is not None:
                    mlflow_object.log_param('Learned Epochs', epoch - patience_counter)
                break

        if mlflow_object is not None:
            mlflow_object.log_metric('learning rate', scheduler.get_last_lr()[0], step=epoch)
        scheduler.step()

    if cfg.models.model.early_stopping:
        model.load_state_dict(torch.load(cfg.training.stoping['path'], weights_only=False))
    (train_loss,
     train_acc,
     train_auc,
     train_conf_matrix,
     train_precision,
     train_recall,
     train_fscore,
     train_support,
     train_tp_ids,
     train_tn_ids,
     train_fp_ids,
     train_fn_ids) = test_one_epoch(model=model,
                                    device=device,
                                    test_loader=train_loader,
                                    criterion=criterion,
                                    pooling_type=pooling_type,
                                    calc_conf_matrix=True)
    if valid_loader is not None:
        (valid_loss,
         valid_acc,
         valid_auc,
         valid_conf_matrix,
         valid_precision,
         valid_recall,
         valid_fscore,
         valid_support,
         valid_tp_ids,
         valid_tn_ids,
         valid_fp_ids,
         valid_fn_ids) = test_one_epoch(model=model,
                                        device=device,
                                        test_loader=valid_loader,
                                        criterion=criterion,
                                        pooling_type=pooling_type,
                                        calc_conf_matrix=True)
    if mlflow_object is not None:
        mlflow_object.log_metric('Final train loss', train_loss)
        mlflow_object.log_metric('Final train accuracy', train_acc)
        mlflow_object.log_metric('Final train roc_auc', train_auc)
        mlflow_object.log_param('Final train Conf. Matrix', ', '.join(map(str, train_conf_matrix.ravel().tolist())))
        mlflow_object.log_param('Final train precision', train_precision)
        mlflow_object.log_param('Final train recall', train_recall)
        mlflow_object.log_param('Final train fscore', train_fscore)
        mlflow_object.log_param('Final train support', train_support)
        mlflow_object.log_param('Final train True Positive IDs', ', '.join(train_tp_ids))
        mlflow_object.log_param('Final train True Negative IDs', ', '.join(train_tn_ids))
        mlflow_object.log_param('Final train False Positive IDs', ', '.join(train_fp_ids))
        mlflow_object.log_param('Final train False Negative IDs', ', '.join(train_fn_ids))

        if valid_loader is not None:
            mlflow_object.log_metric('Final valid loss', valid_loss)
            mlflow_object.log_metric('Final valid accuracy', valid_acc)
            mlflow_object.log_metric('Final valid roc_auc', valid_auc)
            mlflow_object.log_param('Final valid Conf. Matrix', ', '.join(map(str, valid_conf_matrix.ravel().tolist())))
            mlflow_object.log_param('Final valid precision', valid_precision)
            mlflow_object.log_param('Final valid recall', valid_recall)
            mlflow_object.log_param('Final valid fscore', valid_fscore)
            mlflow_object.log_param('Final valid support', valid_support)
            mlflow_object.log_param('Final valid True Positive IDs', ', '.join(valid_tp_ids))
            mlflow_object.log_param('Final valid True Negative IDs', ', '.join(valid_tn_ids))
            mlflow_object.log_param('Final valid False Positive IDs', ', '.join(valid_fp_ids))
            mlflow_object.log_param('Final valid False Negative IDs', ', '.join(valid_fn_ids))

    if valid_loader is not None:
        print(f'Final TRAIN loss: {train_loss}, accuracy: {train_acc}, AUC ROC: {train_auc}\n'
              f'Final VALID loss: {valid_loss}, accuracy: {valid_acc}, AUC ROC: {valid_auc}\n')
    else:
        print(f'Final TRAIN loss: {train_loss}, accuracy: {train_acc}, AUC ROC: {train_auc}')

    if valid_loader is not None:
        return valid_loss, valid_acc, valid_auc

    return None


def test_model(model: torch.nn.Module,
               device: torch.device,
               test_loader: DataLoader,
               criterion: torch.nn.Module,
               pooling_type: str,
               state_path: Optional[str] = None,
               mlflow_object: Optional[Any] = None,
               plot_dict: Optional[dict] = None) -> None:
    """
    Tests the model on the test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be tested.
    device : torch.device
        The device to be used for testing (e.g., 'cuda' or 'cpu').
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    criterion : torch.nn.Module
        Loss function.
    pooling_type : str
        Type of pooling to be used.
    state_path : Optional[str], default=None
        Path to the saved model state.
    mlflow_object : Optional[Any], default=None
        MLflow object for logging metrics and parameters.
    plot_dict : Optional[dict], default=None
        Dictionary containing plotting parameters.

    Returns
    -------
    None
    """
    if state_path is not None:
        model.load_state_dict(torch.load(state_path, weights_only=False))
    (loss,
     accuracy,
     auc,
     conf_matrix,
     precision,
     recall,
     fscore,
     support,
     tp_ids,
     tn_ids,
     fp_ids,
     fn_ids
     ) = test_one_epoch(model=model,
                        device=device,
                        test_loader=test_loader,
                        criterion=criterion,
                        pooling_type=pooling_type,
                        calc_conf_matrix=True)
    print(f'Final TEST loss: {loss},'
          f' accuracy: {accuracy}'
          f' AUC ROC: {auc}\n'
          f'Conf Matrix: {conf_matrix}')

    with open(plot_dict['result_path'], 'a+') as fh:
        percentile_str = f'drop_percentile;{plot_dict["drop_percentile"]};' if 'drop_percentile' in plot_dict else ''
        fh.write(
            f'{percentile_str}'
            f'accuracy;{accuracy};AUCROC;{auc};precision;{precision};'
            f'recall;{recall};fscore;{fscore};support;{support}\n')

    if mlflow_object is not None:
        mlflow_object.log_metric('Test loss', loss)
        mlflow_object.log_metric('Test accuracy', accuracy)
        mlflow_object.log_metric('Test roc_auc', auc)
        mlflow_object.log_param('Test Conf Matrix', ', '.join(map(str, conf_matrix.ravel().tolist())))
        mlflow_object.log_metric('Test precision', precision)
        mlflow_object.log_metric('Test recall', recall)
        mlflow_object.log_param('Test fscore', fscore)
        mlflow_object.log_param('Test support', support)
        mlflow_object.log_param('Test True Positive IDs', ', '.join(tp_ids))
        mlflow_object.log_param('Test True Negative IDs', ', '.join(tn_ids))
        mlflow_object.log_param('Test False Positive IDs', ', '.join(fp_ids))
        mlflow_object.log_param('Test False Negative IDs', ', '.join(fn_ids))
        if 'fname' in plot_dict.keys():
            print('Save adjacency matrices...')
            random.seed(plot_dict['seed'])
            for ids, ids_name in zip([tp_ids, tn_ids, fp_ids, fn_ids], ['TP', 'TN', 'FP', 'FN']):
                elements = random.sample(ids, min(len(ids), plot_dict['max_elements']))
                for el in elements:
                    graph = torch.load(f'{plot_dict["processed_dir"]}/data_test_{el}.pt', weights_only=False)
                    adj = to_dense_adj(edge_index=graph.edge_index, edge_attr=graph.edge_weight)[0].numpy()
                    adj = adj + adj.T - np.diag(np.diag(adj))
                    plot_adj_matrix(adj_matrix=adj,
                                    pic_name=f'{ids_name}_{plot_dict['fname']}_{el}.png',
                                    palette_name=plot_dict['palette_name'],
                                    kind=ids_name,
                                    dataset_type=plot_dict['dataset_type'],
                                    mlflow_object=mlflow_object)


def plot_adj_matrix(adj_matrix: np.ndarray,
                    pic_name: str,
                    palette_name: str,
                    kind: str,
                    dataset_type: str,
                    mlflow_object: Optional[Any] = None) -> None:
    """
    Plots the adjacency matrix and logs it to MLflow.

    Parameters
    ----------
    adj_matrix : np.ndarray
        The adjacency matrix to be plotted.
    pic_name : str
        Name of the picture file.
    palette_name : str
        Name of the color palette to be used.
    kind : str
        Type of the matrix (e.g., 'TP', 'TN', 'FP', 'FN').
    dataset_type : str
        Type of the dataset (e.g., 'correlation_graphs', 'ensemble_graphs').
    mlflow_object : Optional[Any], default=None
        MLflow object for logging metrics and parameters.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    if dataset_type == 'correlation_graphs':
        vmin = -6
        vmax = 6
    elif dataset_type == 'ensemble_graphs':
        vmin = -1
        vmax = 1
    else:
        raise ValueError(f'Значение dataset_type = {dataset_type} не поддерживается')
    ax = sns.heatmap(adj_matrix,
                     cmap=sns.color_palette(palette_name, as_cmap=True),
                     vmin=vmin,
                     vmax=vmax)
    ax.set(xlabel='graph index', ylabel='graph index')
    ax.set_title(f'Test {kind} adjacency matrix')
    mlflow_object.log_figure(fig, pic_name)
    plt.close(fig)
