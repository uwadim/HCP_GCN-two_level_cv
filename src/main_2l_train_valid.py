"""
Main function for training and evaluating a graph classification model using GCN with 3 convolutional layers
 and skip connections.
To work with for two level cross-validation strategy in simple mode - use only 1 of 4 options of data division

It is expected that the best hyperparameters are found, so we use the train and valid sample for training,
 and the test for verification.

Parameters:
    cfg (DictConfig): Configuration object containing all necessary parameters for the experiment.

Returns:
    float: Validation accuracy, which is the metric to optimize during hyperparameter tuning.
"""

import logging

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CyclicLR, LinearLR
from torch_geometric import set_debug
from torch_geometric.loader import DataLoader, ImbalancedSampler
from typing_extensions import Optional

from dataset_2l import HCPDataset_2l
from models import GCN, SkipGCN
from pytorchtools import seed_everything
from samplers import StratifiedKFoldSampler, StratifiedSampler
from train import reset_model, test_model, train_valid_model

# Enable debug mode in PyTorch Geometric (optional)
# set_debug(True)
log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../configs", config_name="train_valid_2l_corr")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function to execute the training and evaluation pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing all necessary parameters.

    Returns
    -------
    float, optional:
        Validation accuracy, which is the metric to optimize during hyperparameter tuning | None.
    """
    # If search for optimal number of edges is enabled, update model parameters with the best
    # parameters from a previous experiment
    if 'filtering_search' in cfg.keys() and cfg['filtering_search']:
        config_df = pd.read_csv('configs/dataset_best_params.csv')
        cfg.models.model.params['hidden_channels'] = \
            config_df.query(f'dataset == "{cfg.data.dataset.root}"'
                            f' and dataset_type == "{cfg.data.dataset_type}"')['hidden_channels'].to_list()[0]
        cfg.models.model.params['dropout'] = \
            config_df.query(f'dataset == "{cfg.data.dataset.root}"'
                            f' and dataset_type == "{cfg.data.dataset_type}"')['dropout'].to_list()[0]

    # Fix the seed for reproducibility
    seed_everything(cfg.models.random_state)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## For DEBUGGING (use CPU)
    # device = torch.device('cpu')

    # parameter for searching the optimal percent of dropped edges
    rebuild_processed = cfg['rebuild_processed'] \
        if 'rebuild_processed' in cfg.keys() and cfg['rebuild_processed'] else False
    # Prepare the training, validation, and test datasets
    train_valid_dataset = HCPDataset_2l(cfg=cfg, kind='train_valid', rebuild_processed=rebuild_processed)
    # rebuild_processed=False - we do not want to erase the train dataset
    test_dataset = HCPDataset_2l(cfg=cfg, kind='test', rebuild_processed=False)

    train_valid_sampler = ImbalancedSampler(train_valid_dataset)
    test_sampler = ImbalancedSampler(test_dataset)
    train_valid_loader = DataLoader(train_valid_dataset,
                                    batch_size=cfg.models.model.params['batch_size'],  # len(train_dataset),
                                    shuffle=False,
                                    sampler=train_valid_sampler,
                                    drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.models.model.params['batch_size'],  # len(test_dataset),
                             shuffle=False,
                             sampler=test_sampler,
                             drop_last=True
                             )

    # Initialize the model with parameters from the configuration file
    model = eval(f'{cfg.models.model.name}(model_params={cfg.models.model.params},'
                 f'num_node_features={train_valid_dataset.num_node_features})')
    reset_model(model)  # Reinitialize the model

    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.models.model['learning_rate'])

    # Define the learning rate scheduler
    scheduler = LinearLR(optimizer,
                         start_factor=1,
                         end_factor=0.1,
                         total_iters=cfg.models['max_epochs'])
    # scheduler = CyclicLR(optimizer,
    #                      base_lr=0.0005,
    #                      max_lr=0.05,
    #                      step_size_up=10,
    #                      mode="triangular2")

    # Set up MLflow for experiment tracking
    mlflow.set_tracking_uri(uri=cfg.mlflow['tracking_uri'])  # type: ignore
    mlflow.set_experiment(experiment_name=cfg.mlflow['experiment_name'])  # type: ignore

    # Define the run name for MLflow
    run_name = f'main'

    with mlflow.start_run(run_name=run_name):  # type: ignore
        # Train and validate the model
        train_valid_model(cfg=cfg,
                          model=model,
                          device=device,
                          train_loader=train_valid_loader,
                          valid_loader=None,
                          criterion=criterion,
                          optimizer=optimizer,
                          pooling_type=cfg.models.model.params['pooling_type'],
                          scheduler=scheduler,
                          mlflow_object=mlflow)

        # Prepare a dictionary for plotting (optional)
        plot_dict = {'result_path': f'results/{cfg.mlflow["experiment_name"]}.txt'}
        if cfg.mlflow['save_adjacency_matrices']:
            plot_dict['fname'] = cfg.mlflow['experiment_name']
            plot_dict['processed_dir'] = f'{cfg.data["root_path"]}/processed'
            plot_dict['seed'] = cfg.models.random_state
            plot_dict['max_elements'] = cfg.mlflow['max_elements']
            plot_dict['palette_name'] = cfg.mlflow['palette_name']
            plot_dict['dataset_type'] = cfg.data['dataset_type']

        # Test the model on the test dataset
        test_model(model=model,
                   device=device,
                   test_loader=test_loader,
                   criterion=criterion,
                   pooling_type=cfg.models.model.params['pooling_type'],
                   state_path=None,
                   mlflow_object=mlflow,
                   plot_dict=plot_dict)

    print('End!')

    return None


if __name__ == "__main__":
    main()
