"""
generate datasets for working with graph neural networks from the torch_geometric package
for two level cross-validation strategy
"""
import os
import shutil
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, AnyNode
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from tornado.escape import native_str


class HCPDataset_2l(Dataset):
    def __init__(self,
                 cfg: DictConfig,
                 rebuild_processed: bool = False,
                 kind: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None) -> None:
        """
        Initializes an instance of the HCPDataset class.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object.
        rebuild_processed : bool, default=False
            Flag indicating whether to rebuild processed data.
            Defaults to False.
        kind : str, {'train', 'valid', 'test', 'train_valid'}, default='train'
            Type of dataset, which can be 'train', 'valid', or 'test'.
            Defaults to 'train'.
        transform : Optional[Callable], default=None
            Data transformation function.
            Defaults to None.
        pre_transform : Optional[Callable], default=None
            Pre-transformation function for data.
            Defaults to None.
        pre_filter : Optional[Callable], default=None
            Data filtering function.
            Defaults to None.

        Returns
        -------
        None
        """
        self.cfg = cfg
        self.kind = kind
        # Check if directories for extraction exist and create them if necessary
        self.root = self.cfg.data.root_path
        # Remove cached files if required
        # or if the coding type in the configuration file does not match the coding type in
        # the extracted files
        root_path = Path(self.root)
        proper_files = list(root_path.rglob(f'*{self.cfg.data.coding_type}*'))
        if root_path.exists() and (rebuild_processed or not proper_files):
            shutil.rmtree(root_path)

        super().__init__(root=self.root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter,
                         force_reload=rebuild_processed)

    @property
    def raw_paths(self) -> List[Path]:
        """
        Returns a list of paths to the raw data.

        Returns
        -------
        List[Path]
            List of paths to raw data, sorted for consistent order across different runs.
        """
        if Path(self.raw_dir).exists():
            # Sort the list to ensure consistent order across different runs
            return sorted(list(Path(self.raw_dir).glob(f'*{self.cfg.data.coding_type}*')))
        return []

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns a list of file names in the raw_dir directory.

        If the raw_dir directory exists, returns a sorted list of file names
        matching the coding type self.cfg.data.coding_type.
        Otherwise, returns an empty list.

        Returns
        -------
        List[str]
            List of file names.
        """
        if not hasattr(self, 'raw_dir'):
            return []

        raw_dir = Path(self.raw_dir)
        if not raw_dir.exists():
            return []

        return sorted([f.name for f in raw_dir.glob(f'*{self.cfg.data.coding_type}*')])

    @property
    def processed_file_names(self):
        """
        Returns a list of processed files that should be in the processed_dir directory.
        If files are found, processing is skipped.

        Returns
        -------
        List[str]
            List of file names in the processed_dir directory, sorted for consistent order across different runs.
        """
        if not hasattr(self, 'processed_dir'):
            return []

        processed_dir = Path(self.processed_dir)
        if not processed_dir.exists():
            return []

        pattern = 'data_train*.pt'
        if self.kind == 'valid':
            pattern = 'data_valid*.pt'
        elif self.kind == 'test':
            pattern = 'data_test*.pt'
        elif self.kind == 'train_valid':
            pattern = 'data_train_valid*.pt'
        return sorted([f.name for f in processed_dir.glob(pattern)])

    def process(self) -> None:
        """
        Processes raw data files and saves them as processed files.
        Splits file names by '_' to extract labels and graph IDs.
        Reads CSV files, reindexes edge indices if necessary, and creates graph data objects.
        Saves processed data based on the dataset kind ('train', 'valid', 'test', 'train_valid').

        Returns
        -------
        None
        """
        for fpath in self.raw_paths:
            # Split the file name by '_'
            # Assuming the file name format is: [id]_[coding_type]_[label]
            # Remove spaces in the file name
            try_list = [s.strip() for s in fpath.stem.split('_')]
            label = self.cfg.data.dataset.labels[try_list[2]]
            # Append current label to graph labels
            graph_id = try_list[0]
            # Adjacency matrix
            adj_numpy = np.load(fpath)
            # The percent of graph edges to leave
            drop_percentile = self.cfg['drop_percentile'] if 'drop_percentile' in self.cfg.keys() else 0
            if drop_percentile > 0:
                threshold = np.percentile(np.abs(adj_numpy), drop_percentile)
                # Set all elements of Adjacency matrix to zero if np.abs(adj_numpy) <= threshold
                adj_numpy[np.abs(adj_numpy) <= threshold] = 0
                # # Find rows with all zeros
                # mask = np.all(adj_numpy == 0, axis=1)
                # # Remove all rows and columns with all zeros
                # adj_numpy = adj_numpy[~mask, :]
                # adj_numpy = adj_numpy[:, ~mask]

            # in tensor form
            adj = torch.tensor(adj_numpy, dtype=torch.float)
            # sparse matrix
            sparse_data = dense_to_sparse(adj)
            # Use all ones as node features
            node_features = torch.tensor(np.ones(adj.shape[0]).reshape(-1, 1), dtype=torch.float)
            # If correlation graphs, calculate time-averaged features as node features
            if self.cfg.data.dataset_type == 'correlation_graphs':
                features_data = \
                    pd.read_pickle(f'{self.cfg.data['correlation_means_root_path']}'
                                f'/{self.cfg.data.dataset.root.upper()}/{"_".join(try_list)}.pickle')['mean']
                node_features = torch.tensor(features_data.to_numpy().reshape(-1, 1), dtype=torch.float)

            data_to_save = Data(x=node_features,
                                edge_index=sparse_data[0],
                                edge_weight=sparse_data[1],
                                y=label,
                                graph_id=f'{graph_id}_{label}')

            train_valid_ids = pd.read_csv(self.cfg.data.train_valid_ids_path, header=None)[0].to_list()
            test_ids = pd.read_csv(self.cfg.data.test_ids_path, header=None)[0].to_list()

            # self.cfg.data.train_ids - is a string looks lite list => eval
            def _aux_covert_to_list_like(conf_obj: ListConfig | str):
                return eval(conf_obj) if isinstance(conf_obj, str) else conf_obj

            if self.kind == 'test' and int(graph_id) in test_ids:
                torch.save(data_to_save, os.path.join(self.processed_dir, f'data_test_{graph_id}_{label}.pt'))
            elif self.kind == 'train_valid' and int(graph_id) in train_valid_ids:
                torch.save(data_to_save, os.path.join(self.processed_dir, f'data_train_valid_{graph_id}_{label}.pt'))
            else:
                continue
            # # self.cfg.data.train_ids - is a string looks lite list => eval
            # def _aux_covert_to_list_like(conf_obj: ListConfig | str):
            #     return eval(conf_obj) if isinstance(conf_obj, str) else conf_obj
            #
            # if self.kind == 'train' and int(graph_id) in _aux_covert_to_list_like(self.cfg.data.train_ids):
            #     torch.save(data_to_save, os.path.join(self.processed_dir, f'data_train_{graph_id}_{label}.pt'))
            # elif self.kind == 'valid' and int(graph_id) in _aux_covert_to_list_like(self.cfg.data.valid_ids):
            #     torch.save(data_to_save, os.path.join(self.processed_dir, f'data_valid_{graph_id}_{label}.pt'))
            # elif self.kind == 'test' and int(graph_id) in _aux_covert_to_list_like(self.cfg.data.test_ids):
            #     torch.save(data_to_save, os.path.join(self.processed_dir, f'data_test_{graph_id}_{label}.pt'))
            # elif self.kind == 'train_valid' and int(graph_id) in _aux_covert_to_list_like(self.cfg.data.train_valid_ids):
            #     torch.save(data_to_save, os.path.join(self.processed_dir, f'data_train_valid_{graph_id}_{label}.pt'))
            # else:
            #     continue

    def len(self):
        """
        Returns the number of processed files.

        Returns
        -------
        int
            Number of processed files.
        """
        return len(self.processed_file_names)

    def get(self, idx):
        """
        Loads and returns a processed file by index.

        Parameters
        ----------
        idx : int
            Index of the file to load.

        Returns
        -------
        torch.Tensor
            Loaded processed data.
        """
        return torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]),
                          weights_only=False)

    def download(self) -> None:
        """
        Unpacks the archive into the self.raw_dir directory.
        Moves files that match the coding type to the raw directory.
        Removes unnecessary directories.
        If the dataset type is 'correlation_graphs', unpacks additional data for time-averaged features.

        Returns
        -------
        None
        """
        # Destination directory
        raw_path = Path(self.raw_dir)
        # Copy all suitable files to the raw_path
        for file in Path(self.cfg.data['root_source_dir']).rglob(
                f'{self.cfg.data.dataset["root"]}/*{self.cfg.data["coding_type"]}*.npy', case_sensitive=False):
            shutil.copy(src=file, dst=raw_path)
