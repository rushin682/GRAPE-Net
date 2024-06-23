import importlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dataloaders.base_dataset import TissueDataset

from .base_dataset import TissueDataset
from .lung_datasets import TcgaDataset, CptacDataset, PcgaDataset, UclDataset, NlstDataset

def find_dataset_using_name(dataset_name): # Rushin: Probably deprecate soon
    """Import the module "dataloaders/[dataset_name].py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of TissueDataset,
    and it is case-insensitive.
    """
    dataset_filename = "dataloaders." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'Dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, TissueDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of TissueDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def separate_data(graph_list, seed, n_folds, fold_idx):
    assert 0 <= fold_idx and fold_idx < n_folds, "fold_idx must be from 0 to 9."
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True)

    groups = []
    labels = []
    for info in graph_list:
        info = info.replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        patient_id = file_name.rsplit('-', 1)[0]
        groups.append(patient_id)
        labels.append(label)
    idx_list = []
    for idx in sgkf.split(np.zeros(len(labels)), labels, groups=groups):
        idx_list.append(idx)

    train_val_idx, test_idx = idx_list[fold_idx]

    train_val_graph_list = [graph_list[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]

    test_graph_list = [graph_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_val_graph_list, test_graph_list, train_val_labels

def read_file(file_name):
    with open(file_name, 'r') as f:
        records = list(f)

    return records