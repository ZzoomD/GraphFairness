import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import remove_self_loops, to_undirected

from typing import List, Tuple
import scipy.sparse as sp
import pandas as pd

from torch_geometric.utils import dropout_adj, convert
from torch_sparse import SparseTensor

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import torch
import scipy.sparse as sp


class DictObject:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    
    def to(self, device):
        for key, value in self.__dict__.items():
            if hasattr(value, 'to') and callable(getattr(value, 'to')):
                self.__dict__[key] = value.to(device)
        return self


DATASETS = {
    'german',
    'bail',
    'credit',
    'pokec_z',
    'pokec_n',
    'nba',
    'germanA',
    'bailA',
    'creditA'
}


"""
DATASET_MAP is a variable that maps dataset names to file names, sensitive attribute, and predict attribute.
"""
DATASET_MAP = {
    "german": {"csv_name":'german', "sens_attr":'Gender', 
                "predict_attr":'GoodCustomer', "remove_attr":['GoodCustomer', 'OtherLoansAtStore', 'PurposeOfLoan'],
                "sens_idx":0, "label_number":100},
    "bail": {"csv_name":'bail', "sens_attr":'WHITE', 
                "predict_attr":'RECID', "remove_attr":['RECID'],
                "sens_idx":0, "label_number":100},
    "credit": {"csv_name":'credit', "sens_attr":'Age', 
                "predict_attr":'NoDefaultNextMonth', "remove_attr":['NoDefaultNextMonth', 'Single'],
                "sens_idx":1, "label_number":6000},
    "pokec_z": {"csv_name":'region_job', "sens_attr":'region', 
                "predict_attr":'I_am_working_in_field', "remove_attr":['I_am_working_in_field', 'user_id'],
                "sens_idx":3, "label_number":4000},
    "pokec_n": {"csv_name":'region_job_2', "sens_attr":'region', 
                "predict_attr":'I_am_working_in_field', "remove_attr":['I_am_working_in_field', 'user_id'],
                "sens_idx":3, "label_number":3500},
    "nba": {"csv_name":'nba', "sens_attr":'country', 
                "predict_attr":'SALARY', "remove_attr":['SALARY', 'user_id'],
                "sens_idx":35, "label_number":100},
    "germanA": {"csv_name":'germanA', "sens_attr":'Gender', 
                "predict_attr":'GoodCustomer', "remove_attr":['GoodCustomer', 'OtherLoansAtStore', 'PurposeOfLoan']},
    "bailA": {"csv_name":'bailA', "sens_attr":'WHITE', 
                "predict_attr":'RECID', "remove_attr":['RECID']},
    "creditA": {"csv_name":'creditA', "sens_attr":'Age', 
                "predict_attr":'NoDefaultNextMonth', "remove_attr":['NoDefaultNextMonth', 'Single']}
}


class FairDataset:
    r"""Graph dataset class for GraphFairness framework.
    
    This class loads and processes graph datasets for fairness research stored in CSV format.
    Each dataset consists of a single graph with node features, sensitive attributes,
    and binary classification labels. The class handles data downloading, graph
    construction, and train/validation/test splitting.
    
    Supported datasets include german, bail, credit, pokec_z, pokec_n, nba, etc.
    Each dataset has predefined sensitive attributes and prediction targets.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    name : str
        The name of the dataset. See :meth:`available_datasets`
        for all available datasets.

    Attributes
    ----------
    data : DictObject
        Processed dataset containing features, edge_index, labels, sens,
        and train/val/test indices.
    name : str
        Lowercase dataset name.
    data_dir : str
        Path to the dataset file.

    Example
    -------
    >>> from graphfairness.datasets import FairDataset

    >>> GraphDataset.available_datasets() # see all available datasets.
    >>> dataset = FairDataset(root='./data', name='german')
    >>> data = dataset.data  # Access processed data
    >>> print(data.features.shape)  # Node features
    >>> print(data.labels.unique())  # Classification labels
    >>> print(data.sens.unique())  # Sensitive attributes
    """

    url = 'https://github.com/ZzoomD/FairData/raw/refs/heads/main/datasets/'

    def __init__(self, root: str, name: str):
        self.name = name.lower()
        self.root = root
        if self.name not in DATASETS:
            raise ValueError(
                f'Unknown dataset {name}. Please take a look at '
                '`FairDataset.available_datasets()` for more information.')
        super().__init__()
        self.data_dir = osp.join(self.root, f'GraphFairness-{self.name}')
        if not osp.exists(self.data_dir):
            self.download()
        self.data = self.load_data(self.name, self.data_dir, 
                                    DATASET_MAP[self.name]["label_number"], split_ratio=[0.5, 0.25, 0.25])
    
    def load_data(self, dataset_name: str, file_base_path: str, label_number: int=100,
                    split_ratio: List[float]=[0.5, 0.25, 0.25]) -> DictObject:
        """
        Load and process commonly used fair graph dataset.
        
        This function loads CSV data, builds graph structure, processes features and labels,
        and splits the dataset into train/validation(optional)/test sets for fairness-aware graph learning.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load (e.g., 'german', 'bail', 'credit', 'pokec_z', etc.)
        file_base_path : str
            Base directory path containing the dataset files
        split_ratio : List[float], optional
            Training/validation/testing split ratios, by default [0.5, 0.25, 0.25]
        label_number : int, optional
            Maximum number of training samples, by default None
            
        Returns
        -------
        DictObject
            Processed dataset containing:
            - features: Node feature matrix (torch.FloatTensor)
            - edge_index: Graph edge indices (SparseTensor)
            - labels: Node labels (torch.LongTensor)
            - sens: Sensitive attributes (torch.FloatTensor)
            - idx_train: Training node indices (torch.LongTensor)
            - idx_val: Validation node indices (torch.LongTensor)
            - idx_test: Testing node indices (torch.LongTensor)
            
        Notes
        -----
        The function handles different dataset types:
        - Loads CSV data and removes specified attributes
        - Builds graph edges from relationship files or constructs them using similarity thresholds
        - Processes features with normalization (preserving sensitive attributes for some datasets)
        - Handles different label formats and sensitive attribute encodings
        - Splits data considering class imbalance for binary classification datasets
        """
        # load csv data and remove attributes
        print(osp.join(file_base_path, f'{DATASET_MAP[dataset_name]["csv_name"]}.csv'))
        idx_features_labels = pd.read_csv(osp.join(file_base_path, f'{DATASET_MAP[dataset_name]["csv_name"]}.csv'))
        header = list(idx_features_labels.columns)
        idx_by_id = True if 'user_id' in header else False
        remove_attr = DATASET_MAP[dataset_name].get("remove_attr", [])
        for attr in remove_attr:
            header.remove(attr)

        # get featrues and labels according to removed attribute header
        predict_attr = DATASET_MAP[dataset_name].get("predict_attr", None)
        if predict_attr is None:
            raise ValueError(f'No predict attribute found for dataset {dataset_name}')
        labels = idx_features_labels[predict_attr].values
        if dataset_name == 'german':
            idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
            idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
            labels[labels == -1] = 0
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    
        # build graph
        if idx_by_id:
            idx = np.array(idx_features_labels["user_id"], dtype=int)
        else:
            idx = np.arange(features.shape[0])
        
        idx_map = {j: i for i, j in enumerate(idx)}
        edge_file_path = osp.join(file_base_path, f'{DATASET_MAP[dataset_name]["csv_name"]}_{"edges" if dataset_name in ["bail", "credit", "german"] else "relationship"}.txt')
        
        if osp.exists(edge_file_path):
            if dataset_name in ["bail", "credit", "german"]:
                edges_unordered = np.genfromtxt(edge_file_path).astype('int')
            else:
                edges_unordered = np.genfromtxt(edge_file_path, dtype=int)
        else:
            # german: 0.8, bail: 0.6, credit: 0.7
            thresh_dict = {"german": 0.8, "bail": 0.6, "credit": 0.7}
            edges_unordered = build_relationship(idx_features_labels[header], thresh=thresh_dict[dataset_name])
            np.savetxt(edge_file_path, edges_unordered)
        
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        edge_index = self.adj2edge_index(adj, features.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        if dataset_name in ['bail', 'credit', 'nba']:
            norm_features = feature_norm(features)
            norm_features[:, DATASET_MAP[dataset_name]["sens_idx"]] = features[:, DATASET_MAP[dataset_name]["sens_idx"]]
            features = norm_features
        labels = torch.LongTensor(labels)

        def get_label_idx():
            import random
            random.seed(20)
            if dataset_name in ['pokec_z', 'pokec_n', 'nba']:
                label_idx = np.where(labels >= 0)[0]
                random.shuffle(label_idx)
                label_idx = [label_idx]
            else:
                label_idx_0 = np.where(labels == 0)[0]
                label_idx_1 = np.where(labels == 1)[0]
                random.shuffle(label_idx_0)
                random.shuffle(label_idx_1)
                label_idx = [label_idx_0, label_idx_1]
            return label_idx
        
        # split the dataset
        label_idx = get_label_idx()
        idx_train, idx_val, idx_test = self.split_nodes(label_idx, split_ratio, label_number)
        
        sens = torch.FloatTensor(idx_features_labels[DATASET_MAP[dataset_name]["sens_attr"]].values.astype(int))
        
        if dataset_name in ['pokec_z', 'pokec_n', 'nba']:
            labels[labels > 1] = 1

        data_dict = dict(dataset=dataset_name, features=features, edge_index=edge_index, 
                         labels=labels, sens=sens, 
                         idx_train=idx_train, idx_val=idx_val, idx_test=idx_test
                         )
        data = DictObject(data_dict)

        return data


    def split_nodes(self, label_idx: List, 
                    split_ratio: List[float]=[0.5, 0.25, 0.25], 
                    label_number: int=100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the node indices into training, validation (optional) and testing sets.

        Parameters
        ----------
        label_idx : List
            List of node index lists for each class. Length 1 (no class balance) or 2 (with class balance).
        split_ratio : List[float], optional
            Train/val/test ratios, default [0.5, 0.25, 0.25]. Length 2 (train/test) or 3 (train/val/test).
        label_number : int, optional
            Maximum training nodes, default 100

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The training, validation and testing indices as PyTorch tensors.
        """
        idx_train = np.concatenate([label_idx_sub[:min(int(split_ratio[0] * len(label_idx_sub)), \
                                    label_number // len(label_idx))] for label_idx_sub in label_idx])

        if len(split_ratio) == 3:
            idx_val = np.concatenate([label_idx_sub[int(split_ratio[0] * len(label_idx_sub)):int((split_ratio[0] + split_ratio[1]) * len(label_idx_sub))] \
                                        for label_idx_sub in label_idx])
            idx_test = np.concatenate([label_idx_sub[int((split_ratio[0] + split_ratio[1]) * len(label_idx_sub)):] for label_idx_sub in label_idx])
        else:
            idx_test = np.concatenate([label_idx_sub[min(int(split_ratio[0] * len(label_idx_sub)), label_number // len(label_idx)):] for label_idx_sub in label_idx])
            idx_val = idx_test

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        
        return idx_train, idx_val, idx_test

    def adj2edge_index(self, adj, nodes_num):
        """
        Convert scipy sparse adjacency matrix to PyTorch Geometric edge index format.
        The returned edge index is a sparse tensor with shape (2, num_edges), facilitating the reproducibility under the same random seed.
        
        Parameters
        ----------
        adj : scipy.sparse matrix
            Sparse adjacency matrix of the graph
        nodes_num : int
            Number of nodes in the graph
            
        Returns
        -------
        torch_sparse.SparseTensor
            Sparse tensor representation of edge indices compatible with PyTorch Geometric
        """
        edge_index_ori = convert.from_scipy_sparse_matrix(adj)[0]
        edge_index = SparseTensor.from_edge_index(edge_index_ori, sparse_sizes=(nodes_num, nodes_num), )
        return edge_index

    @property
    def show_dir(self) -> str:
        return self.data_dir

    def download(self):
        download_url(osp.join(self.url, f'{self.name}/{DATASET_MAP[self.name]["csv_name"]}.csv'), self.data_dir)
        if self.name in ['german', 'bail', 'credit']:
           download_url(osp.join(self.url, f'{self.name}/{DATASET_MAP[self.name]["csv_name"]}_edges.txt'), self.data_dir)
        else:
           download_url(osp.join(self.url, f'{self.name}/{DATASET_MAP[self.name]["csv_name"]}_relationship.txt'), self.data_dir)  

    @staticmethod
    def available_datasets() -> List[str]:
        """
        Return all available datasets.
        """
        return list(DATASETS)

    def __repr__(self) -> str:
        return f'GraphFairness-{self.name.capitalize()}'


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)