from typing import List
import pickle, itertools
from numba import jit, i4, i8, f4, typeof
from numba.typed import List
from numba.experimental import jitclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from scipy.spatial import KDTree
import multiprocessing as mp

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
def get_state_dict(path):
    state_dict = torch.load(path)['state_dict']
    # remove prefix 'net.' and remake state dict
    return OrderedDict(
        {key[4:]: val for key, val in state_dict.items()})


def read_pickle(path):
    with open(path, mode='rb') as f:
        instance = pickle.load(f)
    return instance


def to_pickle(path, instance):
    with open(path, mode='wb') as f:
        pickle.dump(instance, f)


def generate_pivots(
    database: np.array, grid: int, k: int,
    margin=0., n_threads=1):
    """ generate pivots
    database: np.array
        database
    bins: int
        #grid of each dimension
    k: int
        calculate knn distance of pivots
    knnd_scaler: StandardScaler
        standard scaler fitted to knn distance
    margin: float
        margin of data space boundary
    n_threads: int
        number of threads to calculate kNN of pivots
        set s.t. grid ** dim % n_threads == 0

    returns: Pivots
    """
    dim = len(database[0])

    # calculate ranges
    min_values = database.min(axis=0) - margin
    max_values = database.max(axis=0) + margin

    # make grid
    hist, edges = np.histogramdd(
        database,
        bins=grid,
        range=list(zip(min_values, max_values)),
        density=True)
    
    # make pivots
    edges_wo_max = np.array(edges)[:, :-1]
    pivots = np.array(list(
        itertools.product(*edges_wo_max)))
    # add half of bin width into edges
    for i in range(dim):
        bin_width = (max_values[i] - min_values[i]) / grid
        pivots[:, i] += bin_width / 2

    # search knn of pivots
    index = KDTree(database)
    # knnd, _ = index.query(proxy_queries, k=k)

    batch_size = len(pivots) // n_threads
    with mp.Pool(n_threads) as pool:
        async_results = []
        for i in range(0, len(pivots), batch_size):
            proxy_query_batch = pivots[i:i+batch_size]
            async_results.append(pool.apply_async(
                index.query, (proxy_query_batch, k)))

        results = np.array(
            [async_res.get()[0] for async_res in async_results]) \
                .reshape(-1, k)

    pivots = pivots.astype('float32')
    knnd = results.astype('float32')

    return Pivots(
        dim, grid, min_values, max_values,
        pivots, knnd)


@jitclass
class Pivots:
    dim: i4  # dimension
    grid: i4  # #grid of each dimension
    min_values: f4[:]  # min values for each dimension
    max_values: f4[:]  # max values for each dimension
    bin_width: f4[:]  # bin width of histogram
    cell_diag: f4  # diagonal length of each cell
    pivots: f4[:, :]  # pivots
    knnd: f4[:, :]  # knn distance of pivots

    def __init__(self, dim: i4, grid: i4,
        min_values: f4[:], max_values: f4[:],
        pivots: f4[:, :], knnd: f4[:, :]) -> None:
        """
        dim: int32
            dimension of data
        grid: int32
            #grid of each dimension
        min_values: np.array[float32]
            min values for each dimension
        max_values: np.array[float32]
            max values for each dimension
        pivots: np.array[np.array[float32]]
            pivots
        knnd: np.array[np.array[float32]]
            knn distance of pivots
        """
        
        self.dim = dim
        self.grid = grid
        self.min_values = min_values
        self.max_values = max_values
        self.bin_width = (max_values - min_values) / grid
        self.cell_diag = np.sqrt((self.bin_width ** 2).sum())
        self.pivots = pivots
        self.knnd = knnd

    def calc_index(self, query: f4[:]) -> i4:
        """
        returns: int32
            pivot index of a given query in pivots array
        """
        index = 0
        for i in range(self.dim):
            index = index * self.grid + int(
                (query[i] - self.min_values[i]) \
                    / (self.max_values[i] - self.min_values[i]) \
                        * self.grid)
        return index

    def calc_indices(self, queries: f4[:, :]) -> i4[:]:
        """
        returns: np.array[int32]
            pivot indices of given queries in pivots array
        """
        indices = []
        for data in queries:
            index = self.calc_index(data)
            indices.append(index)
        return np.array(indices)
    
    def get_feature(self, query: f4[:]) -> f4[:]:
        """
        returns: np.array[float32]
            0:dim  query coordinates
            dim    normalized distance between query and nearest pivot
            dim+1: pivot's knn distances
        """
        index = self.calc_index(query)
        pivot = self.pivots[index]
        dist_from_pivot = np.linalg.norm(query - pivot)
        return np.concatenate((
            query,
            np.array([dist_from_pivot]) / self.cell_diag,
            self.knnd[index]
        ))

    def get_kth_feature(self, query: f4[:], k: i4) -> f4[:]:
        """
        returns: np.array[float32]
            0:dim  query coordinates
            dim    normalized distance between query and nearest pivot
            dim+1  pivot's k-th nn distance
        """
        index = self.calc_index(query)
        pivot = self.pivots[index]
        dist_from_pivot = np.linalg.norm(query - pivot)
        return np.concatenate((
            query,
            np.array([dist_from_pivot]) / self.cell_diag,
            np.array([self.knnd[index, k-1]])
        ))

    def get_features(self, queries: f4[:, :]) -> f4[:, :]:
        """
        returns: np.array[np.array[float32]]
            features of given queries
        """
        features = np.empty((
            len(queries), len(self.knnd[0])+self.dim+1))
        for i, query in enumerate(queries):
            features[i] = self.get_feature(query)
        return features

    def get_kth_features(self, queries: f4[:, :], ks: i4[:]) -> f4[:, :]:
        """
        returns: np.array[np.array[float32]]
            features of given queries
        """
        features = np.zeros((len(queries), self.dim+2))
        for i, (query, k) in enumerate(zip(queries, ks)):
            features[i] = self.get_kth_feature(query, k)
        return features

    def calc_knnd_upper_bound(self, query: f4[:]) -> f4[:]:
        """
        returns: float32
            upper bound of knn distances of query
        """
        index = self.calc_index(query)
        pivot = self.pivots[index]
        pivot_knnd = self.knnd[index]
        dist_from_pivot = np.linalg.norm(query - pivot)
        upper_bound = pivot_knnd + dist_from_pivot
        return upper_bound

    def calc_knnd_upper_bounds(self, queries: f4[:, :]) -> f4[:, :]:
        """
        returns: float32
            upper bounds of knn distances of queries
        """
        results = np.empty((len(queries), len(self.knnd[0])))
        for i, query in enumerate(queries):
            results[i] = self.calc_knn_upper_bound(query)
        return results


class PivNet(nn.Module):
    def __init__(self, n_units: List[int], pivots: Pivots,
            query_scaler: StandardScaler, knnd_scaler: StandardScaler):
        super().__init__()
        self.dim = pivots.dim
        self.k_max = n_units[-1]
        self.pivots = pivots
        self.query_scaler = query_scaler
        self.knnd_mean = knnd_scaler.mean_[:self.k_max]
        self.knnd_std = knnd_scaler.scale_[:self.k_max]
        self.fc = nn.Sequential(
            nn.Linear(n_units[0], n_units[1]),
            nn.ReLU(),
            nn.Linear(n_units[1], n_units[2]),
            nn.ReLU(),
            nn.Linear(n_units[2], n_units[3]),
            nn.ReLU(),
            nn.Linear(n_units[3], n_units[4]),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        feature = torch.from_numpy(
            self.pivots.get_features(x.numpy())).float()

        # scale query feature
        feature[:, :self.dim] = torch.from_numpy(
            self.query_scaler.transform(feature[:, :self.dim]))

        # scale pivot's knnd
        feature[:, self.dim+1:] = \
            (feature[:, self.dim+1:] - self.knnd_mean) / self.knnd_std

        pred = self.fc(feature)
        return pred

    def estimate(self, x: torch.tensor) -> torch.tensor:
        # returns: unscaled knn distances
        pred = self.forward(x)
        pred = pred * self.knnd_std + self.knnd_mean
        return pred


class PivNetItr(nn.Module):
    def __init__(self, dim: int, k_max: int, n_units: List[int],
            pivots: Pivots, query_scaler: StandardScaler,
            dist_mean: float, dist_std: float):
        super().__init__()
        self.dim = dim
        self.k_max = k_max
        self.k_mean = np.arange(1, k_max+1).mean()
        self.k_std = np.arange(1, k_max+1).std()
        self.pivots = pivots
        self.query_scaler = query_scaler
        self.dist_mean = dist_mean
        self.dist_std = dist_std
        self.fc = nn.Sequential(
            nn.Linear(n_units[0], n_units[1]),
            nn.ReLU(),
            nn.Linear(n_units[1], n_units[2]),
            nn.ReLU(),
            nn.Linear(n_units[2], n_units[3]),
            nn.ReLU(),
            nn.Linear(n_units[3], n_units[4]),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: torch.tensor
            each element consists of k and query
        """
        # scale k
        k = x[:, 0].numpy().astype('int32')
        k_scaled = (x[:, :1] - self.k_mean) / self.k_std
        
        queries = x[:, 1:]
        feature = torch.from_numpy(
            self.pivots.get_kth_features(queries.numpy(), k)).float()

        # scale query feature
        feature[:, :self.dim] = torch.from_numpy(
            self.query_scaler.transform(feature[:, :self.dim]))

        # scale pivot's k-th nnd
        feature[:, self.dim+1] = \
            (feature[:, self.dim+1] - self.dist_mean) / self.dist_std

        feature = torch.cat([k_scaled, feature], dim=1)
        pred = self.fc(feature)
        return pred

    def estimate(self, x):
        # returns: unscaled knn distance
        pred = self.forward(x)
        pred = pred * self.dist_std + self.dist_mean
        return pred
