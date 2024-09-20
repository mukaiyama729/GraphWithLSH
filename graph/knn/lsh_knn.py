from typing import TypeAlias, Union, List, Dict, Tuple
import numpy as np
import abc
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
from scipy.sparse import identity, csr_matrix, vstack, hstack
import rust_graph

from graph.lsh import LSH
from graph.arr_types import Matrix, Vec, IntVec
from graph.hash import IHashTable


class Similarity:
    COS = "cosine"
    EUC = "euclidean"
    JAC = "jaccard"
    PEA = "pearson"
    L1N = "l1"

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        if metric not in [self.COS, self.EUC, self.JAC, self.PEA, self.L1N]:
            raise ValueError(f"Invalid metric: {metric}")

    def similarity(self, vec1: Vec, vec2: Vec) -> float:
        if self.metric == self.COS:
            return self._cosine_similarity(vec1, vec2)
        elif self.metric == self.EUC:
            return -self._euclidean_distance(vec1, vec2)
        elif self.metric == self.JAC:
            return self._jaccard_similarity(vec1, vec2)
        elif self.metric == self.PEA:
            return self._pearson_correlation(vec1, vec2)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def _cosine_similarity(self, vec1: Vec, vec2: Vec) -> float:
        return np.dot(vec1, vec2)

    def _euclidean_distance(self, vec1: Vec, vec2: Vec) -> float:
        return np.linalg.norm(vec1 - vec2)

    def _jaccard_similarity(self, vec1: Vec, vec2: Vec) -> float:
        intersection = np.minimum(vec1, vec2).sum()
        union = np.maximum(vec1, vec2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def _pearson_correlation(self, vec1: Vec, vec2: Vec) -> float:
        return np.corrcoef(vec1, vec2)[0, 1]


class ILSHKNN(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, data: Matrix) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def add_data(self, data: Matrix) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def graph(self) -> Matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def set_hash_tables(self, tables: List[IHashTable]) -> None:
        raise NotImplementedError


class BaseLSHKNN(ILSHKNN):
    def __init__(self, L: int, delta: int, dim: int,  similarity: str = "cosine", threshold: float = 0.3, k: int = 5, max_candidates: int = 500):
        self.L = L
        self.delta = delta
        self.dim = dim
        self.similarity = Similarity(similarity)
        self.threshold = threshold
        self.k = k
        self.alpha = 5
        self.max_candidates = max_candidates
        self.ready = False

    def _process_bucket(self, bucket_indices: List[int]) -> None:

        bucket_vectors = self.data[bucket_indices]
        num_items = len(bucket_indices)

        if num_items > self.max_candidates:
            print('too mach data')
            sampled_indices = np.random.choice(num_items, self.max_candidates, replace=False)
            bucket_indices = [bucket_indices[i] for i in sampled_indices]
            bucket_vectors = bucket_vectors[sampled_indices]
            num_items = self.max_candidates

        # 類似度計算のベクトル化
        if self.similarity.metric == "cosine":
            # データの正規化は事前に行われていると仮定
            sims = rust_graph.DistanceClass().cosine_similarity_matrix_simd(bucket_vectors.tolist())
        elif self.similarity.metric == "euclidean":
            sims = rust_graph.DistanceClass().euclidean_distance_matrix_simd(bucket_vectors.tolist())
        elif self.similarity.metric == "l1":
            diffs = bucket_vectors[:, np.newaxis, :] - bucket_vectors[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2, ord=1)
            sims = dists
        else:
            sims = np.zeros((num_items, num_items))
            for idx_i in range(num_items):
                for idx_j in range(idx_i + 1, num_items):
                    sim = self.similarity.similarity(bucket_vectors[idx_i], bucket_vectors[idx_j])
                    sims[idx_i, idx_j] = sim
                    sims[idx_j, idx_i] = sim

        for idx in range(num_items):
            i = bucket_indices[idx]
            sim_row = sims[idx]

            sim_row[idx] = 0

            top_k = min(self.k, num_items - 1)
            if top_k <= 0:
                continue

            neighbor_indices = np.argpartition(sim_row, range(top_k))[:top_k]
            neighbor_indices[0] = idx

            for neighbor_idx in neighbor_indices:
                j = bucket_indices[neighbor_idx]
                sim = sim_row[neighbor_idx]
                self.distance_matrix[i][j] = sim
                self.knn_pairs[i].add(j)

    def _create_knn_dist(self):
        indices = np.arange(self.data.shape[0])
        for i in tqdm(indices):
            dists = np.array(list(self.distance_matrix[i].values()))
            if len(dists) > self.k:
                knn_dists = np.partition(np.array(list(self.distance_matrix[i].values())), range(self.k))[:self.k]
                knn_dist = knn_dists[-1]
            else:
                knn_dist = dists.max()
            self.k_nn_dist[i] = knn_dist if knn_dist > 0 else 1e-3

    def _update_knn_dist(self, updating_indices: set):
        for i in tqdm(updating_indices):
            dists = np.array(list(self.distance_matrix[i].values()))
            if len(dists) > self.k:
                knn_dist = np.partition(np.array(list(self.distance_matrix[i].values())), range(self.k))[:self.k][-1]
                if i  < self.num_of_data and knn_dist == self.k_nn_dist[i]:
                    continue
                yield i
            else:
                knn_dist = dists.max()
            self.k_nn_dist[i] = knn_dist if knn_dist > 0  else 1e-3

    def _update_knn_graph(self, updating_indices: set):
        for i in tqdm(updating_indices):
            pairs_list = list(self.distance_matrix[i].keys())
            dists_of_pairs = np.array(list(self.distance_matrix[i].values()))
            k_nearest_dists_of_pairs = np.array(list(self.k_nn_dist.values()))[pairs_list]
            self.knn_graph[i, pairs_list] = self._apply_kernel(pairs_list.index(i), pairs_list, dists_of_pairs, k_nearest_dists_of_pairs)

    def _create_knn_graph(self):
        k_nearest_dists = np.array(list(self.k_nn_dist.values()))

        row_indices = []
        col_indices = []
        data_values = []

        for i, dist_dict in tqdm(self.distance_matrix.items()):
            pairs_list = np.array(list(dist_dict.keys()))
            dists_of_pairs = np.array(list(dist_dict.values()))
            k_nearest_dists_of_pairs = k_nearest_dists[pairs_list]

            kernel_values = self._apply_kernel(np.where(pairs_list == i)[0][0], pairs_list, dists_of_pairs, k_nearest_dists_of_pairs)

            row_indices.extend([i] * len(pairs_list))
            col_indices.extend(pairs_list)
            data_values.extend(kernel_values)

        self.knn_graph = csr_matrix((data_values, (row_indices, col_indices)), shape=self.knn_graph.shape)

    def _apply_kernel(self, center_index: int, pairs_list: List[int], dists_of_pairs: Vec, k_nearest_dists: Vec, kernel='rust') -> Vec:
        if kernel == 'alpha_decay':
            from_centers = -(dists_of_pairs / (k_nearest_dists[center_index] + 1e-6)) ** self.alpha
            from_eaches = -(dists_of_pairs / (k_nearest_dists + 1e-6)) ** self.alpha
            return 1/2 * np.exp(from_centers) + 1/2 * from_eaches
        elif kernel == 'rust':
            return rust_graph.alpha_decay(list(dists_of_pairs), k_nearest_dists[center_index], list(k_nearest_dists), self.alpha)

    def _update_mean_std(self, data: Matrix) -> None:
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def _initial_setup(self, data: Matrix) -> None:
        self.num_of_data = data.shape[0]
        self.distance_matrix = defaultdict(dict)
        self.knn_graph = csr_matrix((data.shape[0], data.shape[0]))
        self.k_nn_dist = {}
        self.knn_pairs = defaultdict(set)
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.data = (data - self.mean) / self.std

    def set_hash_tables(self, tables: List[IHashTable]) -> None:
        self.lsh = LSH(self.L, self.delta, self.dim)
        self.lsh.set_tables(tables)
        self.ready = True

    def fit(self, data: Matrix) -> None:

        if not self.ready:
            raise ValueError("Hash tables are not set")

        self._initial_setup(data)
        self.lsh.create_groups(self.data)

        for hash_index, buckets in self.lsh.hash_groups.items():
            for bucket_index, bucket_indices in tqdm(buckets.items(), desc=f"Processing hash table {hash_index+1}/{self.L}"):
                if len(bucket_indices) == 0:
                    continue
                self._process_bucket(bucket_indices)
        self._create_knn_dist()
        self._create_knn_graph()

    def add_data(self, data: Matrix) -> None:
        data = (data - self.mean) / self.std
        n = data.shape[0]
        self.data = np.vstack([self.data, data])
        self.knn_graph = self.expand_csr_matrix(self.knn_graph, n)
        updated_hash_groups = self.lsh.add_groups(data)
        updating_indices = set()

        for hash_index, buckets in updated_hash_groups.items():
            for bucket_index, bucket_indices in tqdm(buckets.items(), desc=f"Processing hash table {hash_index+1}/{self.L}"):
                if len(bucket_indices) == 0:
                    continue
                self._process_bucket(bucket_indices)
                updating_indices = updating_indices.union(set(bucket_indices))

        updated_indices = set(self._update_knn_dist(updating_indices=updating_indices))
        self._update_knn_graph(updating_indices=updated_indices)
        self._update_mean_std(data)
        self.num_of_data += n

    def expand_csr_matrix(self, matrix: csr_matrix, n: int) -> csr_matrix:

        zero_rows = csr_matrix((n, matrix.shape[1]))
        expanded_matrix = vstack([matrix, zero_rows])

        zero_cols = csr_matrix((expanded_matrix.shape[0], n))
        expanded_matrix = hstack([expanded_matrix, zero_cols])

        return expanded_matrix

    def graph(self) -> Matrix:
        return self.knn_graph.tocsr()


class LSHKNN(BaseLSHKNN):
    def __init__(self, L: int, delta: int, dim: int, similarity: str = "cosine", threshold: float = 0.3, k: int = 5, max_candidates: int = 500):
        super().__init__(L, delta, dim, similarity, threshold, k, max_candidates)

    def compute_eigenvector_centrality(self, tol: float = 1e-7, max_iter: int = 1000) -> Vec:

        def spectral_radius(M, approximation=True):
            """
            Compute the spectral radius of M.
            """
            if approximation:
                eigenvalue, _ = eigsh(M, k=1, which='LM')
                max_abs_eigenvalue = np.abs(eigenvalue[0])
                return max_abs_eigenvalue
            return np.max(np.abs(np.linalg.eigvals(M)))

        A = self.knn_graph.tocsr()
        n = A.shape[0]
        np.random.seed(42)
        b_k = np.random.rand(n)
        b_k = b_k / np.linalg.norm(b_k)
        r = spectral_radius(A, approximation=True)
        for it in range(max_iter):
            b_k1 = r**(-1) * A.dot(b_k)
            b_k1 = b_k1 / np.linalg.norm(b_k1)
            if np.linalg.norm(b_k1 - b_k) < tol:
                break
            b_k = b_k1
        print(f'Converged in {it+1} iterations')

        return b_k

    def compute_prob_matrix(self) -> csr_matrix:
        A = self.knn_graph.tocsr()
        degrees = A.sum(axis=1).A1
        degrees[degrees == 0] = 1  # Prevent division by zero
        D_inv = csr_matrix((1.0 / degrees, (np.arange(A.shape[0]), np.arange(A.shape[0]))), shape=A.shape)
        P = D_inv.dot(A).tocsr()

        return P

    def compute_prob_eigenvectors(self, k=128, which='LM') -> Tuple[Vec, Matrix]:
        P = self.compute_prob_matrix()
        eigenvalues, eigenvectors = eigsh(P, k=k, which=which)
        return (eigenvalues, eigenvectors)

    def compute_laplacian_eigenvectors(self, k=128, which='LM') -> Tuple[Vec, Matrix]:
        P = self.compute_prob_matrix()
        L = identity(P.shape[0]) - P
        eigenvalues, eigenvectors = eigsh(L, k=k, which=which)
        return (eigenvalues, eigenvectors)
