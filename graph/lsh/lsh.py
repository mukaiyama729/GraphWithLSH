from typing import List, Dict, Tuple
from collections import defaultdict

from graph.hash.hash_table import HashTable, IHashTable
from graph.arr_types import Matrix, Vec, IntVec

class LSH:
    def __init__(self, L: int, delta: int, dim: int):
        self.L = L
        self.delta = delta
        self.dim = dim
        self.hash_tables = [HashTable(delta, dim) for _ in range(L)]
        self.hash_groups: Dict[int, Dict[int, List[int]]] = {i: {} for i in range(L)}
        self.num_data = 0

    def set_tables(self, tables: List[IHashTable]) -> None:
        self.hash_tables = tables
        self.L = len(tables)
        self.hash_groups = {i: {} for i in range(self.L)}

    def create_groups(self, matrix: Matrix) -> None:
        for i, table in enumerate(self.hash_tables):
            buckets = self._table_buckets(table, matrix)
            self.hash_groups[i] = buckets
        self.num_data = matrix.shape[0]

    def add_groups(self, matrix: Matrix) -> Dict[int, Dict[int, List[int]]]:
        updated_hash_groups: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        for i, table in enumerate(self.hash_tables):
            buckets = self._table_buckets(table, matrix)
            for key, bucket in buckets.items():
                self.hash_groups[i][key] = self.hash_groups[i].get(key, []) + bucket
                updated_hash_groups[i][key] = self.hash_groups[i][key]
        self.num_data += matrix.shape[0]
        return updated_hash_groups

    def _table_buckets(self, table: HashTable, matrix: Matrix) -> Dict[int, List[int]]:
        bucket_indices = table.hashes(matrix)
        buckets = defaultdict(list)
        for idx, bucket_index in enumerate(bucket_indices):
            idx += self.num_data
            buckets[bucket_index].append(idx)
        return buckets
