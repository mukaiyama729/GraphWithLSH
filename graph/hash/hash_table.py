from typing import List, Tuple, TypeVar
import numpy as np
import abc

from graph.arr_types import Vec, Matrix, IntVec

_HashTable = TypeVar("_HashTable", bound="HashTable")

class IHashTable:
    @abc.abstractmethod
    def hashes(self, matrix: Matrix) -> IntVec:
        raise NotImplementedError

    @abc.abstractmethod
    def hash(self, vec: Vec) -> int:
        raise NotImplementedError


class BaseHashTable:
    def __init__(self, delta: int, dim: int):
        self.delta = delta
        self.dim = dim
        self.table = np.random.randn(dim, delta)

    def calc_hash_v(self, vec: Vec) -> int:
        binary_arr = np.where(np.dot(vec, self.table) > 0, 1, 0)
        return int("".join(map(str, binary_arr)), 2)

    def calc_hash_m(self, matrix: Matrix) -> IntVec:
        # ベクトルのドット積結果が0以上かで判定し、ビットを生成
        binary_matrix = np.dot(matrix, self.table) > 0
        # 2進数から整数への変換
        binaries = np.arange(self.delta)
        return np.dot(binary_matrix, 1 << binaries)


class HashTable(BaseHashTable, IHashTable):
    def __init__(self, delta: int, dim: int):
        super().__init__(delta, dim)

    def hashes(self, matrix: Matrix) -> IntVec:
        return self.calc_hash_m(matrix)

    def hash(self, vec: Vec) -> int:
        return self.calc_hash_v(vec)

    def tables_list(self, num_tables: int) -> List[_HashTable]:
        return [HashTable(self.delta, self.dim) for _ in range(num_tables)]
