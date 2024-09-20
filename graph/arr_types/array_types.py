import abc
import numpy as np
from numpy.typing import NDArray

# 型エイリアスを使用して行列とテンソルを定義
Vec = NDArray[np.float64]
IntVec = NDArray[np.int64]
Matrix = NDArray[np.float64]  # 2次元配列
Tensor = NDArray[np.float64]

__all__ = ["Vec", "IntVec", "Matrix", "Tensor"]
