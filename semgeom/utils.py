from typing import List, Dict, Iterable
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale(values: Iterable[float], feature_range=(-1.0, 1.0)) -> np.ndarray:
    """
    Масштабирует 1D список/массив значений в диапазон feature_range.
    Возвращает numpy array той же длины.
    """
    arr = np.array(list(values), dtype=float).reshape(-1, 1)
    if arr.size == 0:
        return np.array([])
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(arr).ravel()

def cos_sim(a, b) -> float:
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)