from typing import List, Dict, Any
import numpy as np
from .utils import scale
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def feature_direction(feature_words: Dict[str, List[str]], model) -> np.ndarray:
    """
    Построить направление признака по спискам полюсов.
    feature_words: {"pos": [...], "neg": [...]}
    model: объект с методом encode(text)->np.ndarray
    Возвращает нормализованный direction (numpy array).
    """
    pos_vecs = [model.encode(w) for w in feature_words["pos"]]
    neg_vecs = [model.encode(w) for w in feature_words["neg"]]
    directions = [p - n for p in pos_vecs for n in neg_vecs]
    dir_mean = np.mean(directions, axis=0)
    norm = np.linalg.norm(dir_mean)
    if norm == 0:
        return dir_mean
    return dir_mean / norm


def project_words(words: List[str], direction: np.ndarray, model, feature_range=(-1, 1)) -> Dict[str, float]:
    """
    Проецирует список слов на direction (предполагается нормализован).
    Возвращает словарь {word: scaled_value} — значения масштабированы через MinMax в feature_range.
    """
    word_vectors = {w: model.encode(w) for w in words}
    raw = {w: float(np.dot(vec, direction)) for w, vec in word_vectors.items()}
    scaled_vals = scale(list(raw.values()), feature_range=feature_range)
    return {w: float(scaled_vals[i]) for i, w in enumerate(raw.keys())}


def get_scalar_projections(words: List[str], feature_words: Dict[str, List[str]], model) -> Dict[str, float]:
    """
    Как в твоей реализации: scalar projections с центрированием по средним pos/neg.
    Возвращает словарь {word: scalar} НЕ масштабированные (raw).
    """
    word_vecs = np.array([model.encode(w) for w in words])

    pos_vecs = np.array([model.encode(w) for w in feature_words['pos']])
    neg_vecs = np.array([model.encode(w) for w in feature_words['neg']])

    direction = np.mean(pos_vecs[:, None, :] - neg_vecs[None, :, :], axis=(0, 1))
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    center = (pos_vecs.mean(axis=0) + neg_vecs.mean(axis=0)) / 2.0

    projections = {}
    for w, vec in zip(words, word_vecs):
        relative = vec - center
        scalar = float(np.dot(relative, direction))
        projections[w] = scalar

    # сортируем от малого к большому (удобно для отображения)
    return dict(sorted(projections.items(), key=lambda x: x[1]))
