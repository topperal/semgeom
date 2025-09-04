# semgeom/pairs.py
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# NOTE: модель не создаётся автоматически в модуле.
# Перед использованием вызовите semgeom.set_model(your_model) или присвойте semgeom.model = your_model
model = None  # will be set by user via set_model()

# -------------------------
# Embedding cache helper
# -------------------------
_EMB_CACHE: Dict[str, np.ndarray] = {}

def set_model(m):
    """
    Установить глобальную модель (SentenceTransformer или аналог).
    Вызывать перед использованием функций, требующих кодирования.
    """
    global model, _EMB_CACHE
    model = m
    _EMB_CACHE = {}  # сброс кеша при смене модели

def _ensure_model():
    if model is None:
        raise RuntimeError(
            "Модель не установлена. Вызовите semgeom.set_model(your_model)\n"
            "например: from sentence_transformers import SentenceTransformer\n"
            "         semgeom.set_model(SentenceTransformer('all-MiniLM-L6-v2'))"
        )

def embed_cached(text: str) -> np.ndarray:
    """
    Возвращает embedding для текста, кешируя результат в памяти.
    Требует, чтобы semgeom.model был установлен (через set_model).
    """
    _ensure_model()
    if text not in _EMB_CACHE:
        _EMB_CACHE[text] = model.encode(text)
    return _EMB_CACHE[text]

# -------------------------
# Geometry & similarity helpers
# -------------------------
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство (устойчиво к малым нормам)."""
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def angular_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Угол между векторами в градусах."""
    cs = cos_sim(a, b)
    cs = max(min(cs, 1.0), -1.0)
    return float(math.degrees(math.acos(cs)))

def rel_euclid(a: np.ndarray, b: np.ndarray) -> float:
    """Относительное евклидово расстояние: ||a-b|| / ||a|| (устойчиво)."""
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))

# -------------------------
# Feature direction helper (локальная реализация)
# -------------------------
def feature_direction(feature_words: Dict[str, List[str]]) -> np.ndarray:
    """
    Построить direction vector по полю feature_words {'pos': [...], 'neg': [...]}
    Возвращает нормализованный вектор (unit vector).
    """
    _ensure_model()
    pos_vecs = np.array([model.encode(w) for w in feature_words["pos"]])
    neg_vecs = np.array([model.encode(w) for w in feature_words["neg"]])
    direction = pos_vecs.mean(axis=0) - neg_vecs.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm

# -------------------------
# Internal normalizer for scale values
# -------------------------
def _normalize_direction_from_scaleval(scale_val: Any) -> np.ndarray:
    """
    Принятие scale_val в разных форматах и возврат нормализованного вектора:
      - если объект имеет атрибут .direction — берём его,
      - если это dict {'pos':..., 'neg':...} — считаем через feature_direction,
      - если numpy array/list — нормализуем и возвращаем.
    """
    if hasattr(scale_val, "direction"):
        d = np.array(scale_val.direction, dtype=float)
    elif isinstance(scale_val, dict):
        d = np.array(feature_direction(scale_val), dtype=float)
    else:
        d = np.array(scale_val, dtype=float)
    norm = np.linalg.norm(d)
    return d / norm if norm != 0 else d

# -------------------------
# Main evaluator
# -------------------------
def evaluate_pairs_on_scales(
    pairs: List[Tuple[str, str]],
    scales_dict: Dict[str, Any],
    include_scaled: bool = False
) -> pd.DataFrame:
    """
    Оценивает пары (orig, para) по наборам смысловых шкал.
    Возвращает DataFrame с колонками:
      orig, para, cos_sim, angle_deg, rel_euclid, и для каждой шкалы:
      <scale>_proj_delta_raw  (и опционально <scale>_proj_delta_scaled).
    """
    _ensure_model()

    # 1) уникальные тексты и их эмбеддинги (кешируем)
    unique_texts = []
    for o, p in pairs:
        if o not in unique_texts: unique_texts.append(o)
        if p not in unique_texts: unique_texts.append(p)
    emb_map = {t: embed_cached(t) for t in unique_texts}

    # 2) подготовить нормализованные направления
    directions = {sname: _normalize_direction_from_scaleval(sval) for sname, sval in scales_dict.items()}

    # 3) вычисление метрик по парам
    rows = []
    for orig, para in pairs:
        u = emb_map[orig]; v = emb_map[para]
        row = {
            "orig": orig,
            "para": para,
            "cos_sim": cos_sim(u, v),
            "angle_deg": angular_deg(u, v),
            "rel_euclid": rel_euclid(u, v)
        }
        for sname, dnorm in directions.items():
            p_orig_raw = float(np.dot(u, dnorm))
            p_para_raw = float(np.dot(v, dnorm))
            row[f"{sname}_proj_delta_raw"] = float(p_para_raw - p_orig_raw)
        rows.append(row)
    df = pd.DataFrame(rows)

    # 4) опционально: добавить scaled дельты (min-max по всем уникальным текстам)
    if include_scaled:
        for sname, dnorm in directions.items():
            all_raw = np.array([float(np.dot(emb_map[t], dnorm)) for t in unique_texts])
            if np.allclose(all_raw.max(), all_raw.min()):
                all_scaled = np.zeros_like(all_raw)
            else:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                all_scaled = scaler.fit_transform(all_raw.reshape(-1, 1)).ravel()
            scaled_map = {t: float(all_scaled[i]) for i, t in enumerate(unique_texts)}
            df[f"{sname}_proj_delta_scaled"] = df.apply(
                lambda r: float(scaled_map[r["para"]] - scaled_map[r["orig"]]),
                axis=1
            )

    # 5) порядок колонок
    base_cols = ["orig", "para", "cos_sim", "angle_deg", "rel_euclid"]
    raw_delta_cols = [f"{s}_proj_delta_raw" for s in scales_dict.keys()]
    cols = base_cols + raw_delta_cols
    if include_scaled:
        cols += [f"{s}_proj_delta_scaled" for s in scales_dict.keys()]
    df = df[cols]
    return df

# -------------------------
# Plot helper for pairs
# -------------------------
def plot_pair_on_two_scales(df_pairs: pd.DataFrame, scales_dict: Dict[str, Any],
                            scale_x: str, scale_y: str, title: Optional[str] = None):
    """
    Отрисовать пары в 2D-пространстве двух шкал (scale_x, scale_y).
    Параметры и поведение описаны в документации (см. README).
    """
    _ensure_model()

    texts = list(pd.unique(df_pairs[["orig", "para"]].values.ravel()))

    def _dir_for_scale(sval):
        return _normalize_direction_from_scaleval(sval)

    def compute_scaled_map(scale_name):
        sval = scales_dict[scale_name]
        dnorm = _dir_for_scale(sval)
        emb = np.array([embed_cached(t) for t in texts])
        all_raw = emb.dot(dnorm)
        if np.allclose(all_raw.max(), all_raw.min()):
            all_scaled = np.zeros_like(all_raw)
        else:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            all_scaled = scaler.fit_transform(all_raw.reshape(-1, 1)).ravel()
        return {t: float(all_scaled[i]) for i, t in enumerate(texts)}

    map_x = compute_scaled_map(scale_x)
    map_y = compute_scaled_map(scale_y)

    fig, ax = plt.subplots(figsize=(9, 9))
    # try new Matplotlib colormap API, fallback to cm.get_cmap
    try:
        cmap = mpl.colormaps.get_cmap("coolwarm")
    except Exception:
        cmap = mpl.cm.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    for _, row in df_pairs.iterrows():
        o = row["orig"]; p = row["para"]
        x0 = map_x[o]; y0 = map_y[o]
        x1 = map_x[p]; y1 = map_y[p]
        cs = float(row["cos_sim"])
        ax.arrow(x0, y0, x1 - x0, y1 - y0, color=cmap(norm(cs)), alpha=0.9,
                 width=0.002, head_width=0.03, length_includes_head=True)
        ax.scatter([x0], [y0], color=cmap(norm(cs)), edgecolor='k', s=40)
        ax.text(x0, y0, o, fontsize=8, alpha=0.8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="cos_sim (orig vs para)")

    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel(f"{scale_x} (scaled)")
    ax.set_ylabel(f"{scale_y} (scaled)")
    ax.set_title(title or f"{scale_x} vs {scale_y}")
    plt.tight_layout()
    plt.show()
