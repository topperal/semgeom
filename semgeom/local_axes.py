# semgeom/local_axes.py
"""
Local semantic axes: выбор полюсов (pos/neg) внутри одного документа/предложения
без POS-тегов и построение локальных осей.

Экспортирует:
 - pick_pos_neg_PCA
 - pick_pos_neg_max_distance
 - pick_pos_neg_center_extremes
 - pick_pos_neg_kmeans
 - plot_local_axes_strategies
 - auto_axis_from_sentence
 - compare_word_on_axes
 - plot_axes_scatter

Все функции работают с глобальной моделью, установленной через semgeom.set_model(...)
и используют embed_cached(...) для кодирования.
"""

from typing import List, Dict, Tuple, Any
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

# импортируем утилиты из pairs (предполагается что pairs.py уже добавлен в пакет)
from .pairs import embed_cached, feature_direction, set_model, cos_sim  # type: ignore
import pandas as pd

# ---- небольшой локальный helper: токенизация ----
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Простая токенизация на слова (алфавитно-цифровые последовательности)."""
    return _WORD_RE.findall(text)


def _encode_words(words: List[str]) -> np.ndarray:
    """Векторизуем список слов через embed_cached (возвращаем (N, D) массив)."""
    return np.vstack([embed_cached(w) for w in words])


def _scale_dict(vals: List[float], feature_range=(-1, 1)) -> Dict[str, float]:
    """Масштабируем список значений в заданный диапазон и возвращаем mapping index->scaled."""
    arr = np.array(vals).reshape(-1, 1)
    if np.allclose(arr.max(), arr.min()):
        scaled = np.zeros_like(arr).ravel()
    else:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled = scaler.fit_transform(arr).ravel()
    return dict(enumerate(scaled))


# -----------------------------
# Стратегии выбора pos / neg
# -----------------------------
def pick_pos_neg_PCA(words: List[str]) -> Dict[str, List[str]]:
    """
    Стратегия: 1-я компонента PCA на embedding'ах слов.
    Возвращает {'pos':[word_pos], 'neg':[word_neg]} - extremes по PCA component.
    """
    vecs = _encode_words(words)
    pca = PCA(n_components=1, random_state=0)
    comps = pca.fit_transform(vecs).ravel()
    pos_word = words[int(np.argmax(comps))]
    neg_word = words[int(np.argmin(comps))]
    return {"pos": [pos_word], "neg": [neg_word]}


def pick_pos_neg_max_distance(words: List[str]) -> Dict[str, List[str]]:
    """
    Стратегия: выбрать пару слов с максимальным попарным косинусным (евклидовым) расстоянием.
    """
    vecs = _encode_words(words)
    dists = cosine_distances(vecs)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return {"pos": [words[int(i)]], "neg": [words[int(j)]]}


def pick_pos_neg_center_extremes(words: List[str]) -> Dict[str, List[str]]:
    """
    Стратегия: взять слово максимально удалённое от центра (pos),
    и слово, максимально противоположное по направлению (neg).
    """
    vecs = _encode_words(words)
    center = vecs.mean(axis=0)
    rel = vecs - center
    norms = np.linalg.norm(rel, axis=1)
    pos_idx = int(np.argmax(norms))
    # берём проекции всех rel на rel[pos_idx], выбираем минимальную (наиболее противоположную)
    ref = rel[pos_idx]
    projs = rel.dot(ref)
    neg_idx = int(np.argmin(projs))
    return {"pos": [words[pos_idx]], "neg": [words[neg_idx]]}


def pick_pos_neg_kmeans(words: List[str], n_init: int = 10) -> Dict[str, List[str]]:
    """
    Стратегия: кластеризация KMeans (k=2), выбираем центры кластеров и ближайшие слова.
    Возвращает ближайшие слова к центрам как полюса.
    """
    vecs = _encode_words(words)
    if vecs.shape[0] < 2:
        return {"pos": [words[0]], "neg": [words[0]]}
    km = KMeans(n_clusters=2, n_init=n_init, random_state=0).fit(vecs)
    centers = km.cluster_centers_
    # nearest-to-center indices
    d0 = cosine_distances(vecs, centers[[0]]).ravel()
    d1 = cosine_distances(vecs, centers[[1]]).ravel()
    c0_idx = int(np.argmin(d0))
    c1_idx = int(np.argmin(d1))
    return {"pos": [words[c0_idx]], "neg": [words[c1_idx]]}


# -----------------------------
# Визуализация: сравнить стратегии на предложении
# -----------------------------
def plot_local_axes_strategies(sentence: str, strategies: dict = None, top_n: int = None):
    """
    Для заданного sentence строит несколько локальных осей (разными стратегиями)
    и рисует 1D scatter для каждой стратегии (слова по проекции на локальную ось).

    - sentence: строка
    - strategies: опциональный словарь вида {name: function(words)->{'pos':[...],'neg':[...]} }
      (если None используется набор по умолчанию)
    - top_n: если задано, отображать только top_n слов по абсолютной проекции (по модулю)
    """
    if strategies is None:
        strategies = {
            "PCA axis": pick_pos_neg_PCA,
            "Max-distance": pick_pos_neg_max_distance,
            "Center-extremes": pick_pos_neg_center_extremes,
            "KMeans clusters": pick_pos_neg_kmeans
        }

    words = _tokenize(sentence)
    if len(words) == 0:
        raise ValueError("Пустое предложение/нет токенов.")

    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, func) in zip(axes, strategies.items()):
        feat = func(words)
        # compute direction via feature_direction imported from pairs
        dir_vec = feature_direction(feat)
        # compute raw projections
        vecs = _encode_words(words)
        raw_proj = vecs.dot(dir_vec)
        # optionally take top_n by abs value
        idxs = np.arange(len(words))
        if top_n is not None and top_n < len(words):
            idxs = np.argsort(np.abs(raw_proj))[-top_n:]
        # scale to [-1,1] for plotting
        scaled_vals = MinMaxScaler(feature_range=(-1, 1)).fit_transform(raw_proj.reshape(-1, 1)).ravel()
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        for i in idxs:
            w = words[int(i)]; val = float(scaled_vals[int(i)])
            is_pole = w in (feat.get("pos", []) + feat.get("neg", []))
            ax.scatter(0, val, color="red" if is_pole else "blue", s=40)
            ax.text(0.05, val, w, fontsize=9, va="center")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Авто-полюса из предложения (вариант)
# -----------------------------
def auto_axis_from_sentence(sentence: str, pole1_name: str = "pos", pole2_name: str = "neg") -> Dict[str, Any]:
    """
    Простая стратегия: выбрать слово максимально удалённое от центра как pos,
    и минимально удалённое как neg. Возвращает словарь с 'pos' и 'neg' и именами полюсов.
    """
    words = _tokenize(sentence)
    if not words:
        return {"pos": [], "neg": [], "pole1": pole1_name, "pole2": pole2_name}
    vecs = _encode_words(words)
    center = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - center, axis=1)
    pos = words[int(np.argmax(dists))]
    neg = words[int(np.argmin(dists))]
    return {"pos": [pos], "neg": [neg], "pole1": pole1_name, "pole2": pole2_name}


# -----------------------------
# Сравнительная таблица и scatter
# -----------------------------
def compare_word_on_axes(sentence: str, axes_defs: Dict[str, Dict[str, Any]]) -> Any:
    """
    Для sentence и набора axes_defs (axis_name -> axis_def):
      axis_def: должен содержать {'pos':[...], 'neg':[...], 'pole1':name, 'pole2':name} или
                просто {'pos':[...], 'neg':[...]} (тогда pole1/pole2 берутся как pos/neg).
    Возвращает pandas.DataFrame с колонками: word, <axis1>, <axis1>_label, ...
    """
    import pandas as pd
    words = _tokenize(sentence)
    df = pd.DataFrame({"word": words})
    for axis_name, axis_def in axes_defs.items():
        # ensure pos/neg available
        if not isinstance(axis_def, dict) or "pos" not in axis_def or "neg" not in axis_def:
            raise ValueError(f"axis_def for '{axis_name}' must contain 'pos' and 'neg' lists")
        direction = feature_direction(axis_def)
        vecs = _encode_words(words)
        proj = vecs.dot(direction)
        # scaled for readability
        if len(proj) > 1:
            proj_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(proj.reshape(-1, 1)).ravel()
        else:
            proj_scaled = proj
        df[axis_name] = proj_scaled
        pole1 = axis_def.get("pole1", axis_def.get("pos", ["POS"])[0])
        pole2 = axis_def.get("pole2", axis_def.get("neg", ["NEG"])[0])
        df[f"{axis_name}_label"] = df[axis_name].apply(lambda x: pole1 if float(x) < 0 else pole2)
    return df


def plot_axes_scatter(df, axis_name: str, color_map: dict = None):
    """
    Нарисовать одномерный scatter для axis_name в dataframe (как в compare_word_on_axes).
    color_map: mapping label->color (опционально).
    """
    if color_map is None:
        color_map = {}
    plt.figure(figsize=(6, 4))
    pole_label_col = f"{axis_name}_label"
    colors = [color_map.get(lbl, 'gray') for lbl in df[pole_label_col]]
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.scatter(df[axis_name], range(len(df)), c=colors, s=80)
    for i, row in df.iterrows():
        plt.text(row[axis_name], i, row["word"], fontsize=9, va="center", ha="left")
    plt.xlabel(axis_name)
    plt.yticks([])
    plt.title(f"Axis: {axis_name}")
    plt.show()

def build_direction_from_examples(pos_examples: List[str], neg_examples: List[str]) -> np.ndarray:
    """
    Построить направление признака на основе примеров:
      direction = mean(emb(pos_examples)) - mean(emb(neg_examples))
    Возвращает нормализованный вектор (numpy array).
    """
    pos_vecs = np.vstack([embed_cached(w) for w in pos_examples])
    neg_vecs = np.vstack([embed_cached(w) for w in neg_examples])
    dir_vec = pos_vecs.mean(axis=0) - neg_vecs.mean(axis=0)
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        return dir_vec
    return dir_vec / norm


def build_local_axis_pca(tokens: List[str]) -> np.ndarray:
    """
    Локальная ось: первая компонентa PCA на embedding'ах токенов предложения.
    Возвращает нормализованный вектор направления.
    """
    if len(tokens) == 0:
        raise ValueError("Пустой список токенов.")
    embs = np.vstack([embed_cached(t) for t in tokens])
    if embs.shape[0] < 2:
        # не хватает слов для PCA — вернём нулевой вектор
        vec = embs.mean(axis=0)
        return vec / (np.linalg.norm(vec) + 1e-12)
    pca = PCA(n_components=1, random_state=0)
    pca.fit(embs)
    direction = pca.components_[0]
    norm = np.linalg.norm(direction)
    return direction / (norm + 1e-12)


def build_importance_axis(tokens: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Построение оси "важное ↔ второстепенное" внутри предложения.
    Идея: важность слова = 1 - cos_sim(word_vec, sent_mean_vec)
    Возвращает tuple(direction_vector, importance_raw_array)
      - direction_vector: вектор из 'least important' в 'most important' (нормализованный)
      - importance_raw_array: array(len(tokens),) значений важности
    """
    if len(tokens) == 0:
        raise ValueError("Пустой список токенов.")
    embs = np.vstack([embed_cached(t) for t in tokens])
    sent_vec = embs.mean(axis=0)
    # importance: чем ближе вектор к sent_vec — тем меньше важность; берем 1 - cos_sim
    importance = np.array([1.0 - cos_sim(e, sent_vec) for e in embs])
    # direction: от наименее важного к наиболее важному (max - min)
    idx_max = int(np.argmax(importance))
    idx_min = int(np.argmin(importance))
    direction = embs[idx_max] - embs[idx_min]
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction, importance
    return direction / norm, importance


def analyze_sentence_local_axes(sentence: str) -> pd.DataFrame:
    """
    Для одного предложения возвращает DataFrame с колонками:
      token, proj_pca (scaled), proj_importance (scaled), importance_raw

    Оси:
     - локальная PCA ось (примерно "noun↔verb" / главная семантическая ось)
     - importance axis (important vs minor) построенная локально
    """
    tokens = _tokenize(sentence)
    if len(tokens) == 0:
        return pd.DataFrame(columns=["token", "proj_pca", "proj_imp", "importance_raw"])

    # PCA axis
    axis_pca = build_local_axis_pca(tokens)
    embs = np.vstack([embed_cached(t) for t in tokens])
    proj_pca_raw = embs.dot(axis_pca)

    # importance axis
    axis_imp, importance_raw = build_importance_axis(tokens)
    proj_imp_raw = embs.dot(axis_imp)

    # scale both to [-1,1] for display
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if len(proj_pca_raw) > 1 and not np.allclose(proj_pca_raw.max(), proj_pca_raw.min()):
        proj_pca_scaled = scaler.fit_transform(proj_pca_raw.reshape(-1, 1)).ravel()
    else:
        proj_pca_scaled = proj_pca_raw

    if len(proj_imp_raw) > 1 and not np.allclose(proj_imp_raw.max(), proj_imp_raw.min()):
        proj_imp_scaled = scaler.fit_transform(proj_imp_raw.reshape(-1, 1)).ravel()
    else:
        proj_imp_scaled = proj_imp_raw

    df = pd.DataFrame({
        "token": tokens,
        "proj_pca": proj_pca_scaled,
        "proj_imp": proj_imp_scaled,
        "importance_raw": importance_raw
    })
    return df


def plot_local_axes(df: pd.DataFrame, title: str = None):
    """
    Визуализация: два ряда — PCA-projection и Importance-projection.
    Подписи слов расположены так, чтобы было читаемо.
    """
    if df is None or df.shape[0] == 0:
        print("Пустой DataFrame — нечего рисовать.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # PCA
    axes[0].scatter(range(len(df)), df["proj_pca"], c=df["proj_pca"], cmap="coolwarm", s=120)
    for i, tok in enumerate(df["token"]):
        axes[0].text(i, df["proj_pca"].iloc[i] + 0.03, tok, ha="center", fontsize=9)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("PCA projection (scaled)")
    axes[0].set_title("Local PCA axis (главная семантическая ось)")

    # Importance
    axes[1].scatter(range(len(df)), df["proj_imp"], c=df["importance_raw"], cmap="viridis", s=120)
    for i, tok in enumerate(df["token"]):
        axes[1].text(i, df["proj_imp"].iloc[i] + 0.03, tok, ha="center", fontsize=9)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Importance projection (scaled)")
    axes[1].set_title("Importance axis (less -> more important)")

    plt.suptitle(title or "Local semantic axes")
    plt.xticks(range(len(df)), df["token"], rotation=45)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()
    
    
"""
Local semantic axes: части речи в 2D (noun / verb / adjective / adverb)

Этот модуль:
 - строит две локальные оси (noun↔verb и adjective↔adverb) на основе
   небольших списков-прототипов (их можно менять).
 - вычисляет проекции слов предложения на эти оси,
   масштабирует проекции в [-1,1] и помещает каждое слово в один из 4 квадрантов:
     (x>0,y>0) -> NOUN
     (x<0,y>0) -> ADJECTIVE
     (x<0,y<0) -> ADVERB
     (x>0,y<0) -> VERB
 - предоставляет функции для анализа предложения и отрисовки результата.

Зависимости: numpy, pandas, matplotlib, sklearn (MinMaxScaler)
Использует глобальную модель, установленную через semgeom.set_model(...)
и embed_cached(...) для кодирования.
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# импортируем утилиты из pairs (в пакете предполагается наличие pairs.py)
from .pairs import embed_cached, feature_direction, set_model, cos_sim  # type: ignore

# токенизация простая (слова)
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def pos_axes_prototypes() -> Dict[str, Dict[str, List[str]]]:
    """
    По умолчанию: noun/entity/object vs verb/action
                  adjective/quality vs adverb/manner
    """
    return {
        "x_axis": {"pos": ["noun", "entity", "object"], "neg": ["verb", "action"]},
        "y_axis": {"pos": ["adjective", "quality"], "neg": ["adverb", "manner"]}
    }


def _scale_to_range(arr: np.ndarray, frange: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    arr = np.asarray(arr).astype(float)
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if np.allclose(mx, mn):
        return np.zeros_like(arr) + ((frange[0] + frange[1]) / 2.0)
    # linear scaling to [frange[0], frange[1]]
    a = (frange[1] - frange[0]) / (mx - mn)
    b = frange[0] - a * mn
    return a * arr + b


def analyze_sentence_pos_axes(sentence: str,
                              axes_defs: Dict[str, Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Для предложения возвращает DataFrame:
      token, x_scaled, y_scaled, quadrant_label

    axes_defs: словарь с двумя осями, например:
        {
          "x_axis": {"pos":["noun","entity"], "neg":["verb","action"]},
          "y_axis": {"pos":["adjective"], "neg":["adverb"]}
        }
    """
    if axes_defs is None:
        axes_defs = pos_axes_prototypes()

    tokens = _tokenize(sentence)
    if len(tokens) == 0:
        return pd.DataFrame(columns=["token", "x_scaled", "y_scaled", "quadrant_label"])

    # берём оси
    axis_x_def = axes_defs.get("x_axis")
    axis_y_def = axes_defs.get("y_axis")
    if axis_x_def is None or axis_y_def is None:
        raise ValueError("axes_defs must contain 'x_axis' and 'y_axis' with pos/neg examples")

    axis_x = feature_direction(axis_x_def)
    axis_y = feature_direction(axis_y_def)

    # нормализация
    axis_x /= (np.linalg.norm(axis_x) + 1e-12)
    axis_y /= (np.linalg.norm(axis_y) + 1e-12)

    # эмбеддинги токенов
    embs = np.vstack([embed_cached(w) for w in tokens])

    x_raw = embs.dot(axis_x)
    y_raw = embs.dot(axis_y)

    x_scaled = _scale_to_range(x_raw, (-1, 1))
    y_scaled = _scale_to_range(y_raw, (-1, 1))

    # квадранты: теперь имена берём из полюсов
    labels = []
    for xv, yv in zip(x_scaled, y_scaled):
        if xv > 0 and yv > 0:
            lbl = axis_x_def["pos"][0].upper()
        elif xv < 0 and yv > 0:
            lbl = axis_y_def["pos"][0].upper()
        elif xv < 0 and yv < 0:
            lbl = axis_y_def["neg"][0].upper()
        elif xv > 0 and yv < 0:
            lbl = axis_x_def["neg"][0].upper()
        else:
            lbl = "MIXED"
        labels.append(lbl)

    df = pd.DataFrame({
        "token": tokens,
        "x_scaled": x_scaled,
        "y_scaled": y_scaled,
        "quadrant_label": labels
    })
    return df


def plot_pos_axes_4zones(df: pd.DataFrame,
                         axes_defs: Dict[str, Dict[str, List[str]]] = None,
                         title: str = None,
                         annotate: bool = True,
                         color_map: Dict[str, str] = None):
    """
    Рисует 2D scatter (x_scaled,y_scaled) c зонами по пользовательским осям.
    """
    if df is None or df.shape[0] == 0:
        print("Empty DataFrame.")
        return

    if axes_defs is None:
        axes_defs = pos_axes_prototypes()

    axis_x_def = axes_defs["x_axis"]
    axis_y_def = axes_defs["y_axis"]

    if color_map is None:
        color_map = {"NOUN": "#1f77b4", "VERB": "#ff7f0e",
                     "ADJECTIVE": "#2ca02c", "ADVERB": "#d62728", "MIXED": "gray"}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)

    for lbl, group in df.groupby("quadrant_label"):
        col = color_map.get(lbl, "gray")
        ax.scatter(group["x_scaled"], group["y_scaled"], label=lbl,
                   s=120, color=col, edgecolor="k", alpha=0.9)

    if annotate:
        for _, row in df.iterrows():
            ax.text(row["x_scaled"] + 0.02, row["y_scaled"] + 0.02, row["token"], fontsize=9)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(f"{axis_x_def['pos'][0]} ↔ {axis_x_def['neg'][0]}")
    ax.set_ylabel(f"{axis_y_def['pos'][0]} ↔ {axis_y_def['neg'][0]}")
    ax.set_title(title or "POS-like zones (2D projection)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---- вспомогательные: удобный wrapper для демонстрации ----
def demo_sentence_pos_viz(sentence: str, axes_defs: Dict[str, Dict[str, List[str]]] = None, annotate: bool = True):
    """
    Быстрый демонстрационный wrapper:
      - соберёт DataFrame через analyze_sentence_pos_axes
      - покажет его и нарисует plot_pos_axes_4zones
    """
    df = analyze_sentence_pos_axes(sentence, axes_defs=axes_defs)
    print(df[["token", "x_scaled", "y_scaled", "quadrant_label"]])
    plot_pos_axes_4zones(df, title=f"POS-like 2D zones — '{sentence}'", annotate=annotate)
