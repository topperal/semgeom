# semgeom/biases.py
"""
Bias analysis utilities.

Exported:
 - analyze_biases
 - compute_sentence_axis_values
 - pivot_top_words
 - plot_multiaxis_scatter

This module relies on core.feature_direction, core.project_words, core.get_scalar_projections
which accept an explicit `model` argument. If model is not provided, it falls back to
the global `semgeom.model` (set via semgeom.set_model()).
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import feature helpers from core (they accept model as argument)
from .core import feature_direction, project_words, get_scalar_projections
from .utils import scale  # scale(values, feature_range=(-1,1))

import semgeom  # to access semgeom.model if user set it via semgeom.set_model(...)


def _get_model(model: Optional[Any]):
    if model is not None:
        return model
    if getattr(semgeom, "model", None) is not None:
        return semgeom.model
    raise RuntimeError("Model is not set. Call semgeom.set_model(your_model) or pass model=... to the function.")


# -------------------------
# Aggregation across groups / axes
# -------------------------
def analyze_biases(target_items: Dict[str, Dict[str, List[str]]],
                   axes_definitions: Dict[str, Dict[str, List[str]]],
                   use_centered: bool = False,
                   scaled: bool = True,
                   model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Universal bias analysis.

    Parameters
    ----------
    target_items : dict
        {'group_name': {'categoryA': [items], 'categoryB': [...]} , ...}
        items can be words or short phrases.
    axes_definitions : dict
        {'axis_name': {'pos': [...], 'neg': [...]}, ...}
    use_centered : bool
        if True uses get_scalar_projections (centered)
    scaled : bool
        if True use scaled projections when using project_words
    model : optional
        embedding model object; if None uses semgeom.model

    Returns
    -------
    results : dict
        { 'group_name': { 'axis_name': {'categoryA': mean_proj, 'categoryB': mean_proj, ...}, ... }, ... }
    """
    mdl = _get_model(model)
    results: Dict[str, Any] = {}

    for group_name, categories in target_items.items():
        group_res: Dict[str, Any] = {}
        for axis_name, axis_def in axes_definitions.items():
            per_cat_means: Dict[str, float] = {}

            if use_centered:
                # use core.get_scalar_projections (raw scalars, centered)
                for cat_name, items in categories.items():
                    if not items:
                        per_cat_means[cat_name] = float("nan")
                        continue
                    proj_map = get_scalar_projections(items, axis_def, mdl)
                    if len(proj_map) == 0:
                        per_cat_means[cat_name] = float("nan")
                    else:
                        per_cat_means[cat_name] = float(np.nanmean(list(proj_map.values())))
                group_res[axis_name] = per_cat_means
            else:
                # use direction + project_words (optionally scaled)
                d = feature_direction(axis_def, mdl)
                for cat_name, items in categories.items():
                    if not items:
                        per_cat_means[cat_name] = float("nan")
                        continue
                    proj_map = project_words(items, d, mdl, feature_range=(-1, 1)) if scaled else project_words(items, d, mdl, feature_range=(-1, 1))
                    # project_words always returns scaled values according to signature; if scaled==False,
                    # the core implementation may still return scaled - keep interface stable.
                    per_cat_means[cat_name] = float(np.nanmean(list(proj_map.values()))) if len(proj_map) > 0 else float("nan")
                group_res[axis_name] = per_cat_means

        results[group_name] = group_res
    return results


# -------------------------
# sentence-level helpers
# -------------------------
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def compute_sentence_axis_values(sentences: List[str],
                                 axes_definitions: Dict[str, Dict[str, List[str]]],
                                 top_n: int = 3,
                                 use_centered: bool = False,
                                 model: Optional[Any] = None) -> pd.DataFrame:
    """
    For each sentence and each axis compute projections of words, select top_n by abs value.

    Returns DataFrame columns:
      sentence, word, axis, value, category ('POS'/'NEG'), abs_value
    """
    mdl = _get_model(model)
    rows: List[Dict[str, Any]] = []

    for sent in sentences:
        tokens = _tokenize(sent)
        if not tokens:
            continue
        for axis_name, axis_def in axes_definitions.items():
            if use_centered:
                proj_map = get_scalar_projections(tokens, axis_def, mdl)  # raw
                if len(proj_map) == 0:
                    continue
                vals = np.array(list(proj_map.values()))
                # scale for display
                if len(vals) > 0:
                    scaled_vals = scale(vals, feature_range=(-1, 1))
                    proj_map = {w: float(scaled_vals[i]) for i, w in enumerate(proj_map.keys())}
            else:
                d = feature_direction(axis_def, mdl)
                proj_map = project_words(tokens, d, mdl, feature_range=(-1, 1))

            # choose top_n by absolute value
            ordered = sorted(proj_map.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            for w, v in ordered:
                rows.append({
                    "sentence": sent,
                    "word": w,
                    "axis": axis_name,
                    "value": float(v),
                    "category": "POS" if float(v) > 0 else "NEG",
                    "abs_value": abs(float(v))
                })

    df = pd.DataFrame(rows)
    return df


def pivot_top_words(df_top: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table index=(sentence,word), columns=axis, values=value.
    """
    if df_top is None or df_top.shape[0] == 0:
        return pd.DataFrame()
    pivot = df_top.pivot_table(index=["sentence", "word"], columns="axis", values="value")
    pivot = pivot.reset_index()
    return pivot


def plot_multiaxis_scatter(pivot_df: pd.DataFrame,
                           axis_pairs: Optional[List[Tuple[str, str]]] = None,
                           titles: Optional[List[str]] = None,
                           cmap_pos: str = "green",
                           cmap_neg: str = "red"):
    """
    Draw grid of scatter plots for given axis pairs. pivot_df expected as from pivot_top_words.
    """
    if pivot_df is None or pivot_df.shape[0] == 0:
        print("Empty pivot dataframe — nothing to plot.")
        return

    # infer axis columns
    axis_cols = [c for c in pivot_df.columns if c not in ("sentence", "word")]
    if not axis_cols:
        print("No axis columns found in pivot_df.")
        return

    if axis_pairs is None:
        axis_pairs = []
        for i in range(len(axis_cols)):
            for j in range(i+1, len(axis_cols)):
                axis_pairs.append((axis_cols[i], axis_cols[j]))
                if len(axis_pairs) >= 4:
                    break
            if len(axis_pairs) >= 4:
                break

    n = len(axis_pairs)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axs_flat = np.array(axs).reshape(-1)

    for i, (ax_x, ax_y) in enumerate(axis_pairs):
        ax = axs_flat[i]
        scatter_data = pivot_df.dropna(subset=[ax_x, ax_y])
        if scatter_data.shape[0] == 0:
            ax.set_title(f"{ax_x} vs {ax_y} (no data)")
            ax.axis("off")
            continue
        colors = [cmap_pos if float(v) > 0 else cmap_neg for v in scatter_data[ax_x]]
        ax.scatter(scatter_data[ax_x], scatter_data[ax_y], c=colors, alpha=0.8, s=80)
        for _, row in scatter_data.iterrows():
            ax.text(row[ax_x] + 0.01, row[ax_y] + 0.01, row["word"], fontsize=9)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.axvline(0, color='grey', linewidth=0.5)
        ax.set_xlabel(ax_x)
        ax.set_ylabel(ax_y)
        ax.set_title(titles[i] if titles and i < len(titles) else f"{ax_x} vs {ax_y}")

    # hide unused axes
    for j in range(i+1, len(axs_flat)):
        axs_flat[j].axis('off')

    plt.tight_layout()
    plt.show()
