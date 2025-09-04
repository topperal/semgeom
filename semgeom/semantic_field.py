# semgeom/semantic_field.py
"""
Semantic field utilities.

Re-uses core.feature_direction, core.project_words, core.get_scalar_projections
and utils.scale, utils.cos_sim.

Provides:
 - find_top_k_neighbors
 - semantic_field (build DataFrame of neighbors + projections + PCA coords)
 - plot_semantic_field (visualization)
 - geometry helpers (convex hull, alpha-shape optional, coverage)
"""

from typing import List, Dict, Any, Optional, Tuple, Iterable
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import directed_hausdorff

try:
    import alphashape
    from shapely.geometry import Point, Polygon, MultiPoint
    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False

# display fallback for script mode
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)


# reuse core / utils (these expect explicit `model` argument in this project)
from .core import feature_direction, project_words, get_scalar_projections
from .utils import scale as util_scale, cos_sim

import semgeom  # to read semgeom.model fallback if user set it via semgeom.set_model(...)


def _get_model(model: Optional[Any]):
    """Return model: prefer explicit param, else semgeom.model, else raise."""
    if model is not None:
        if not hasattr(model, "encode"):
            raise RuntimeError("Provided model does not implement .encode(text) -> vector.")
        return model
    if getattr(semgeom, "model", None) is not None:
        if not hasattr(semgeom.model, "encode"):
            raise RuntimeError("Global semgeom.model does not implement .encode(text) -> vector.")
        return semgeom.model
    raise RuntimeError("Model is not set. Call semgeom.set_model(your_model) or pass model=... to the function.")


# -------------------------
# Neighbors
# -------------------------
def find_top_k_neighbors(target: str, candidate_vocab: Iterable[str], model: Optional[Any] = None,
                         top_k: int = 50, exclude_self: bool = True) -> List[Tuple[str, float]]:
    """
    Return list of (word, similarity) sorted by descending cosine similarity to target.
    Uses provided model or semgeom.model if model is None.
    """
    mdl = _get_model(model)
    vocab = list(candidate_vocab)
    if target not in vocab:
        vocab.append(target)
    emb = np.vstack([mdl.encode(w) for w in vocab])
    tvec = mdl.encode(target).reshape(1, -1)
    sims = cosine_similarity(tvec, emb).ravel()
    order = np.argsort(sims)[::-1]
    neighbors: List[Tuple[str, float]] = []
    for idx in order:
        w = vocab[idx]
        if exclude_self and w == target:
            continue
        neighbors.append((w, float(sims[idx])))
        if len(neighbors) >= top_k:
            break
    return neighbors


# -------------------------
# Geometry helpers
# -------------------------
def convex_hull_area_2d(coords2d: np.ndarray) -> Tuple[float, Optional[ConvexHull], np.ndarray]:
    coords2d = np.asarray(coords2d)
    if coords2d.shape[0] < 3:
        return 0.0, None, coords2d
    hull = ConvexHull(coords2d)
    area = float(hull.volume)  # for 2D convex hull volume==area
    hull_coords = coords2d[hull.vertices]
    return area, hull, hull_coords


def alpha_shape_area(coords2d: np.ndarray, alpha: Optional[float] = None) -> Tuple[Optional[float], Optional[Any], Optional[float]]:
    """
    If alphashape+shapely available, return (area, polygon_object, alpha_used).
    Otherwise returns (None, None, None).
    """
    if not SHAPELY_AVAILABLE:
        return None, None, None
    pts = [(float(x), float(y)) for x, y in np.asarray(coords2d)]
    if len(pts) < 4:
        mp = MultiPoint(pts)
        return float(mp.convex_hull.area), mp.convex_hull, None
    try:
        if alpha is None:
            alpha = alphashape.optimizealpha(pts)
        poly = alphashape.alphashape(pts, alpha)
        if poly is None:
            mp = MultiPoint(pts)
            return float(mp.convex_hull.area), mp.convex_hull, alpha
        return float(poly.area), poly, alpha
    except Exception:
        try:
            area, hull, hull_coords = convex_hull_area_2d(np.array(pts))
            hull_poly = Polygon(hull_coords)
            return float(area), hull_poly, None
        except Exception:
            return None, None, None


def coverage_fraction_polygon(polygon_or_coords, candidate_coords2d: np.ndarray) -> float:
    """
    Fraction of candidate_coords2d falling inside polygon_or_coords.
    polygon_or_coords can be shapely Polygon (if available) or ndarray hull_coords.
    """
    candidate_coords2d = np.asarray(candidate_coords2d)
    if SHAPELY_AVAILABLE and isinstance(polygon_or_coords, Polygon):
        poly = polygon_or_coords
        inside = [poly.contains(Point(xy)) or poly.touches(Point(xy)) for xy in candidate_coords2d]
        return float(np.mean(inside))
    else:
        hull_coords = np.asarray(polygon_or_coords)
        if hull_coords.shape[0] < 3:
            return 0.0
        delaunay = Delaunay(hull_coords)
        mask = delaunay.find_simplex(candidate_coords2d) >= 0
        return float(mask.mean())


# -------------------------
# High-level semantic field builder
# -------------------------
def semantic_field(target: str,
                   candidate_vocab: Iterable[str],
                   axes_definitions: Dict[str, Dict[str, List[str]]],
                   model: Optional[Any] = None,
                   top_k: int = 30,
                   scale_range: Tuple[float, float] = (-1.0, 1.0),
                   use_centered: bool = True) -> pd.DataFrame:
    mdl = _get_model(model)
    vocab = list(candidate_vocab)
    neighbors = find_top_k_neighbors(target, vocab, mdl, top_k=top_k, exclude_self=True)
    neigh_words = [w for w, _ in neighbors]
    if target not in neigh_words:
        neigh_words = [target] + neigh_words

    embs = np.vstack([mdl.encode(w) for w in neigh_words])
    tvec = mdl.encode(target).reshape(1, -1)
    sims = cosine_similarity(tvec, embs).ravel()
    sim_map = {w: float(s) for w, s in zip(neigh_words, sims)}

    projections: Dict[str, Dict[str, float]] = {}
    for axis_name, axis_def in axes_definitions.items():
        if use_centered:
            raw_map = get_scalar_projections(neigh_words, axis_def, mdl)
            vals = np.array([raw_map[w] for w in neigh_words]).reshape(-1, 1)
            if len(vals) > 1 and not np.allclose(vals.max(), vals.min()):
                scaled = MinMaxScaler(feature_range=scale_range).fit_transform(vals).ravel()
            else:
                scaled = vals.ravel()
            projections[axis_name] = dict(zip(neigh_words, [float(x) for x in scaled]))
        else:
            d = feature_direction(axis_def, mdl)
            scaled_map = project_words(neigh_words, d, mdl, feature_range=scale_range)
            projections[axis_name] = scaled_map

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(embs)

    df = pd.DataFrame({
        "word": neigh_words,
        "similarity_to_target": [sim_map[w] for w in neigh_words],
        "pc1": coords[:, 0],
        "pc2": coords[:, 1]
    })
    for axis_name in axes_definitions.keys():
        df[axis_name] = [projections[axis_name][w] for w in neigh_words]

    return df


def plot_semantic_field(df: pd.DataFrame,
                        target_word: str,
                        axes_to_plot: Optional[List[str]] = None,
                        show_table: bool = True,
                        top_n_table: int = 15,
                        annotate: bool = True):
    if df is None or df.shape[0] == 0:
        print("Empty DataFrame — nothing to plot.")
        return

    if axes_to_plot is None:
        axes_to_plot = [c for c in df.columns if c not in ("word", "similarity_to_target", "pc1", "pc2")]

    n_axes = len(axes_to_plot)
    rows = max(1, n_axes)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    if n_axes == 1:
        axes = np.array([axes])

    for i, axis_name in enumerate(axes_to_plot):
        ax_scatter = axes[i, 0]
        ax_bar = axes[i, 1]

        cmap = plt.get_cmap("coolwarm")
        vals = df[axis_name].values.astype(float)
        norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sc = ax_scatter.scatter(df["pc1"], df["pc2"], c=vals, cmap=cmap, s=80, edgecolor="k")
        # highlight target
        tgt_row = df[df["word"] == target_word]
        if not tgt_row.empty:
            ax_scatter.scatter(tgt_row["pc1"], tgt_row["pc2"], color="gold", edgecolor="k", s=180, label="TARGET")
        if annotate:
            # annotate every point with bbox filled with the same color as its projection
            for _, r in df.iterrows():
                val = float(r[axis_name])
                color = cmap(norm(val))
                ax_scatter.text(r["pc1"] + 0.01, r["pc2"] + 0.01, r["word"],
                                fontsize=9,
                                bbox=dict(facecolor=color, alpha=0.75, edgecolor='none', pad=0.6))
        ax_scatter.set_title(f"PCA field — colored by '{axis_name}' projection")
        ax_scatter.axis("off")
        cbar = fig.colorbar(sm, ax=ax_scatter, label=f"{axis_name} (scaled)")

        # bar chart sorted by projection
        sorted_df = df.sort_values(axis_name)
        ax_bar.hlines(y=range(len(sorted_df)), xmin=0, xmax=sorted_df[axis_name], color="tab:blue", alpha=0.7)
        ax_bar.set_yticks(range(len(sorted_df)))
        ax_bar.set_yticklabels(sorted_df["word"])
        ax_bar.set_xlabel("projection (scaled)")
        ax_bar.set_title(f"Projections on '{axis_name}' (neg -> pos)")

    plt.suptitle(f"Semantic field for '{target_word}' — top {len(df)} neighbors", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    if show_table:
        display_cols = ["word", "similarity_to_target"] + axes_to_plot + ["pc1", "pc2"]
        display_df = df[display_cols].copy()
        display_df = display_df.sort_values("similarity_to_target", ascending=False).head(top_n_table)
        pd.set_option("display.max_rows", None)
        display(display_df.reset_index(drop=True))


def plot_field_result(result, target, show_hull=True, show_alpha=True):
    coords = np.asarray(result["coords2d"])
    words = result["words"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # plot all points
    ax.scatter(coords[:, 0], coords[:, 1], s=80, alpha=0.8, edgecolor='k', zorder=1)
    # highlight hull vertices if available
    hull = result.get("hull", None)
    if hull is not None:
        proto_idxs = hull.vertices
        ax.scatter(coords[proto_idxs, 0], coords[proto_idxs, 1], s=160, facecolors='none', edgecolor='r', linewidths=1.5, zorder=3, label='hull vertices')
    # annotate words
    for i, w in enumerate(words):
        ax.text(coords[i, 0] + 0.01, coords[i, 1] + 0.01, w, fontsize=9)
    if show_hull and hull is not None:
        for simplex in hull.simplices:
            x = coords[simplex, 0]; y = coords[simplex, 1]
            ax.plot(x, y, 'k-')
    if show_alpha and SHAPELY_AVAILABLE and result.get("alpha_poly") is not None:
        poly = result["alpha_poly"]
        try:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, 'r--', linewidth=2, label=f"alpha (a={result['alpha_param']:.3f})")
        except Exception:
            pass
    ax.legend()
    ax.set_title(f"Field for '{target}' (convex_area={result['convex_area']:.3f})")
    plt.show()


def plot_combined_fields(results: Dict[str, Any], targets: List[str], show_hull: bool = True, show_alpha: bool = True, figsize=(9, 9)):
    """
    Draw overlay of all target fields (each target in different color),
    plotting points and convex hull edges (and alpha-shape if available).
    """
    import matplotlib.pyplot as plt
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for idx, t in enumerate(targets):
        c = colors[idx % len(colors)]
        r = results[t]
        coords = np.asarray(r["coords2d"])[:, :2]
        ax.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.4, label=t, color=c)
        if show_hull and r.get("hull") is not None:
            hull = r["hull"]
            for s in getattr(hull, "simplices", []):
                ax.plot(coords[s, 0], coords[s, 1], color=c, linewidth=1)
        if show_alpha and SHAPELY_AVAILABLE and isinstance(r.get("alpha_poly", None), Polygon):
            poly = r["alpha_poly"]
            try:
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, linestyle='--', color=c, linewidth=2, alpha=0.9)
            except Exception:
                pass
    ax.legend()
    ax.set_title("Combined semantic fields (convex hulls solid, alpha dashed if available)")
    plt.show()

def symmetric_hausdorff(A_coords, B_coords): 
    d1 = directed_hausdorff(A_coords, B_coords)[0] 
    d2 = directed_hausdorff(B_coords, A_coords)[0] 
    return float(max(d1, d2))

def analyze_targets_geometry(targets: List[str],
                             candidate_vocab: Iterable[str],
                             model: Optional[Any] = None,
                             k: int = 50,
                             pca_n: int = 2,
                             use_alpha: bool = True,
                             bootstrap_ngroups: int = 200,
                             topN_for_bootstrap: Optional[int] = None,
                             rng_seed: int = 42) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], Any], pd.DataFrame]:
    """
    End-to-end geometry analysis for a list of targets.

    Returns:
      - results: dict[target] -> detailed metrics and arrays
      - pairwise: dict[(t1,t2)] -> hausdorff, intersection areas...
      - summary_df: DataFrame summary per target
    """
    mdl = _get_model(model)
    rng = np.random.default_rng(rng_seed)

    candidate_list = list(candidate_vocab)
    # encode candidate vocab once (for coverage + bootstrap sampling)
    candidate_embs = np.vstack([mdl.encode(w) for w in candidate_list]) if len(candidate_list) > 0 else np.zeros((0, mdl.encode("a").shape[0]))

    # 1) collect neighbor blocks and embeddings
    neighbors_map: Dict[str, List[str]] = {}
    emb_blocks: List[np.ndarray] = []
    words_blocks: List[List[str]] = []
    for t in targets:
        neigh = find_top_k_neighbors(t, candidate_list, mdl, top_k=k, exclude_self=True)
        neigh_words = [w for w, s in neigh]
        block_words = [t] + [w for w in neigh_words if w != t]
        block_embs = np.vstack([mdl.encode(w) for w in block_words])
        neighbors_map[t] = block_words
        emb_blocks.append(block_embs)
        words_blocks.append(block_words)

    # 2) fit global scaler + PCA on union of emb_blocks
    concat = np.vstack(emb_blocks) if len(emb_blocks) > 0 else np.zeros((0, mdl.encode("a").shape[0]))
    scaler = StandardScaler().fit(concat)
    concat_scaled = scaler.transform(concat)
    pca = PCA(n_components=pca_n, random_state=0).fit(concat_scaled)

    # transform blocks into coords
    coords_list = []
    for emb in emb_blocks:
        scaled = scaler.transform(emb)
        coords = pca.transform(scaled)[:, :pca_n]
        coords_list.append(coords)

    # candidate coords for coverage
    candidate_coords2d = pca.transform(scaler.transform(candidate_embs))[:, :2] if candidate_embs.shape[0] > 0 else np.zeros((0,2))

    # 3) per-target metrics
    results: Dict[str, Any] = {}
    for i, t in enumerate(targets):
        emb = emb_blocks[i]
        words = words_blocks[i]
        coords2d = coords_list[i]
        convex_area, hull_obj, hull_coords = convex_hull_area_2d(coords2d[:, :2])

        alpha_area, alpha_poly, alpha_param = (None, None, None)
        if use_alpha and SHAPELY_AVAILABLE:
            alpha_area, alpha_poly, alpha_param = alpha_shape_area(coords2d[:, :2], alpha=None)

        if hull_coords is not None and hull_coords.shape[0] >= 3:
            if SHAPELY_AVAILABLE:
                hull_poly = Polygon(hull_coords)
                coverage = coverage_fraction_polygon(hull_poly, candidate_coords2d)
            else:
                coverage = coverage_fraction_polygon(hull_coords, candidate_coords2d)
        else:
            coverage = 0.0
            hull_poly = None

        results[t] = {
            "words": words,
            "embeddings": emb,
            "coords2d": coords2d,
            "hull": hull_obj,
            "hull_coords": hull_coords,
            "convex_area": convex_area,
            "alpha_area": alpha_area,
            "alpha_poly": alpha_poly,
            "alpha_param": alpha_param,
            "coverage_fraction": coverage
        }

    # 4) bootstrap hull areas (fixed scaler+pca)
    N_vocab = len(candidate_list)
    topN_for_bootstrap = topN_for_bootstrap or N_vocab
    for t in targets:
        areas = []
        t_emb = mdl.encode(t)
        for _ in range(bootstrap_ngroups):
            samp_idx = rng.integers(0, N_vocab, size=N_vocab)
            sampled_embs = candidate_embs[samp_idx]
            sims = cosine_similarity(t_emb.reshape(1, -1), sampled_embs).ravel()
            top_idxs = np.argsort(sims)[::-1][:min(k, topN_for_bootstrap)]
            chosen_embs = sampled_embs[top_idxs]
            if chosen_embs.shape[0] < 3:
                areas.append(0.0)
                continue
            chosen_scaled = scaler.transform(chosen_embs)
            chosen_coords = pca.transform(chosen_scaled)[:, :2]
            area, _, _ = convex_hull_area_2d(chosen_coords)
            areas.append(area)
        areas = np.array(areas)
        results[t]["bootstrap_areas"] = areas
        results[t]["bootstrap_mean"] = float(np.mean(areas))
        results[t]["bootstrap_std"] = float(np.std(areas))

    # 5) representational sparsity via NNLS using hull vertices as prototypes
    for t in targets:
        r = results[t]
        hull = r["hull"]
        emb_block = r["embeddings"]  # Ni x D
        words = r["words"]
        if hull is None or getattr(hull, "vertices", None) is None or len(hull.vertices) == 0:
            results[t]["representational_sparsity"] = None
            continue
        proto_idxs = np.array(hull.vertices, dtype=int)
        prototypes = emb_block[proto_idxs]  # m x D
        A = prototypes.T  # D x m
        stats = []
        for j, w in enumerate(words):
            b = emb_block[j]
            # NNLS
            try:
                sol, _ = nnls(A, b)
            except Exception:
                sol = np.zeros(A.shape[1])
            s = sol.sum()
            soln = sol / (s + 1e-12) if s > 0 else sol
            nnz = int((soln > 1e-3).sum())
            recon = (prototypes.T @ soln)  # D-length
            err = float(np.linalg.norm(b - recon) / (np.linalg.norm(b) + 1e-12))
            stats.append({"word": w, "n_nonzero": nnz, "recon_err": err, "weights": soln, "proto_indices": proto_idxs.tolist()})
        results[t]["representational_sparsity"] = stats

    # 6) pairwise comparisons (hausdorff, hull intersection if shapely available)
    pairwise: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            t1 = targets[i]; t2 = targets[j]
            coords1 = results[t1]["coords2d"][:, :2]; coords2 = results[t2]["coords2d"][:, :2]
            hdist = symmetric_hausdorff(coords1, coords2)
            inter_area_convex = None
            inter_frac_convex = None
            inter_area_alpha = None
            inter_frac_alpha = None

            hull1 = results[t1]["hull_coords"]; hull2 = results[t2]["hull_coords"]
            if hull1 is not None and hull2 is not None and hull1.shape[0] >= 3 and hull2.shape[0] >= 3 and SHAPELY_AVAILABLE:
                poly1 = Polygon(hull1); poly2 = Polygon(hull2)
                inter = poly1.intersection(poly2)
                inter_area_convex = float(inter.area)
                min_area = max(1e-12, min(results[t1]["convex_area"], results[t2]["convex_area"]))
                inter_frac_convex = float(inter_area_convex / min_area)
            pairwise[(t1, t2)] = {
                "hausdorff": hdist,
                "intersection_area_convex": inter_area_convex,
                "intersection_fraction_convex_of_smaller": inter_frac_convex,
                "intersection_area_alpha": inter_area_alpha,
                "intersection_fraction_alpha_of_smaller": inter_frac_alpha
            }

    # 7) summary DataFrame
    rows = []
    for t in targets:
        r = results[t]
        rows.append({
            "target": t,
            "convex_area": r["convex_area"],
            "alpha_area": r["alpha_area"],
            "alpha_param": r["alpha_param"],
            "coverage_fraction": r["coverage_fraction"],
            "bootstrap_mean_area": r["bootstrap_mean"],
            "bootstrap_std_area": r["bootstrap_std"],
            "neighbors_count": len(r["words"])
        })
    summary_df = pd.DataFrame(rows)
    return results, pairwise, summary_df