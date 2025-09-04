"""
semgeom package initializer

Pattern:
 - expose a small public API (core helpers, pairs, viz, local_axes, biases, semantic_field)
 - provide package-level `model` and `set_model()` that also delegates to pairs.set_model when available.
"""

# public model slot (set by user)
model = None

def set_model(m):
    """
    Set the global model used by semgeom modules.

    Example:
      from sentence_transformers import SentenceTransformer
      semgeom.set_model(SentenceTransformer("all-MiniLM-L6-v2"))
    """
    global model
    model = m
    # also try to propagate to pairs module (if available) to keep its internal cache consistent
    try:
        from .pairs import set_model as _pairs_set_model  # late import to avoid import cycles
        _pairs_set_model(m)
    except Exception:
        # pairs may not be importable yet or may implement a different API; ignore silently
        pass

# --- Now import submodules and selected functions (after model/set_model defined to avoid cycles) ---
from .core import feature_direction, project_words, get_scalar_projections
from .viz import plot_projection_from_dict, plot_3d_projection_interactive
from .utils import scale, cos_sim as cos_sim_utils
from .pairs import (
    embed_cached, evaluate_pairs_on_scales,
    plot_pair_on_two_scales, cos_sim, angular_deg, rel_euclid
)
from .local_axes import (
    pick_pos_neg_PCA, pick_pos_neg_max_distance, pick_pos_neg_center_extremes,
    pick_pos_neg_kmeans, plot_local_axes_strategies, auto_axis_from_sentence,
    compare_word_on_axes, plot_axes_scatter, build_direction_from_examples,
    build_local_axis_pca, build_importance_axis, analyze_sentence_local_axes,
    plot_local_axes, pos_axes_prototypes, analyze_sentence_pos_axes,
    plot_pos_axes_4zones, demo_sentence_pos_viz
)
from .biases import (
    analyze_biases, compute_sentence_axis_values, pivot_top_words, plot_multiaxis_scatter
)
from .semantic_field import (
    semantic_field, plot_semantic_field, find_top_k_neighbors,
    convex_hull_area_2d, alpha_shape_area, coverage_fraction_polygon, plot_field_result, 
    plot_combined_fields, analyze_targets_geometry
)
from .topic_coverage import (
    preprocess_text, assign_tokens_to_categories,
    document_topic_coverage, document_topic_coverage_by_sentence,
    plot_doc_topic_proportions, plot_sentence_level
)

# Re-export (public API)
__all__ = [
    # model management
    "model", "set_model",

    # core / projections
    "feature_direction", "project_words", "get_scalar_projections",

    # pairs / embedding helpers & geometry metrics
    "embed_cached", "evaluate_pairs_on_scales", "plot_pair_on_two_scales",
    "cos_sim", "angular_deg", "rel_euclid",

    # utils
    "scale", "cos_sim_utils",

    # local axes
    "pick_pos_neg_PCA", "pick_pos_neg_max_distance", "pick_pos_neg_center_extremes",
    "pick_pos_neg_kmeans", "plot_local_axes_strategies", "auto_axis_from_sentence",
    "compare_word_on_axes", "plot_axes_scatter", "build_direction_from_examples",
    "build_local_axis_pca", "build_importance_axis", "analyze_sentence_local_axes",
    "plot_local_axes", "pos_axes_prototypes", "analyze_sentence_pos_axes",
    "plot_pos_axes_4zones", "demo_sentence_pos_viz",

    # biases
    "analyze_biases", "compute_sentence_axis_values", "pivot_top_words", "plot_multiaxis_scatter",

    # semantic field
    "semantic_field", "plot_semantic_field", "find_top_k_neighbors",
    "convex_hull_area_2d", "alpha_shape_area", "coverage_fraction_polygon", "plot_field_result", 
    "plot_combined_fields", "analyze_targets_geometry",

    # visualization helpers
    "plot_projection_from_dict", "plot_3d_projection_interactive",
    
    # topic coverage
    "preprocess_text", "assign_tokens_to_categories",
    "document_topic_coverage", "document_topic_coverage_by_sentence",
    "plot_doc_topic_proportions", "plot_sentence_level",
]
