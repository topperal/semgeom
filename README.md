
# semgeom - Semantic Geometry

Semantic Geometry provides a toolkit for exploring semantic fields, building projection axes, measuring biases, and visualizing embedding geometry produced by transformer-based sentence / word encoders (for example, models from sentence-transformers). The library is split into focused modules so you can mix low-level projection utilities with higher-level analysis pipelines and visualizations.

SemGeom is a Python library designed to analyze the interpretability of transformer models. It provides tools to inspect and understand how transformers process and represent semantic information, helping researchers and practitioners gain deeper insights into model behavior.

Features:
- Analyze internal representations of transformer-based models
- Explore semantic geometry of embeddings
- Visualize and interpret attention patterns
- Lightweight and easy to integrate into existing pipelines


## Quick Start

Install required packages:

```bash
pip install sentence-transformers numpy pandas scikit-learn matplotlib seaborn scipy
# optional extras used in visualization/geometry:
pip install plotly shapely adjustText
# optional for some examples:
pip install nltk
```

Set a global model used by the library (recommended pattern — do this once per process):

```bash
from sentence_transformers import SentenceTransformer
import semgeom

model = SentenceTransformer("all-MiniLM-L6-v2")
semgeom.set_model(model)
```

Most functions accept a model parameter so you can pass a model explicitly; however the package also expects a globally-set model (via semgeom.set_model) in many helper functions and higher-level pipelines.

# Usage

## core.py — Low-level projection primitives

This module contains the basic building blocks to compute **semantic projection axes** (directions) and project word embeddings onto those axes.

---

### 1. `feature_direction(feature_words, model)`

**Inputs**
- `feature_words`: `{"pos": List[str], "neg": List[str]}`  
  Small lists of positive and negative seed words that define the semantic axis.
- `model`: a sentence-transformer style model object exposing `.encode(...)` → `np.ndarray`.

**Returns**
- `np.ndarray` — unit-length vector (same dimensionality as model embeddings) giving the semantic axis **pos - neg**.

**What it does**
1. Encodes each `pos` and `neg` token (or short phrase) with `model.encode`.
2. Forms all pairwise differences `pos_vec - neg_vec` and averages them.
3. Normalizes the average vector to unit length.

**Notes**
- Works best with **multiple** `pos` and `neg` examples; using a single pair is allowed.
- If the mean vector has zero norm, returns the mean vector (zero-vector).

---

### 2. `project_words(words, direction, model, feature_range=(-1,1))`

**Inputs**
- `words`: list of tokens/phrases to project.
- `direction`: axis vector (e.g., output of `feature_direction`).
- `model`: encoder.
- `feature_range`: tuple `(min, max)` used for global **MinMaxScaler** scaling of raw dot products (default `(-1, 1)`).

**Returns**
- `dict{word: scaled_value}` — scaled projection values for each word.

**What it does**
1. Encodes words via `model`.
2. Computes raw dot product `embedding · direction`.
3. Scales raw values across the given word list to `feature_range` with `sklearn.preprocessing.MinMaxScaler`.

**Notes**
- If you need **raw unscaled** projections use `get_scalar_projections` (next function).

---

### 3. `get_scalar_projections(words, feature_words, model)`

**Inputs**
- `words`: list of tokens/phrases.
- `feature_words`: axis definition as `{"pos":[...],"neg":[...]}`.
- `model`: encoder.

**Returns**
- `dict{word: raw_scalar}` — raw dot product for each word (no scaling).

**What it does**
- Internally computes axis with `feature_direction` and then dot products for each word.

**Notes**
- Useful when you want absolute raw magnitudes or want to apply your own scaling.

---

### Example

```python
categories = {
    "animals": ["dog","cat","elephant","mouse","tiger","whale","bird","hamster"],
    "professions": ["teacher","doctor","nurse","pilot","chef"]
}
features = {
    "size": {"pos": ["large","big","huge"], "neg": ["small","little","tiny"]},
    "danger": {"pos": ["dangerous","deadly","threatening"], "neg": ["safe","harmless","calm"]}
}

# 1) Build size axis
dir_size = feature_direction(features["size"], model)
print("Size axis (first 6 dims):", dir_size[:6])

# 2) Project animals on size axis (scaled -1..1)
proj_animals_size = project_words(categories["animals"], dir_size, model)
print("Animal projections (sample):")
for w, val in list(proj_animals_size.items())[:8]:
    print(f"  {w:12s} -> {val:.4f}")

# 3) Raw scalar projections for professions on danger axis
proj_prof_danger = get_scalar_projections(categories["professions"], features["danger"], model)
print("Professions raw danger projections:")
for w, v in proj_prof_danger.items():
    print(f"  {w:12s} -> {v:.4f}")

# 4) Quick viz (matplotlib window)
plot_projection_from_dict(proj_animals_size, title="Animals on SIZE axis")
```

## viz.py — Quick visual helpers

Small helpers to quickly plot 1-D/2-D/3-D projections and categorical distributions.

---

### `plot_projection_from_dict(projections, title=None)`

**Inputs**
- `projections`: `dict{word: numeric}` — word → scalar value.
- `title`: optional plot title string.

**Behaviour**
- Creates a horizontal ordered scatter/line with words on y-axis and projection value on x-axis.
- Adds a vertical line at 0 to denote the axis midpoint.
- If running in an interactive environment, the plot will pop up; in headless envs it will render inline in notebooks.

---

### `plot_3d_projection_interactive(words, feature_words, model, title=None)`

**Inputs**
- `words`: list of tokens/phrases to plot.
- `feature_words`: dict of axes, e.g. `{"axis1": {"pos":[...], "neg":[...]}, "axis2": {...}}`.
- `model`: encoder.

**Behaviour**
- Computes directions for each axis and projects words.
- If **Plotly** is installed, shows an interactive 3D scatter with hover labels.
- Falls back to **matplotlib 3D** if Plotly is not available.

---
### Example

```python
# 1) prepare a small list of words to visualize
animals = ["cat", "dog", "elephant", "mouse", "tiger", "whale", "hamster", "horse"]

# 2) define a single feature axis (must be a dict with 'pos' and 'neg')
size_axis = {"pos": ["big", "large", "huge"], "neg": ["small", "little", "tiny"]}

# 3) Call the interactive 3D visualization helper.
semgeom.viz.plot_3d_projection_interactive(animals, size_axis, model, title="3D projection (size axis)")
```

## pairs.py — Pairwise sentence evaluation on semantic scales

Evaluate sentence pairs (original vs paraphrase or two variants) along semantic scales (axes). Useful for testing meaning-preservation, paraphrase shifts, or semantic perturbations on defined scales.

---

### `evaluate_pairs_on_scales(pairs, scales_dict, include_scaled=False)`

**Inputs**
- `pairs`: list of 2-tuples `(original, paraphrase)` (both strings).
- `scales_dict`: dictionary `{scale_name: {"pos":[...], "neg":[...]}}`.
- `include_scaled`: if `True`, also return **scaled deltas** per-scale (global scaling across all pairs).

**Returns**
`pandas.DataFrame` with one row per pair and columns:
- `orig`, `para`
- `cos_sim` – cosine similarity between the two sentence embeddings
- one or more `scale_raw_delta` columns (difference `para - orig`)
- optionally one or more `scale_scaled_delta` columns (global MinMax scaler)

**Behaviour**
1. Encodes each original and paraphrase sentence.
2. Projects both onto every axis defined in `scales_dict`.
3. Computes differences (`para - orig`) for each axis.
4. Optionally computes globally-scaled deltas if `include_scaled=True`.

---

### `plot_pair_on_two_scales(df, scales_dict, scale_x, scale_y, title=None)`

**Inputs**
- `df`: DataFrame result from `evaluate_pairs_on_scales`.
- `scale_x`, `scale_y`: names of two scales to plot.
- `title`: optional plot title.

**Behaviour**
- Draws **arrows** from `orig` → `para` in the 2-D plane (`scale_x` vs `scale_y`).
- Colors arrows by **cosine similarity** (blue = similar, red = dissimilar).
- Labels original points with the sentence text (truncated if long).

---

### Example

```python

features = {
    "size": {"pos": ["large","big"], "neg": ["small","tiny"]},
    "danger": {"pos": ["dangerous","deadly"], "neg": ["safe","harmless"]}
}

pairs = [
    ("The cat sat on the mat.", "A cat was sitting on a mat."),
    ("He finished the job quickly.", "He completed the task in no time."),
    ("The forest fire was dangerous.", "The wildfire posed a serious threat.")
]

df = semgeom.evaluate_pairs_on_scales(pairs, features, include_scaled=False)
print(df.round(3))

semgeom.plot_pair_on_two_scales(df, features, scale_x="size", scale_y="danger",
                                title="Pairs on size vs danger")
```


---

## local_axes.py - Extracting axes from a single sentence

Utilities for constructing local axes inside a sentence (PCA, k-means, extreme selection), comparing multiple axis-picking strategies, and visualizing token-level projections. These functions are helpful for exploring different ways to choose local semantic axes from the words/tokens that appear in a sentence and for visualizing how tokens project onto those axes.

### `pick_pos_neg_PCA(words)`

**Inputs**  
- `words: List[str]` — token list (must be encodable by the global model).

**Behaviour**  
- Embeds every token, L2-normalizes, then computes the **1st principal component**.  
- Selects tokens with **highest** and **lowest** loadings on that component as the **positive** and **negative** anchors.  
- Returns exactly two tokens (one in `pos`, one in `neg`) by default.

**Returns**  
- `Dict[str, List[str]]` — `{"pos": [word], "neg": [word]}`.

---

### `pick_pos_neg_max_distance(words)`

**Inputs**  
- `words: List[str]`.

**Behaviour**  
- Computes **cosine distances** between every pair of token embeddings.  
- Identifies the pair with **maximum distance** and uses them as the **pos / neg** poles.

**Returns**  
- `Dict[str, List[str]]` — `{"pos": [word], "neg": [word]}`.

---

### `pick_pos_neg_center_extremes(words)`

**Inputs**  
- `words: List[str]`.

**Behaviour**  
- Calculates the **centroid** of all token embeddings.  
- Chooses the token **farthest** from the centroid as *positive* and the **closest** as *negative*.

**Returns**  
- `Dict[str, List[str]]` — `{"pos": [extreme], "neg": [center]}`.

---

### `pick_pos_neg_kmeans(words)`

**Inputs**  
- `words: List[str]`.

**Behaviour**  
- Runs **K-means (k = 2)** on the embeddings.  
- Returns the **nearest token to each centroid** as the *pos / neg* anchors.

**Returns**  
- `Dict[str, List[str]]` — `{"pos": [nearest_to_centroid_1], "neg": [nearest_to_centroid_2]}`.

---

### `build_direction_from_examples(pos, neg, model=None)`

**Inputs**  
- `pos: List[str]` — positive seed words.  
- `neg: List[str]` — negative seed words.  
- `model` (optional) — explicit encoder.

**Behaviour**  
- Computes the **mean embedding difference** (pos − neg) and **normalizes** to unit length.

**Returns**  
- `np.ndarray` — unit-length axis vector.

---

### `build_local_axis_pca(tokens, model=None)`

**Inputs**  
- `tokens: List[str]`.

**Behaviour**  
- Extracts the **first principal component** of the token embeddings.  
- Normalizes to unit length.

**Returns**  
- `np.ndarray` — unit-length PCA direction vector.

---

### `build_importance_axis(tokens)`

**Inputs**  
- `tokens: List[str]`.

**Behaviour**  
- Estimates **importance** as **1 − cosine similarity** to the sentence centroid.  
- Builds an axis from the **least important** to the **most important** token.

**Returns**  
- `Tuple[np.ndarray, np.ndarray]` — `(direction, importance_raw)`  
  - `direction`: unit vector  
  - `importance_raw`: raw importance scores aligned to tokens.

---

### `analyze_sentence_local_axes(sentence)`

**Inputs**  
- `sentence: str`.

**Behaviour**  
- Tokenizes, computes **PCA axis** and **importance axis**, projects tokens, and **scales** to `[-1, 1]`.

**Returns**  
- `pandas.DataFrame` with columns:  
  `token`, `proj_pca`, `proj_imp`, `importance_raw`, …

---

### `plot_local_axes(df, title=None)`

**Inputs**  
- `df: pd.DataFrame` — result of `analyze_sentence_local_axes`.  
- `title: str` (optional).

**Behaviour**  
- Creates a **two-row plot**:  
  - Row 1: PCA projections (scaled).  
  - Row 2: Importance projections (scaled).  
- Annotates extreme tokens.

**Returns**  
- `matplotlib.figure.Figure`.

---

### `pos_axes_prototypes()`

**Inputs**  
- None.

**Behaviour**  
- Returns a **default set** of semantic axes (noun vs verb, adjective vs adverb) using prototype words.

**Returns**  
- `Dict[str, Dict[str, List[str]]]` — `{"x_axis": {...}, "y_axis": {...}}`.

---

### `analyze_sentence_pos_axes(sentence, axes_defs=None)`

**Inputs**  
- `sentence: str`.  
- `axes_defs` (optional) — two-axis dict; defaults to `pos_axes_prototypes()`.

**Behaviour**  
- Tokenizes, projects tokens onto **x** and **y** axes, **scales** to `[-1, 1]`, assigns **quadrants**.

**Returns**  
- `pandas.DataFrame` with columns:  
  `token`, `x_scaled`, `y_scaled`, `quadrant_label`.

---

### `plot_pos_axes_4zones(df, axes_defs=None, title=None, annotate=False)`

**Inputs**  
- `df: pd.DataFrame` — from `analyze_sentence_pos_axes`.  
- `axes_defs`, `title`, `annotate` (optional).

**Behaviour**  
- Plots tokens in a **2-D scatter** with **quadrant lines** at 0,0.  
- Colors by quadrant; optionally labels tokens.

**Returns**  
- `matplotlib.figure.Figure`.

---

### `plot_axes_scatter(df, axis_name, text_labels=False)`

**Inputs**  
- `df: pd.DataFrame` — from `analyze_sentence_local_axes` or `compare_word_on_axes`.  
- `axis_name: str` — column to plot.  
- `text_labels: bool` — show token labels.

**Behaviour**  
- **1-D horizontal scatter** of token projections on the chosen axis.  
- Adds vertical line at 0.

**Returns**  
- `matplotlib.figure.Figure`.

---

### `compare_word_on_axes(sentence, axes_defs)`

**Inputs**  
- `sentence: str`.  
- `axes_defs: Dict[str, Dict[str, List[str]]]` — map `axis_name → {"pos":[...], "neg":[...]}`.

**Behaviour**  
- Tokenizes, projects every token onto **each axis**, returns tidy DataFrame.

**Returns**  
- `pandas.DataFrame` with columns:  
  `token`, `<axis>_raw`, `<axis>_scaled`.

---

### `plot_local_axes_strategies(sentence, strategies, top_n=4)`

**Inputs**  
- `sentence: str`.  
- `strategies: Dict[str, Callable]` — strategy name → axis-finder function.  
- `top_n: int` — how many extreme tokens to label.

**Behaviour**  
- Runs each strategy on the sentence, plots **side-by-side 1-D projections** for comparison.

**Returns**  
- `matplotlib.figure.Figure`.

---

### `auto_axis_from_sentence(sentence)`

**Inputs**  
- `sentence: str`.

**Behaviour**  
- Quick **single-axis** generation: picks **farthest from centroid** as *pos*, **closest** as *neg*.

**Returns**  
- `Dict[str, List[str]]` — `{"pos": [...], "neg": [...]}`.

---

### `demo_sentence_pos_viz(sentence)`

**Inputs**  
- `sentence: str`.

**Behaviour**  
- **One-liner demo**: tokenizes, uses prototype axes, prints DataFrame, and shows **4-zone plot**.

**Returns**  
- None (prints & displays).

### Example

```python
sentence = "The cat quickly chased the tiny mouse in the quiet garden"
tokens   = local_axes._tokenize(sentence)
print("Tokens:", tokens)

# ------------------------------------------------------------------
# 1. Axis-picking strategies on raw tokens
# ------------------------------------------------------------------
print("\n=== Strategy-based pole selection ===")
for name, func in {
    "PCA":        local_axes.pick_pos_neg_PCA,
    "MaxDist":    local_axes.pick_pos_neg_max_distance,
    "CenterExt":  local_axes.pick_pos_neg_center_extremes,
    "KMeans":     local_axes.pick_pos_neg_kmeans
}.items():
    poles = func(tokens)
    print(f"{name:10} -> {poles}")

# ------------------------------------------------------------------
# 2. Quick comparison plot of the four strategies
# ------------------------------------------------------------------
local_axes.plot_local_axes_strategies(
    sentence,
    strategies={
        "PCA":        local_axes.pick_pos_neg_PCA,
        "MaxDist":    local_axes.pick_pos_neg_max_distance,
        "CenterExt":  local_axes.pick_pos_neg_center_extremes,
        "KMeans":     local_axes.pick_pos_neg_kmeans
    },
    top_n=3
)

# ------------------------------------------------------------------
# 3. One-liner automatic axis + sentence-level pipeline
# ------------------------------------------------------------------
print("\n=== Auto axis from sentence ===")
auto_axis = local_axes.auto_axis_from_sentence(sentence)
print(auto_axis)

# ------------------------------------------------------------------
# 4. Direction builders from curated examples
# ------------------------------------------------------------------
print("\n=== Direction builders ===")
dir_pca      = local_axes.build_local_axis_pca(tokens)
dir_import   = local_axes.build_importance_axis(tokens)[0]
dir_explicit = local_axes.build_direction_from_examples(
    ["fast","quick"], ["slow","sluggish"]
)
print("PCA axis shape:", dir_pca.shape)
print("Importance axis shape:", dir_import.shape)
print("Explicit axis shape:", dir_explicit.shape)

# ------------------------------------------------------------------
# 5. Analyze & plot local PCA + importance axes
# ------------------------------------------------------------------
df_local = local_axes.analyze_sentence_local_axes(sentence)
print("\nLocal axes DataFrame:")
print(df_local.head())

local_axes.plot_local_axes(df_local, title="PCA & Importance axes")

# ------------------------------------------------------------------
# 6. POS-like 4-zone demo with prototype axes
# ------------------------------------------------------------------
prototypes = local_axes.pos_axes_prototypes()
print("\nPrototype axes:", prototypes)

df_4zone = local_axes.analyze_sentence_pos_axes(sentence, axes_defs=prototypes)
print("\n4-zone DataFrame:")
print(df_4zone.head())

local_axes.plot_pos_axes_4zones(
    df_4zone, prototypes, title="POS-like 4-zone plot", annotate=True
)

# ------------------------------------------------------------------
# 7. Compare multiple axes on the same tokens
# ------------------------------------------------------------------
custom_axes = {
    "size":   {"pos": ["big","huge"], "neg": ["tiny","small"]},
    "speed":  {"pos": ["quick","fast"], "neg": ["slow","sluggish"]}
}

df_compare = local_axes.compare_word_on_axes(sentence, custom_axes)
print("\nMulti-axis comparison:")
print(df_compare)

# 1-D scatter for the "speed" axis
local_axes.plot_axes_scatter(df_compare, axis_name="speed")

# ------------------------------------------------------------------
# 8. One-liner convenience demo
# ------------------------------------------------------------------
local_axes.demo_sentence_pos_viz("The sleepy cat quietly purred on the warm sofa")
```

---

## biases.py — Bias analysis across semantic dimensions

Utilities to quantify and visualize **systematic bias** or **semantic drift** across **groups of words or sentences** along user-defined axes (e.g., sentiment, dominance, gender stereotypes).

---

### `analyze_biases(target_items, axes_definitions, use_centered=False, scaled=True, model=None)`

**Inputs**  
- `target_items`: nested dict `{group_name: {category: [words_or_sentences]}}`.  
- `axes_definitions`: dict `{axis_name: {"pos": [seed], "neg": [seed]}}`.  
- `use_centered`: if `True`, uses **centered scalar projections** (no explicit direction).  
- `scaled`: if `True`, returns **MinMax-scaled** values in `[-1, 1]`.  
- `model`: optional encoder; falls back to `semgeom.model`.

**Behaviour**  
- Embeds every word/sentence, projects onto each axis, and computes **mean projection** per category.  
- Produces a **summary table** ready for statistical comparison or plotting.

**Returns**  
- Nested dict: `{group_name: {axis_name: {category: mean_score}}}`.

---

### `compute_sentence_axis_values(sentences, axes_definitions, top_n=3, use_centered=False, model=None)`

**Inputs**  
- `sentences`: list of raw sentences.  
- `axes_definitions`: same axis-def dict as above.  
- `top_n`: how many **most-extreme tokens** (by absolute projection) to keep per sentence & axis.  
- `use_centered`, `model` — as above.

**Behaviour**  
- Tokenizes each sentence, projects tokens onto every axis, and keeps the `top_n` most extreme values.  
- Useful for **fine-grained inspection** of which words drive bias in a sentence.

**Returns**  
- `pandas.DataFrame` with columns:  
  `sentence`, `word`, `axis`, `value`, `category (POS/NEG)`, `abs_value`.

---

### `pivot_top_words(df_top)`

**Inputs**  
- `df_top`: DataFrame returned by `compute_sentence_axis_values`.

**Behaviour**  
- Reshapes the long table into a **wide pivot**: index = `(sentence, word)`, columns = axis names, values = projection.

**Returns**  
- `pandas.DataFrame` ready for scatter matrices or regression.

---

### `plot_multiaxis_scatter(pivot_df, axis_pairs=None, titles=None, cmap_pos="green", cmap_neg="red")`

**Inputs**  
- `pivot_df`: wide DataFrame from `pivot_top_words`.  
- `axis_pairs`: list of `(axis1, axis2)` tuples to plot (auto-generated if None).  
- `titles`: optional subplot titles.  
- `cmap_pos`, `cmap_neg`: colors for positive and negative values.

**Behaviour**  
- Creates a **grid of 2-D scatter plots** (`axis1` vs `axis2`).  
- Points colored **green/red** by sign of their projection on the x-axis.  
- Annotates each point with the underlying word for quick inspection.

**Returns**  
- `matplotlib.figure.Figure` (interactive in notebooks).

### Example

```python
# ------------------------------------------------------------------
# 1. Example 1 – High-level bias analysis across word groups
# ------------------------------------------------------------------
target_items = {
    "gender_words": {
        "male":   ["engineer", "doctor", "professor", "driver", "CEO"],
        "female": ["nurse", "teacher", "maid", "librarian", "secretary"]
    }
}

axes_definitions = {
    "sentiment": {"pos": ["happy", "joy", "love"], "neg": ["sad", "hate", "pain"]},
    "dominance": {"pos": ["leader", "power", "control"], "neg": ["follower", "weak", "subordinate"]},
    "complexity": {"pos": ["analytical", "complex", "detailed"], "neg": ["simple", "basic", "easy"]},
}

print("=== Bias analysis (mean projection per category) ===")
results = analyze_biases(
    target_items,
    axes_definitions,
    use_centered=False,
    scaled=True
)

for group, axes in results.items():
    print(f"\nGroup: {group}")
    for axis, cats in axes.items():
        print(f"  Axis: {axis}")
        for cat, val in cats.items():
            print(f"    {cat:12s} -> {val:.4f}")

# ------------------------------------------------------------------
# 2. Example 2 – Sentence-level fine-grained inspection
# ------------------------------------------------------------------
test_sentences = [
    "The brilliant engineer solved the complex problem.",
    "The caring nurse comforted the patients.",
    "The meticulous secretary organized the meeting.",
]

print("\n=== Top words per sentence/axis ===")
df_top = compute_sentence_axis_values(
    test_sentences,
    axes_definitions,
    top_n=3,
    use_centered=False
)
print(df_top.head(12))

# ------------------------------------------------------------------
# 3. Pivot the long table into wide format for scatter
# ------------------------------------------------------------------
pivot_df = pivot_top_words(df_top)
print("\n=== Pivoted table (sample) ===")
print(pivot_df.head())

# ------------------------------------------------------------------
# 4. Multi-axis scatter grid
# ------------------------------------------------------------------
print("\n=== Multi-axis scatter (close window to continue) ===")
plot_multiaxis_scatter(
    pivot_df,
    axis_pairs=[("sentiment", "dominance"), ("dominance", "complexity")],
    titles=["Sentiment vs Dominance", "Dominance vs Complexity"]
)
```

---

## semantic_field.py — Geometry of semantic neighborhoods

Functions to build a local semantic field around a target word using a candidate vocabulary, project neighbors on multiple axes, compute convex hulls and alpha-shapes, run PCA for visualization, bootstrap hull areas, and compare fields across multiple targets.

---

### `semantic_field(target, candidate_vocab, axes_definitions, model=None, top_k=30, use_centered=True)`

**Inputs**  
- `target`: the word whose semantic field you want to explore.  
- `candidate_vocab`: iterable of candidate neighbor words.  
- `axes_definitions`: dict `{axis_name: {"pos":[...], "neg":[...]}}` for semantic projections.  
- `model`: optional encoder; falls back to `semgeom.model`.  
- `top_k`: how many nearest neighbors to keep.  
- `use_centered`: if `True`, uses centered scalar projections.

**Behaviour**  
- Finds the `top_k` most similar neighbors.  
- Computes **embeddings**, **axis projections**, **PCA** 2-D coordinates, and **similarity to target**.  
- Returns a tidy `pandas.DataFrame` ready for further analysis or plotting.

**Returns**  
- `DataFrame` with columns: `word`, `similarity_to_target`, `pc1`, `pc2`, and one column per axis projection.

---

### `plot_semantic_field(df, target_word, axes_to_plot=None, show_table=True, top_n_table=15)`

**Inputs**  
- `df`: DataFrame returned by `semantic_field`.  
- `target_word`: used for highlighting in plots.  
- `axes_to_plot`: list of axis names to color by (defaults to all).  
- `show_table`: print the top-N neighbors table inline.  
- `top_n_table`: how many rows to display.

**Behaviour**  
- **Scatter plot** of neighbors in PCA space, colored by chosen axis projections.  
- Highlights the **target word** in gold.  
- Optionally prints a ranked table.

**Returns**  
- None (displays figure).

---

### `find_top_k_neighbors(target, candidate_vocab, model=None, top_k=50, exclude_self=True)`

**Inputs**  
- `target`: query word.  
- `candidate_vocab`: iterable of potential neighbors.  
- `model`, `top_k`, `exclude_self` — as above.

**Behaviour**  
- Computes cosine similarities and returns the **top-k** most similar words.

**Returns**  
- `List[Tuple[str, float]]` — `(word, similarity)` sorted descending.

---

### `convex_hull_area_2d(coords2d)`

**Inputs**  
- `coords2d`: `np.ndarray (N,2)` — 2-D coordinates.

**Behaviour**  
- Computes the **convex hull** of the point cloud.  
- Returns area and hull object.

**Returns**  
- `Tuple[float, Optional[ConvexHull], np.ndarray]` — `(area, hull_obj, hull_coords)`.

---

### `alpha_shape_area(coords2d, alpha=None)`

**Inputs**  
- `coords2d`: `np.ndarray (N,2)`.  
- `alpha`: optional α-parameter (auto-optimized if `None`).

**Behaviour**  
- Builds an **α-shape** (concave hull) using `alphashape` if available, else falls back to convex hull.

**Returns**  
- `Tuple[Optional[float], Optional[Any], Optional[float]]` — `(area, polygon_obj, alpha_used)`.

---

### `coverage_fraction_polygon(polygon_or_coords, candidate_coords2d)`

**Inputs**  
- `polygon_or_coords`: a `shapely.Polygon`, hull coordinates, or similar.  
- `candidate_coords2d`: `np.ndarray (M,2)` — points to test.

**Behaviour**  
- Computes the **fraction of candidate points** that fall **inside** the polygon.

**Returns**  
- `float` — coverage ratio in `[0, 1]`.

---

### `plot_field_result(result, target, show_hull=True, show_alpha=True)`

**Inputs**  
- `result`: dict output from `analyze_targets_geometry`.  
- `target`: target word (title).  
- `show_hull`, `show_alpha`: toggle convex hull / α-shape edges.

**Behaviour**  
- Plots the 2-D field points, optionally overlaying hull and α-shape edges.

**Returns**  
- None (displays figure).

---

### `plot_combined_fields(results, targets, show_hull=True, show_alpha=True, figsize=(9,9))`

**Inputs**  
- `results`: dict `{target: result_dict}` from `analyze_targets_geometry`.  
- `targets`: list of target words to overlay.  
- `show_hull`, `show_alpha`, `figsize`: as above.

**Behaviour**  
- **Overlay** all target fields in one plot, each in a different color.  
- Draws convex hulls (solid) and α-shapes (dashed) if available.

**Returns**  
- None (displays figure).

---

### `analyze_targets_geometry(targets, candidate_vocab, model=None, k=50, pca_n=2, use_alpha=True, bootstrap_ngroups=200, rng_seed=42)`

**Inputs**  
- `targets`: list of target words.  
- `candidate_vocab`: iterable of possible neighbors.  
- `model`, `k`, `pca_n`, `use_alpha`, `bootstrap_ngroups`, `rng_seed` — configuration knobs.

**Behaviour**  
- End-to-end workflow:  
  1. Collect top-k neighbors per target.  
  2. Fit **shared PCA** space.  
  3. Compute **convex hull area**, **α-shape area**, **coverage**, and **bootstrap** hull-area distribution.  
  4. Optional **representational sparsity** via NNLS over hull vertices.

**Returns**  
- `Tuple[Dict[str, Any], Dict[Tuple[str,str], Any], pd.DataFrame]`  
  - `results`: per-target dict with embeddings, coords, hulls, areas, bootstrap stats.  
  - `pairwise`: target-pair comparisons (Hausdorff, intersection).  
  - `summary_df`: concise metrics overview.

### Example

```python
# --------------------------------------------------
# 1.  Build a single semantic field
# --------------------------------------------------
candidate_vocab = [
    "cat","kitten","feline","dog","puppy","canine","wolf","fox","lion","tiger",
    "mouse","rat","squirrel","hamster","horse","cow","sheep","goat",
    "pet","domestic","feral","stray","companion","hunter","predator","prey",
    "cute","fluffy","playful","aggressive","friendly","fast","slow","wild","tame","adorable","scary"
]

axes_definitions = {
    "wild_domestic": {"pos": ["wild","feral","untamed"], "neg": ["domestic","pet","tame"]},
    "cute_scary":    {"pos": ["cute","adorable","sweet"], "neg": ["scary","frightening","terrifying"]},
    "fast_slow":     {"pos": ["fast","quick","speedy"], "neg": ["slow","sluggish","lethargic"]}
}

print("=== semantic_field demo ===")
df_field = semgeom.semantic_field(
    "cat", candidate_vocab, axes_definitions,
    top_k=25, use_centered=True
)
print(df_field.head())

# --------------------------------------------------
# 2.  Visualise the field
# --------------------------------------------------
semgeom.plot_semantic_field(
    df_field,
    target_word="cat",
    axes_to_plot=list(axes_definitions.keys()),
    show_table=True,
    top_n_table=10
)

# --------------------------------------------------
# 3.  Top-k neighbours utility
# --------------------------------------------------
print("\n=== find_top_k_neighbors demo ===")
neighbors = semgeom.find_top_k_neighbors(
    "dog", candidate_vocab, top_k=8
)
print(neighbors)

# --------------------------------------------------
# 4.  Geometry helpers on raw coordinates
# --------------------------------------------------
coords2d = df_field[["pc1", "pc2"]].values

area_convex, hull_obj, hull_coords = semgeom.convex_hull_area_2d(coords2d)
print("\nConvex hull area:", area_convex)

area_alpha, alpha_poly, alpha_used = semgeom.alpha_shape_area(coords2d, alpha=None)
print("Alpha-shape area:", area_alpha, "alpha used:", alpha_used)

# --------------------------------------------------
# 5.  Coverage fraction example
# --------------------------------------------------
coverage = semgeom.coverage_fraction_polygon(
    hull_coords,  # convex hull
    coords2d      # same points
)
print("Coverage fraction (self)", coverage)

# --------------------------------------------------
# 6.  Multi-target geometry analysis
# --------------------------------------------------
targets = ["cat", "dog", "tiger", "bird", "human"]

results, pairwise, summary = semgeom.analyze_targets_geometry(
    targets,
    candidate_vocab,
    model=semgeom.model,
    k=20,
    pca_n=2,
    use_alpha=True,
    bootstrap_ngroups=100,
    rng_seed=42
)

print("\n=== Summary table ===")
print(summary)

# --------------------------------------------------
# 7.  Overlay all fields in one figure
# --------------------------------------------------
semgeom.plot_combined_fields(
    results,
    targets,
    show_hull=True,
    show_alpha=True,
    figsize=(9, 9)
)

# --------------------------------------------------
# 8.  Individual field plot for "cat"
# --------------------------------------------------
semgeom.plot_field_result(
    results["cat"],
    target="cat",
    show_hull=True,
    show_alpha=True
)

# --------------------------------------------------
# 9.  Optional: bootstrap hull-area histogram
# --------------------------------------------------
areas = results["cat"]["bootstrap_areas"]
if areas is not None and len(areas):
    plt.figure(figsize=(6, 3))
    plt.hist(areas, bins=30, alpha=0.7, color="skyblue")
    plt.title(f"Bootstrap hull areas for 'cat' (mean={np.mean(areas):.3f})")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.show()
```
---
## topic_coverage.py — Document-level topic / category assignment

This module tokenizes documents, assigns each token to the closest category (or soft-weighted distribution), and provides sentence-level and document-level summaries and plotting functions.

---

### `preprocess_text(text, token_re, stopwords=None, lowercase=True, remove_stopwords=True)`

**Inputs**  
- `text`: raw string.  
- `token_re`: compiled regex for token extraction.  
- `stopwords`: iterable or set of words to drop.  
- `lowercase`, `remove_stopwords`: flags.

**Behaviour**  
- Lower-cases (if requested), applies regex, removes stopwords.  

**Returns**  
- `List[str]` — cleaned tokens.

---

### `assign_tokens_to_categories(tokens, categories, model=None, method="mean", top_k_words=1, soft=False, temperature=1.0)`

**Inputs**  
- `tokens`: list of cleaned tokens.  
- `categories`: dict `{category_name: [seed_words]}`.  
- `model`: optional encoder; falls back to `semgeom.model`.  
- `method`: `"mean"` (centroid) or `"nearest_word"` (best single seed).  
- `soft`: if `True`, returns **soft probability** distributions instead of hard assignments.  
- `temperature`: softmax temperature for soft mode.

**Behaviour**  
- Embeds tokens and category seeds, computes similarity, assigns each token.  
- Returns both **hard counts** and **soft probabilities**.

**Returns**  
- `(pandas.DataFrame, summary_dict)`  
  - DataFrame has columns: `token`, `assigned`, `best_cat`, `best_sim`, `all_sims`, `soft_probs`.  
  - `summary_dict` contains counts, proportions, soft proportions.

---

### `document_topic_coverage(text, categories, token_re, stopwords=None, **kwargs)`

**Inputs**  
- `text`: full raw document.  
- `categories`, `token_re`, `stopwords` — as above.  
- remaining kwargs forwarded to `assign_tokens_to_categories`.

**Behaviour**  
- Convenience wrapper: tokenise → assign → summarise.

**Returns**  
- Same tuple as `assign_tokens_to_categories`.

---

### `document_topic_coverage_by_sentence(text, categories, sent_split_re, token_re, stopwords=None, **kwargs)`

**Inputs**  
- `text`: raw document.  
- `sent_split_re`: regex to split sentences (e.g. r'(?<=[.!?])\s+').  
- remaining args/kwargs as above.

**Behaviour**  
- Splits into sentences, then applies `document_topic_coverage` **per sentence**.

**Returns**  
- List of dictionaries, one per sentence:  
  `{"sentence": str, "df_tokens": DataFrame, **summary_dict}`.

---

### `plot_doc_topic_proportions(summary, title="Doc topic proportions", top_n=12)`

**Inputs**  
- `summary`: dict returned by `document_topic_coverage`.  
- `title`, `top_n`: plot parameters.

**Behaviour**  
- Horizontal **bar chart** of **hard** and (optionally) **soft** category proportions.

**Returns**  
- None (displays figure).

---

### `plot_sentence_level(per_sentence, category_list=None, top_n=6)`

**Inputs**  
- `per_sentence`: list returned by `document_topic_coverage_by_sentence`.  
- `category_list`: restrict shown categories (defaults to observed).  
- `top_n`: how many top categories to include.

**Behaviour**  
- **Stacked bar chart** showing category proportions **per sentence**.

**Returns**  
- None (displays figure).

### Example

```python
# --------------------------------------------------
# 1.  Basic preprocessing helpers
# --------------------------------------------------
TOKEN_RE      = re.compile(r"[A-Za-z0-9\-']+")
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

STOPWORDS = {
    "the","a","an","in","on","at","by","for","with","and","or","but",
    "is","are","was","were","to","of","this","that","these","those"
}

text = """
The zookeeper fed the tiger and the lion. Later the family walked their dog in the park.
A sudden storm brought heavy rain and strong winds.
"""

tokens = semgeom.topic_coverage.preprocess_text(
    text,
    TOKEN_RE,
    stopwords=STOPWORDS,
    lowercase=True,
    remove_stopwords=True
)
print("Clean tokens:", tokens[:20])

# --------------------------------------------------
# 2.  Category setup
# --------------------------------------------------
categories = {
    "animals":      ["dog","cat","tiger","lion","wolf","bird","mouse","elephant"],
    "weather":      ["rain","storm","wind","snow","thunder","cloud","sun"],
    "professions":  ["zookeeper","teacher","doctor","nurse","engineer"],
    "locations":    ["park","zoo","forest","city","garden"]
}

# --------------------------------------------------
# 3.  Token-level assignment
# --------------------------------------------------
print("\n=== assign_tokens_to_categories (hard & soft) ===")
df_tokens, summary = semgeom.topic_coverage.assign_tokens_to_categories(
    tokens,
    categories,
    method="mean",
    soft=True,
    temperature=1.0
)
print(df_tokens.head(10))
print("Hard counts:", summary["hard_counts"])
print("Hard proportions:", {k: f"{v:.2f}" for k, v in summary["hard_proportions"].items() if v})
print("Soft proportions:", {k: f"{v:.2f}" for k, v in summary["soft_proportions"].items() if v})

# --------------------------------------------------
# 4.  Whole-document wrapper
# --------------------------------------------------
print("\n=== document_topic_coverage ===")
df_doc, summary_doc = semgeom.topic_coverage.document_topic_coverage(
    text,
    categories,
    TOKEN_RE,
    stopwords=STOPWORDS,
    method="mean",
    soft=False
)
print("Top 5 rows:")
print(df_doc.head())
semgeom.topic_coverage.plot_doc_topic_proportions(
    summary_doc,
    title="Document-level topic coverage"
)

# --------------------------------------------------
# 5.  Sentence-level wrapper
# --------------------------------------------------
print("\n=== document_topic_coverage_by_sentence ===")
per_sent = semgeom.topic_coverage.document_topic_coverage_by_sentence(
    text,
    categories,
    SENT_SPLIT_RE,
    TOKEN_RE,
    stopwords=STOPWORDS,
    method="mean",
    soft=False
)

for sent_dict in per_sent:
    print("\nSentence:", sent_dict["sentence"])
    props = {k: f"{v:.2f}" for k, v in sent_dict["hard_proportions"].items() if v}
    print("Proportions:", props)

semgeom.topic_coverage.plot_sentence_level(
    per_sent,
    top_n=4
)
```
