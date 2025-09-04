# semgeom/topic_coverage.py
"""
Topic coverage analysis:
- preprocess text
- assign tokens to predefined semantic categories (animals, professions, cities, etc.)
- compute document-level and sentence-level coverage
- visualize results
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import semgeom  # for global model slot

# -----------------------
# Model helpers
# -----------------------
def _get_model(model=None):
    if model is not None:
        return model
    if getattr(semgeom, "model", None) is not None:
        return semgeom.model
    raise RuntimeError("No model set. Call semgeom.set_model(...) first or pass model=...")


def _precompute_category_vectors(categories, model):
    """Precompute embeddings and centroids for categories"""
    cat_embs = {}
    cat_means = {}
    for cname, words in categories.items():
        embs = np.vstack([model.encode(w) for w in words])
        cat_embs[cname] = {"words": words, "embeddings": embs}
        mean = embs.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        cat_means[cname] = mean
    return cat_embs, cat_means


# -----------------------
# Preprocess
# -----------------------
def preprocess_text(text, token_re, stopwords=None, lowercase=True, remove_stopwords=True):
    if lowercase:
        text = text.lower()
    tokens = token_re.findall(text)
    if remove_stopwords and stopwords is not None:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


# -----------------------
# Core function
# -----------------------
def assign_tokens_to_categories(tokens, categories, model=None,
                                method="mean", top_k_words=1, ignore_threshold=None,
                                soft=False, temperature=1.0):
    mdl = _get_model(model)
    cat_embs, cat_means = _precompute_category_vectors(categories, mdl)

    if not tokens:
        return pd.DataFrame(), {}

    unique = list(dict.fromkeys(tokens))
    emb_map = {t: mdl.encode(t) for t in unique}

    cat_names = list(categories.keys())
    mean_mat = np.vstack([cat_means[c] for c in cat_names])

    rows = []
    for t in tokens:
        emb = emb_map[t]
        if method == "mean":
            sims = (mean_mat @ emb) / (np.linalg.norm(emb) + 1e-12)
        else:  # nearest_word
            sims = []
            for cname in cat_names:
                embs = cat_embs[cname]["embeddings"]
                sims_to_words = (embs @ emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(emb) + 1e-12)
                sims.append(np.sort(sims_to_words)[-top_k_words:].mean())
            sims = np.array(sims)

        if soft:
            scaled = sims / (temperature + 1e-12)
            ex = np.exp(scaled - np.max(scaled))
            probs = ex / ex.sum()
            probs_dict = {cat_names[i]: float(probs[i]) for i in range(len(cat_names))}
        else:
            probs_dict = None

        best_idx = int(np.argmax(sims))
        best_cat = cat_names[best_idx]
        best_sim = float(sims[best_idx])
        assigned = None if (ignore_threshold and best_sim < ignore_threshold) else best_cat
        all_sims = {cat_names[i]: float(sims[i]) for i in range(len(cat_names))}
        rows.append({"token": t, "assigned": assigned, "best_cat": best_cat,
                     "best_sim": best_sim, "all_sims": all_sims, "soft_probs": probs_dict})

    df = pd.DataFrame(rows)
    hard_counts = df['assigned'].value_counts(dropna=False).to_dict()
    proportions = {k: v/len(df) for k,v in hard_counts.items()}
    if soft:
        sum_probs = {c: 0.0 for c in cat_names}
        for p in df['soft_probs']:
            for c, v in p.items():
                sum_probs[c] += v
        total = sum(sum_probs.values()) + 1e-12
        soft_props = {c: sum_probs[c]/total for c in cat_names}
    else:
        soft_props = None

    summary = {
        "n_tokens": len(df),
        "assigned_count": int(df['assigned'].notna().sum()),
        "hard_counts": hard_counts,
        "hard_proportions": proportions,
        "soft_proportions": soft_props
    }
    return df, summary


# -----------------------
# Wrappers
# -----------------------
def document_topic_coverage(text, categories, token_re, stopwords=None, **kwargs):
    tokens = preprocess_text(text, token_re, stopwords=stopwords,
                             remove_stopwords=kwargs.pop("remove_stopwords", True))
    return assign_tokens_to_categories(tokens, categories, **kwargs)


def document_topic_coverage_by_sentence(text, categories, sent_split_re, token_re, stopwords=None, **kwargs):
    sents = [s.strip() for s in sent_split_re.split(text) if s.strip()]
    results = []
    for s in sents:
        df, summ = document_topic_coverage(s, categories, token_re, stopwords=stopwords, **kwargs)
        results.append({"sentence": s, "df_tokens": df, **summ})
    return results


# -----------------------
# Visualization
# -----------------------
def plot_doc_topic_proportions(summary, title="Doc topic proportions", top_n=12):
    hard = summary['hard_proportions']; soft = summary.get('soft_proportions')
    dfp = pd.DataFrame({
        "category": list(hard.keys()),
        "hard": [hard[k] for k in hard],
        "soft": [soft[k] if soft else np.nan for k in hard]
    })
    dfp = dfp.sort_values("hard", ascending=False).head(top_n)
    plt.figure(figsize=(8, 0.4*len(dfp)+2))
    x = np.arange(len(dfp))
    plt.barh(x-0.15, dfp['hard'], height=0.3, label="hard")
    if soft:
        plt.barh(x+0.15, dfp['soft'], height=0.3, label="soft")
    plt.yticks(x, dfp['category']); plt.gca().invert_yaxis()
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()


def plot_sentence_level(per_sentence, category_list=None, top_n=6):
    if not per_sentence:
        return
    if category_list is None:
        category_list = list(set().union(*[set(x['hard_proportions'].keys()) for x in per_sentence]))
    rows = []
    for s in per_sentence:
        row = {"sentence": s['sentence']}
        for c in category_list:
            row[c] = s['hard_proportions'].get(c, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("sentence")
    chosen = df.sum().sort_values(ascending=False).head(top_n).index
    df = df[chosen]
    df.plot(kind="bar", stacked=True, figsize=(10,0.6*len(df)+3))
    plt.title("Sentence-level topic coverage"); plt.tight_layout(); plt.show()