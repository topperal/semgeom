from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    HAS_MPL = False

def plot_projection_from_dict(projections, title="Semantic Projection"):
    sorted_items = sorted(projections.items(), key=lambda x: x[1])
    labels, values = zip(*sorted_items)

    sns.set(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams['figure.dpi'] = 120

    for i, label in enumerate(labels):
        y_offset_text = 0.3 if i % 2 == 0 else -0.2
        y_offset_point = 0
        plt.plot([values[i], values[i]], [y_offset_point, y_offset_text], color='gray', linestyle='--', linewidth=0.7)
        plt.text(values[i], y_offset_text, label, rotation=45, ha='center',
                 va='bottom' if i % 2 == 0 else 'top', fontsize=9)

    plt.title(title)
    plt.yticks([])
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()
    plt.show()


def plot_3d_projection_interactive(words, feature_words, model, title="3D Semantic Projection"):
    import numpy as np
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    word_vecs = np.array([model.encode(w) for w in words])
    word_labels = words

    pos_words = feature_words["pos"]
    neg_words = feature_words["neg"]

    pos_vecs = np.array([model.encode(w) for w in pos_words])
    neg_vecs = np.array([model.encode(w) for w in neg_words])
    direction = np.mean(pos_vecs[:, None, :] - neg_vecs[None, :, :], axis=(0, 1))
    direction /= np.linalg.norm(direction)

    pca = PCA(n_components=3)
    all_vecs = np.vstack([word_vecs, pos_vecs, neg_vecs])
    pca.fit(all_vecs)
    word_vecs_3d = pca.transform(word_vecs)
    pos_vecs_3d = pca.transform(pos_vecs)
    neg_vecs_3d = pca.transform(neg_vecs)

    pos_center = pos_vecs_3d.mean(axis=0)
    neg_center = neg_vecs_3d.mean(axis=0)
    center = (pos_center + neg_center) / 2
    feature_axis = pos_center - neg_center
    feature_axis /= np.linalg.norm(feature_axis)

    projections = []
    for vec in word_vecs_3d:
        v = vec - center
        scalar_proj = np.dot(v, feature_axis)
        proj = center + scalar_proj * feature_axis
        projections.append(proj)
    projections = np.array(projections)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=word_vecs_3d[:, 0],
        y=word_vecs_3d[:, 1],
        z=word_vecs_3d[:, 2],
        mode='markers+text',
        marker=dict(size=4, color='blue'),
        text=word_labels,
        textposition='top center',
        name='Words',
        hovertemplate='%{text}<extra></extra>'
    ))

    # 🔴 Удлиняем красную ось
    line_length = np.linalg.norm(pos_center - neg_center)
    extended_pos = center + feature_axis * line_length * 2.5
    extended_neg = center - feature_axis * line_length * 5.0

    fig.add_trace(go.Scatter3d(
        x=[extended_neg[0], extended_pos[0]],
        y=[extended_neg[1], extended_pos[1]],
        z=[extended_neg[2], extended_pos[2]],
        mode='lines',
        line=dict(color='red', width=5, dash='dash'),
        name='Feature Axis (extended)'
    ))

    # ⚪ Автоматические подписи с feature_words
    fig.add_trace(go.Scatter3d(
        x=[neg_center[0]],
        y=[neg_center[1]],
        z=[neg_center[2]],
        mode='text+markers',
        text=neg_words[0],
        marker=dict(size=6, color='red'),
        textposition='top right',
        name='NEG'
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos_center[0]],
        y=[pos_center[1]],
        z=[pos_center[2]],
        mode='text+markers',
        text=pos_words[0],
        marker=dict(size=6, color='red'),
        textposition='top right',
        name='POS'
    ))

    # 🔵 Проекции
    for i in range(len(words)):
        fig.add_trace(go.Scatter3d(
            x=[word_vecs_3d[i][0], projections[i][0]],
            y=[word_vecs_3d[i][1], projections[i][1]],
            z=[word_vecs_3d[i][2], projections[i][2]],
            mode='lines',
            line=dict(color='lightblue', width=2),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
