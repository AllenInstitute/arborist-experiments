import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: str, title: str = "") -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    normed = np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normed, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="P(pred | true)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _embed(X: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1)
            return reducer.fit_transform(X), "UMAP"
        except Exception:
            pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=0).fit_transform(X), "PCA"


def plot_embedding(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    out_path: str,
    method: str = "umap",
    title: str = "",
) -> None:
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    emb, method_used = _embed(Xs, method)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_classes = len(class_names)
    cmap = plt.get_cmap("tab20" if n_classes > 10 else "tab10")
    for i, name in enumerate(class_names):
        m = y == i
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7, label=name, color=cmap(i % cmap.N))
    ax.set_xlabel(f"{method_used}-1")
    ax.set_ylabel(f"{method_used}-2")
    ax.legend(fontsize=7, loc="best", markerscale=1.5, frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
