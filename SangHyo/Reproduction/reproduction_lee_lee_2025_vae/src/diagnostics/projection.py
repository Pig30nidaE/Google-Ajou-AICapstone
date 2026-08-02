"""실제/합성 자료의 2D 투영 시각화 (synthetic_data_risk.md §3.3)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["pca_projection", "umap_projection", "plot_projection"]


def pca_projection(X: np.ndarray, *, n_components: int = 2, seed: int = 42) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=seed).fit_transform(np.asarray(X, float))


def umap_projection(X: np.ndarray, *, n_components: int = 2, seed: int = 42) -> np.ndarray | None:
    """UMAP. 미설치 시 None을 반환한다 (선택 의존성)."""
    try:
        import umap  # type: ignore
    except ImportError:
        log.info("umap-learn 미설치 — UMAP 투영을 건너뛴다")
        return None
    return umap.UMAP(n_components=n_components, random_state=seed).fit_transform(np.asarray(X, float))


def plot_projection(
    coords: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "projection",
) -> Path | None:
    """투영 산점도를 저장한다. matplotlib 미설치면 None."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        log.info("matplotlib 미설치 — 그림을 건너뛴다")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in pd.unique(labels):
        m = labels == name
        marker = "x" if "synthetic" in str(name).lower() else "o"
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.55, label=str(name), marker=marker)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def build_projection_frame(
    real_dem: pd.DataFrame,
    synth_dem: pd.DataFrame,
    other: dict[str, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """투영 입력과 라벨을 만든다."""
    frames, labels = [real_dem, synth_dem], (
        ["real_Dem"] * len(real_dem) + ["synthetic_Dem"] * len(synth_dem)
    )
    for name, df in (other or {}).items():
        frames.append(df)
        labels += [name] * len(df)
    return pd.concat(frames, ignore_index=True).to_numpy(float), np.array(labels)
