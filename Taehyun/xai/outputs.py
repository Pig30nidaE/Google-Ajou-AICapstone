"""출력 헬퍼: DataFrame 변환 및 플롯 파일 저장

이 모듈의 함수는 시간축이 이미 2D로 축약되었다고 가정 (analyzer에서 처리).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# 지원하는 SHAP 플롯 종류
PLOT_KINDS = ("summary", "bar", "beeswarm", "violin")


def importance_dataframe(
    shap_2d: np.ndarray,
    feature_names: list[str],
    class_label: str | None = None,
) -> pd.DataFrame:
    """feature별 글로벌 importance 반환 (mean |SHAP| 내림차순)

    Parameters
    ----------
    shap_2d
        shape (N, F). 부호 보존된 값이어도 되고, abs_mean으로 집계된 값이어도 됨.
    feature_names
        길이 F.
    class_label
        값이 주어지면 'class' 컬럼으로 추가 (멀티클래스 long-form concat용).

    Returns
    -------
    DataFrame
        컬럼: [class(옵션), feature, mean_abs_shap, mean_signed_shap]
        mean_abs_shap 내림차순 정렬, index reset.
    """
    if shap_2d.ndim != 2:
        raise ValueError(f"Expected 2D shap values, got shape {shap_2d.shape}")
    if shap_2d.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: shap has {shap_2d.shape[1]}, "
            f"feature_names has {len(feature_names)}"
        )

    # 글로벌 importance = N개 샘플에 대한 |SHAP| 평균
    mean_abs = np.mean(np.abs(shap_2d), axis=0)
    # 부호 평균은 방향성 참고용 (양수면 해당 클래스 예측에 기여)
    mean_signed = np.mean(shap_2d, axis=0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    )
    if class_label is not None:
        df.insert(0, "class", class_label)

    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_plots(
    shap_2d: np.ndarray,
    X_2d: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    kinds: Iterable[str] = ("summary", "bar"),
    class_label: str | None = None,
    max_display: int = 20,
) -> list[Path]:
    """선택한 SHAP 플롯들을 out_dir에 저장하고 경로 리스트 반환

    - matplotlib backend는 Agg로 강제 (헤드리스/Colab 환경 호환)
    - 파일명: shap_<kind>[_<class_label>].png
    """
    # 디스플레이 없는 환경 대비 (기존 backend가 있으면 force하지 않음)
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import shap as shap_lib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{class_label}" if class_label is not None else ""
    written: list[Path] = []

    for kind in kinds:
        if kind not in PLOT_KINDS:
            raise ValueError(f"Unknown plot kind: {kind!r}. Choose from {PLOT_KINDS}.")

        plt.figure()
        # shap.summary_plot은 plot_type 인자에 따라 4종 그래프를 모두 그림
        if kind == "summary":
            shap_lib.summary_plot(
                shap_2d, X_2d, feature_names=feature_names,
                show=False, max_display=max_display,
            )
        elif kind == "bar":
            shap_lib.summary_plot(
                shap_2d, X_2d, feature_names=feature_names,
                plot_type="bar", show=False, max_display=max_display,
            )
        elif kind == "beeswarm":
            shap_lib.summary_plot(
                shap_2d, X_2d, feature_names=feature_names,
                plot_type="dot", show=False, max_display=max_display,
            )
        elif kind == "violin":
            shap_lib.summary_plot(
                shap_2d, X_2d, feature_names=feature_names,
                plot_type="violin", show=False, max_display=max_display,
            )

        path = out_dir / f"shap_{kind}{suffix}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(path)

    return written
