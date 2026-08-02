"""이상치 처리 — none / percentile / isolation_forest.

percentile 방식은 논문 §5.1 서술("각 특성값의 상위 및 하위 10% 범위를 벗어나는 데이터를
제외")의 해석이 갈리므로 두 축으로 분리한다.

``scope``
    ``global``     전체 학습자료 기준 분위수
    ``per_class``  클래스별 학습자료 기준 분위수
``action``
    ``drop_row``   한 변수라도 범위를 벗어나면 행 삭제
    ``clip``       변수별 clipping (행 수 불변)

**중요**: 어느 방식이든 분위수와 IsolationForest는 **fit에 주어진 자료에서만** 학습된다.
호출자가 train fold만 넘기는 책임을 지며, 감사기가 그 범위를 검증한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["OutlierHandler", "make_outlier_handler", "OutlierResult"]


@dataclass
class OutlierResult:
    """transform 결과."""

    keep_mask: np.ndarray          # (n,) 유지할 행
    X: pd.DataFrame                # clip이면 수정된 프레임, 아니면 원본
    n_dropped: int
    n_clipped_cells: int = 0


class OutlierHandler:
    """이상치 처리기 공통 인터페이스."""

    name = "none"

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "OutlierHandler":
        return self

    def transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> OutlierResult:
        return OutlierResult(np.ones(len(X), bool), X, 0)

    def describe(self) -> dict:
        return {"method": self.name}


class NoOutlierHandler(OutlierHandler):
    name = "none"


@dataclass
class PercentileOutlierHandler(OutlierHandler):
    """변수별 분위수 기반 이상치 처리.

    ⚠️ 논문의 q=0.10 + scope=global + action=drop_row 조합은 실측상 잔존율 3.05%로
    논문이 보고한 90%와 맞지 않는다 (report_inconsistencies.md I-1 증거 B).
    이는 구현 오류가 아니라 논문 서술의 산술적 귀결이다.
    """

    q: float = 0.10
    scope: str = "global"      # global | per_class
    action: str = "drop_row"   # drop_row | clip
    name: str = field(default="percentile", init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.q < 0.5:
            raise ValueError(f"q must be in (0, 0.5), got {self.q}")
        if self.scope not in {"global", "per_class"}:
            raise ValueError(f"unknown scope {self.scope!r}")
        if self.action not in {"drop_row", "clip"}:
            raise ValueError(f"unknown action {self.action!r}")
        self._bounds: dict[object, tuple[pd.Series, pd.Series]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "PercentileOutlierHandler":
        self._bounds = {}
        if self.scope == "global":
            self._bounds[None] = (X.quantile(self.q), X.quantile(1.0 - self.q))
        else:
            if y is None:
                raise ValueError("scope='per_class'에는 y가 필요하다")
            for cls in np.unique(y):
                sub = X[y == cls]
                self._bounds[int(cls)] = (sub.quantile(self.q), sub.quantile(1.0 - self.q))
        return self

    def _bounds_for(self, y: np.ndarray | None, n: int):
        if self.scope == "global":
            lo, hi = self._bounds[None]
            return [(np.ones(n, bool), lo, hi)]
        if y is None:
            raise ValueError("scope='per_class'에는 y가 필요하다")
        out = []
        for cls, (lo, hi) in self._bounds.items():
            out.append((y == cls, lo, hi))
        return out

    def transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> OutlierResult:
        if not self._bounds:
            raise RuntimeError("fit을 먼저 호출하라")
        keep = np.ones(len(X), bool)
        Xc = X.copy() if self.action == "clip" else X
        n_clipped = 0
        for mask, lo, hi in self._bounds_for(y, len(X)):
            if not mask.any():
                continue
            block = X.loc[mask]
            if self.action == "drop_row":
                inside = ((block >= lo) & (block <= hi)).all(axis=1).to_numpy()
                keep[mask] = inside
            else:
                clipped = block.clip(lower=lo, upper=hi, axis=1)
                n_clipped += int((clipped.to_numpy() != block.to_numpy()).sum())
                Xc.loc[mask] = clipped
        return OutlierResult(keep, Xc, int((~keep).sum()), n_clipped)

    def describe(self) -> dict:
        return {"method": "percentile", "q": self.q, "scope": self.scope, "action": self.action}


@dataclass
class IsolationForestOutlierHandler(OutlierHandler):
    """다변량 이상치 탐지 (논문 §4.2·그림 1).

    ``contamination=0.1``은 정의상 점수 하위 10%를 이상치로 표시하므로
    전체 잔존율이 정확히 90%가 된다 — 논문이 보고한 10,964 / 12,183 = 89.994%와 정합한다
    (report_inconsistencies.md I-1 증거 A).
    """

    contamination: float = 0.1
    n_estimators: int = 100
    max_samples: object = "auto"
    random_state: int = 42
    name: str = field(default="isolation_forest", init=False)

    def __post_init__(self) -> None:
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "IsolationForestOutlierHandler":
        from sklearn.ensemble import IsolationForest

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        ).fit(X.to_numpy())
        return self

    def transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> OutlierResult:
        if self._model is None:
            raise RuntimeError("fit을 먼저 호출하라")
        pred = self._model.predict(X.to_numpy())
        keep = pred == 1
        return OutlierResult(keep, X, int((~keep).sum()))

    def describe(self) -> dict:
        return {
            "method": "isolation_forest",
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }


def make_outlier_handler(cfg: dict, *, seed: int = 42) -> OutlierHandler:
    """config에서 이상치 처리기를 만든다."""
    method = (cfg or {}).get("method", "none")
    if method in (None, "none"):
        return NoOutlierHandler()
    if method == "percentile":
        p = (cfg.get("percentile") or {})
        return PercentileOutlierHandler(
            q=float(p.get("q", 0.10)),
            scope=p.get("scope", "global"),
            action=p.get("action", "drop_row"),
        )
    if method == "isolation_forest":
        p = (cfg.get("isolation_forest") or {})
        return IsolationForestOutlierHandler(
            contamination=float(p.get("contamination", 0.1)),
            n_estimators=int(p.get("n_estimators", 100)),
            max_samples=p.get("max_samples", "auto"),
            random_state=int(p.get("random_state", seed)),
        )
    raise ValueError(f"unknown outlier method {method!r}")
