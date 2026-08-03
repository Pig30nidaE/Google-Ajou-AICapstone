"""fold 범위 전처리 파이프라인.

사용자 지시 9의 순서를 코드 구조로 강제한다.

    1. 피험자 ID 기준 fold 분리        (호출자)
    2. 이상치 처리기 fit ← train만
    3. imputer fit       ← train만
    4. scaler fit        ← train만
    5. VAE fit           ← train fold의 실제 Dem만   (augmentation 모듈)
    6. train에만 합성행 추가                          (augmentation 모듈)
    7. 분류기 학습
    8. 평가 피험자에는 학습된 변환만 적용
    9. 평가 피험자에 합성행 절대 미추가

``fit``은 감사기에 자기 fit 범위를 **신고**하며, 감사기가 train 밖 자료를 보았는지 검증한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..audit.leakage import LeakageAuditor
from ..data.loader import LifelogData
from .outliers import OutlierHandler, make_outlier_handler

log = logging.getLogger(__name__)

__all__ = ["FoldPreprocessor", "PreprocessReport"]


@dataclass
class PreprocessReport:
    """fit 시점의 관측치."""

    fold_id: str
    n_fit_rows: int
    n_fit_subjects: int
    outlier: dict
    n_dropped_train: int
    n_clipped_cells: int
    n_imputed_cells: int
    scaler_scope: str
    n_scaler_fit_rows: int


class FoldPreprocessor:
    """한 fold의 이상치·결측·스케일 변환을 보유한다.

    Args:
        cfg: config의 ``preprocessing`` + ``outlier`` 하위 트리.
        auditor: 누수 감사기.
        fold_id: 감사 로그에 쓰이는 fold 식별자.
        seed: 난수 seed.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        auditor: LeakageAuditor,
        fold_id: str,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg or {}
        self.auditor = auditor
        self.fold_id = fold_id
        self.seed = seed

        self.outlier: OutlierHandler = make_outlier_handler(self.cfg.get("outlier"), seed=seed)
        self.imputer = SimpleImputer(
            strategy=(self.cfg.get("preprocessing") or {}).get("impute_strategy", "median")
        )
        self.scaler = StandardScaler()
        self.scaler_scope = (self.cfg.get("preprocessing") or {}).get(
            "scaler_scope", "train_real_only"
        )
        self.feature_names: list[str] = []
        self.report: PreprocessReport | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, train: LifelogData) -> "FoldPreprocessor":
        """train fold 자료에서만 이상치 처리기·imputer를 fit한다.

        scaler는 ``scaler_scope``에 따라 나중에 :meth:`fit_scaler`로 fit한다.
        증강 후 스케일링(논문 §4.2 순서)을 지원하기 위한 분리다.
        """
        self.auditor.check_features(train.X.columns)
        self.feature_names = list(train.X.columns)

        # (2) 이상치 처리기 — train fold 자료에서만
        if self.outlier.name != "none":
            self.auditor.record_fit(
                "outlier_detector", self.fold_id,
                subjects=train.subject, row_ids=train.row_id, n_rows=train.n,
            )
        self.outlier.fit(train.X, train.y)
        res = self.outlier.transform(train.X, train.y)

        # (3) imputer — 이상치 처리 후의 train 자료에서만
        kept = train.take(res.keep_mask)
        X_kept = res.X.loc[res.keep_mask] if self.outlier.name == "percentile" else kept.X
        self.auditor.record_fit(
            "imputer", self.fold_id,
            subjects=kept.subject, row_ids=kept.row_id, n_rows=kept.n,
        )
        self.imputer.fit(X_kept.to_numpy())
        n_imputed = int(np.isnan(X_kept.to_numpy()).sum())

        self.report = PreprocessReport(
            fold_id=self.fold_id,
            n_fit_rows=train.n,
            n_fit_subjects=len(set(train.subject)),
            outlier=self.outlier.describe(),
            n_dropped_train=res.n_dropped,
            n_clipped_cells=res.n_clipped_cells,
            n_imputed_cells=n_imputed,
            scaler_scope=self.scaler_scope,
            n_scaler_fit_rows=0,
        )
        self._fitted = True
        return self

    def apply_outlier(self, data: LifelogData) -> LifelogData:
        """학습된 이상치 규칙을 적용한다 (train에만 쓰는 것을 권장).

        평가 자료에 대해서는 행을 **삭제하지 않는다** — 평가셋에서 행을 지우면
        평가 대상이 바뀌므로. ``clip`` 방식만 적용된다.
        """
        res = self.outlier.transform(data.X, data.y)
        out = data.with_features(res.X)
        return out.take(res.keep_mask)

    def apply_outlier_eval(self, data: LifelogData) -> LifelogData:
        """평가 자료용: clip만 적용하고 행은 유지한다."""
        res = self.outlier.transform(data.X, data.y)
        return data.with_features(res.X)

    def fit_scaler(self, data: LifelogData) -> "FoldPreprocessor":
        """scaler를 fit한다. 범위는 ``scaler_scope``가 결정한다.

        ``train_real_only``      합성행을 제외한 train (실험 B·C 기본)
        ``train_with_synthetic`` 합성행 포함 train
        ``all_data``             전체 데이터 — **누수**. 실험 A에서만 허용된다.
        """
        if self.scaler_scope == "train_real_only":
            fit_on = data.real_only()
        elif self.scaler_scope in {"train_with_synthetic", "all_data"}:
            fit_on = data
        else:
            raise ValueError(f"unknown scaler_scope {self.scaler_scope!r}")

        self.auditor.record_fit(
            "scaler", self.fold_id,
            subjects=fit_on.subject, row_ids=fit_on.row_id, n_rows=fit_on.n,
        )
        self.scaler.fit(fit_on.X.to_numpy())
        if self.report is not None:
            self.report.n_scaler_fit_rows = fit_on.n
        return self

    def transform(self, data: LifelogData) -> LifelogData:
        """impute + scale을 적용한다 (fit은 하지 않는다)."""
        if not self._fitted:
            raise RuntimeError("fit을 먼저 호출하라")
        arr = self.imputer.transform(data.X.to_numpy())
        arr = self.scaler.transform(arr)
        return data.with_features(pd.DataFrame(arr, columns=self.feature_names))

    def inverse_transform_features(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """표준화 공간 -> 원 단위. VAE 생성물의 유효성 검사에 쓴다."""
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return pd.DataFrame(self.scaler.inverse_transform(arr), columns=self.feature_names)

    def describe(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "outlier": self.outlier.describe(),
            "impute_strategy": self.imputer.strategy,
            "scaler_scope": self.scaler_scope,
            "n_features": len(self.feature_names),
        }
