"""증강 방법 통일 인터페이스.

지원: ``none`` / ``class_weight`` / ``random_oversampling`` / ``smote`` / ``vae``

모든 증강기는 **train fold만** 받는다. 평가 fold를 넘기는 것은 호출자의 오류이며
감사기가 이를 잡는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..audit.leakage import LeakageAuditor
from ..data.loader import LifelogData
from ..data.schema import (
    CLASS_TO_CODE,
    INTEGER_VALUED_FEATURES,
    NON_NEGATIVE_FEATURES,
)
from ..utils.config import config_hash
from .provenance import SyntheticProvenance

log = logging.getLogger(__name__)

__all__ = ["AugmentationResult", "augment_train_fold", "postprocess_synthetic"]


@dataclass
class AugmentationResult:
    """증강 결과."""

    data: LifelogData
    class_weight: dict[int, float] | None
    n_synthetic: int
    diagnostics: dict
    synthetic_X_raw: pd.DataFrame | None = None   # 원 단위 합성 feature (진단용)
    real_source_X_raw: pd.DataFrame | None = None  # 생성에 쓰인 실제 행 (원 단위)


def _resolve_n_synthetic(cfg: dict, n_real_minority: int, class_counts: dict[int, int], target: int) -> int:
    """생성할 합성행 수를 결정한다.

    ``n_synthetic``          절대 개수 (실험 A: 논문 유도값 4,000)
    ``ratio_to_real``        실제 소수 클래스 행 수의 배수
    ``match_majority``       다수 클래스 행 수에 맞춤
    """
    if cfg.get("n_synthetic") is not None:
        return int(cfg["n_synthetic"])
    if cfg.get("ratio_to_real") is not None:
        return int(round(n_real_minority * float(cfg["ratio_to_real"])))
    if cfg.get("match_majority", False):
        majority = max(class_counts.values())
        return max(0, majority - class_counts.get(target, 0))
    return 0


def postprocess_synthetic(
    X_syn_raw: pd.DataFrame,
    reference_raw: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """생성값의 유효 범위를 처리한다 (assumptions.md D-6).

    Args:
        X_syn_raw: **원 단위** 합성 feature.
        reference_raw: train fold의 실제 원 단위 feature (범위 기준).
        cfg: ``postprocess`` 하위 트리.

    Returns:
        (처리된 프레임, 위반 건수 진단).
    """
    cfg = cfg or {}
    X = X_syn_raw.copy()
    diag: dict[str, object] = {}

    if cfg.get("enforce_nonnegative", True):
        cols = [c for c in NON_NEGATIVE_FEATURES if c in X.columns]
        n_neg = int((X[cols] < 0).to_numpy().sum())
        diag["n_negative_values_clipped"] = n_neg
        diag["negative_rate"] = float(n_neg / max(X[cols].size, 1))
        X[cols] = X[cols].clip(lower=0.0)

    if cfg.get("clip_to_train_range", True):
        lo, hi = reference_raw.min(), reference_raw.max()
        below = (X < lo).to_numpy().sum()
        above = (X > hi).to_numpy().sum()
        diag["n_below_train_min"] = int(below)
        diag["n_above_train_max"] = int(above)
        diag["out_of_range_rate"] = float((below + above) / max(X.size, 1))
        X = X.clip(lower=lo, upper=hi, axis=1)

    if cfg.get("round_integer_valued", False):
        cols = [c for c in INTEGER_VALUED_FEATURES if c in X.columns]
        diag["n_columns_rounded"] = len(cols)
        X[cols] = X[cols].round()
    else:
        diag["n_columns_rounded"] = 0
        diag["note_integer"] = (
            "논문 미보고이므로 정수 반올림을 적용하지 않았다. "
            f"정수형 관측 변수 {len(INTEGER_VALUED_FEATURES)}개에 비정수 값이 생성될 수 있다."
        )
    return X, diag


# --------------------------------------------------------------------------------------
def augment_train_fold(
    train: LifelogData,
    cfg: dict,
    *,
    auditor: LeakageAuditor,
    fold_id: str,
    preprocessor=None,
    seed: int = 42,
    inner_fold_id: str | None = None,
) -> AugmentationResult:
    """train fold를 증강한다.

    Args:
        train: **표준화된** train fold (합성행 없음).
        cfg: config의 ``augmentation`` 트리.
        preprocessor: :class:`FoldPreprocessor`. inverse scaling에 쓴다.

    Returns:
        AugmentationResult.
    """
    method = (cfg or {}).get("method", "none")
    target_name = cfg.get("target_class", "Dem")
    target = CLASS_TO_CODE[target_name]
    counts = {int(c): int((train.y == c).sum()) for c in np.unique(train.y)}

    if method == "none":
        return AugmentationResult(train, None, 0, {"method": "none"})

    if method == "class_weight":
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(train.y)
        w = compute_class_weight("balanced", classes=classes, y=train.y)
        return AugmentationResult(
            train, {int(c): float(v) for c, v in zip(classes, w)}, 0, {"method": "class_weight"}
        )

    minority_mask = train.y == target
    n_real = int(minority_mask.sum())
    if n_real == 0:
        raise ValueError(f"train fold에 {target_name} 기록이 없다")
    real = train.take(minority_mask)

    sub_cfg = dict(cfg.get(method) or {})
    n_syn = _resolve_n_synthetic(sub_cfg, n_real, counts, target)
    if n_syn <= 0:
        raise ValueError(
            f"augmentation.method={method!r}인데 생성할 합성행 수가 0이다. "
            f"augmentation.{method}에 n_synthetic, ratio_to_real 또는 "
            "match_majority=true 중 하나를 설정하라. no-op 증강 후보는 허용하지 않는다."
        )

    log.info(
        "[%s] %s 증강: 실제 %s %d행 (피험자 %d명) -> 합성 %d행",
        fold_id, method, target_name, n_real, len(real.subjects()), n_syn,
    )

    if method == "random_oversampling":
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, n_real, size=n_syn)
        X_syn = real.X.iloc[pick].reset_index(drop=True)
        diag = {"method": "random_oversampling", "n_synthetic": n_syn}
        X_syn_raw = None
    elif method == "smote":
        X_syn, diag = _smote_generate(train, real, target, n_syn, sub_cfg, seed)
        X_syn_raw = None
    elif method == "vae":
        fit_scope = sub_cfg.get("fit_scope", "train_dem_only")
        if fit_scope != "train_dem_only":
            raise ValueError(
                "augmentation.vae.fit_scope는 현재 train_dem_only만 지원한다. "
                f"요청값={fit_scope!r}. all_dem은 split 전 원자료를 별도로 넘겨야 하며, "
                "현재 배선에서 허용하면 설정과 실제 학습 범위가 달라진다."
            )
        auditor.record_vae_fit(
            fold_id,
            subjects=real.subject,
            labels=real.y,
            row_ids=real.row_id,
            expected_label=target,
            n_rows=real.n,
        )
        X_syn, X_syn_raw, diag = _vae_generate(
            real,
            cfg,
            sub_cfg,
            n_synthetic=n_syn,
            preprocessor=preprocessor,
            seed=seed,
            fold_id=fold_id,
        )
    else:
        raise ValueError(f"unknown augmentation method {method!r}")

    auditor.record_synthetic(
        fold_id, source_subjects=real.subject, n_rows=len(X_syn), target="train"
    )

    prov = SyntheticProvenance.create(
        source_class=target_name,
        generator=method,
        generator_seed=seed,
        generator_config_hash=config_hash(sub_cfg),
        source_subjects=real.subject,
        n_source_rows=n_real,
        source_outer_fold=fold_id,
        source_inner_fold=inner_fold_id,
    )
    out = train.append_synthetic(X_syn, target, prov.to_frame(len(X_syn)))
    diag["n_source_subjects"] = len(real.subjects())
    diag["n_source_rows"] = n_real
    diag["source_subject_hash"] = prov.source_subject_hash
    return AugmentationResult(
        out, None, len(X_syn), diag,
        synthetic_X_raw=X_syn_raw,
        real_source_X_raw=(
            preprocessor.inverse_transform_features(real.X) if preprocessor is not None else None
        ),
    )


def _smote_generate(train, real, target, n_syn, cfg, seed):
    """SMOTE. 이웃 보간이 같은 피험자 안에서만 일어나는지 진단한다."""
    from imblearn.over_sampling import SMOTE

    k = int(cfg.get("k_neighbors", 5))
    k = max(1, min(k, len(real.X) - 1))
    counts = {int(c): int((train.y == c).sum()) for c in np.unique(train.y)}
    strategy = {target: counts[target] + n_syn}
    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    Xr, yr = sm.fit_resample(train.X, train.y)
    X_syn = pd.DataFrame(Xr[len(train.X):], columns=train.features).reset_index(drop=True)
    diag = {
        "method": "smote",
        "k_neighbors": k,
        "n_synthetic": len(X_syn),
        "note": (
            "SMOTE는 소수 클래스 이웃 사이를 보간한다. "
            f"현재 fold의 train Dem 피험자는 {len(real.subjects())}명이며, "
            "보간이 사실상 피험자 내부에서 일어날 수 있다 "
            "(synthetic_data_risk.md §3.5)."
        ),
    }
    return X_syn, diag


def _vae_generate(
    real,
    cfg,
    sub_cfg,
    *,
    n_synthetic,
    preprocessor,
    seed,
    fold_id,
):
    """VAE 학습·생성·후처리. 학습 자료는 호출자가 이미 train fold로 제한했다."""
    from .vae import TabularVAE, VAEConfig

    input_space = sub_cfg.get("input_space", "scaled")
    if input_space not in {"raw", "scaled"}:
        raise ValueError(
            f"augmentation.vae.input_space는 'raw' 또는 'scaled'여야 한다: {input_space!r}"
        )
    if input_space == "raw":
        if preprocessor is None:
            raise ValueError("input_space='raw'에는 preprocessor가 필요하다 (inverse scaling)")
        X_fit = preprocessor.inverse_transform_features(real.X).to_numpy()
    else:
        X_fit = real.X.to_numpy()

    vcfg = VAEConfig.from_dict(sub_cfg, input_dim=X_fit.shape[1], seed=seed)
    vae = TabularVAE(vcfg).fit(X_fit, subjects=real.subject)

    gen = vae.sample(n_synthetic, seed=seed)
    gen_df = pd.DataFrame(gen, columns=real.features)

    # 후처리는 항상 원 단위에서 수행한다 (범위·부호 제약이 원 단위로 정의되므로).
    if input_space == "raw":
        gen_raw = gen_df
        reference_raw = preprocessor.inverse_transform_features(real.X)
    else:
        gen_raw = preprocessor.inverse_transform_features(gen_df) if preprocessor else gen_df
        reference_raw = preprocessor.inverse_transform_features(real.X) if preprocessor else real.X

    gen_raw, valid_diag = postprocess_synthetic(gen_raw, reference_raw, sub_cfg.get("postprocess"))

    # 모델 입력 공간으로 되돌린다.
    if input_space == "raw" or preprocessor is None:
        X_syn = gen_raw
        if preprocessor is not None:
            arr = preprocessor.scaler.transform(preprocessor.imputer.transform(gen_raw.to_numpy()))
            X_syn = pd.DataFrame(arr, columns=real.features)
    else:
        arr = preprocessor.scaler.transform(gen_raw.to_numpy())
        X_syn = pd.DataFrame(arr, columns=real.features)

    diag = {
        "method": "vae",
        "n_synthetic": len(X_syn),
        "latent_dim": vcfg.latent_dim,
        "input_space": input_space,
        "beta": vcfg.beta,
        "recon_reduction": vcfg.recon_reduction,
        "kl_reduction": vcfg.kl_reduction,
        "epochs_run": len(vae.log.train_total),
        "stopped_early": vae.log.stopped_early,
        "best_epoch": vae.log.best_epoch,
        "final_train_recon": vae.log.train_recon[-1] if vae.log.train_recon else None,
        "final_train_kl": vae.log.train_kl[-1] if vae.log.train_kl else None,
        "recon_mse_fit_space": vae.log.final_recon_mse_scaled_space,
        "paper_reported_recon_error": 2e-4,
        "paper_recon_error_note": (
            "논문 §5.1의 0.0002는 척도가 미보고다 (report_inconsistencies.md I-15). "
            "위 값과 직접 비교할 수 없다."
        ),
        "validity": valid_diag,
        "posterior": {
            "mu_std_mean": vae.log.posterior_mu_std_mean,
            "mu_std_max": vae.log.posterior_mu_std_max,
            "n_active_latent_units": vae.log.n_active_latent_units,
            "latent_dim": vcfg.latent_dim,
            "prior_mismatch_suspected": vae.log.prior_mismatch_suspected,
        },
        "generation_fidelity": _generation_fidelity(gen_raw, reference_raw),
        "vae_source_subjects": vae.source_subjects,
    }
    return X_syn.reset_index(drop=True), gen_raw.reset_index(drop=True), diag


def _generation_fidelity(synth_raw: pd.DataFrame, real_raw: pd.DataFrame) -> dict:
    """생성 표본의 분산이 실제 소수 클래스 분산을 재현하는지 즉시 검사한다.

    2026-08-03 실행에서 46개 변수 전부가 표준편차 비 0.30 이하로 붕괴했는데도
    이 사실이 결과표에 드러나지 않아 "VAE 증강이 효과 없다"는 결론이
    모델링 결함과 구분되지 않았다. 그래서 증강 시점에 바로 계산해 진단에 남긴다
    (synthetic_data_risk.md §3.1의 "표준편차가 실제의 50% 미만 → 다양성 붕괴" 기준).
    """
    real_std = real_raw.std(ddof=1).replace(0.0, np.nan)
    ratio = (synth_raw.std(ddof=1) / real_std).dropna()
    if ratio.empty:
        return {"n_features": 0, "note": "실제 표준편차가 0이라 비교 불가"}

    median_ratio = float(ratio.median())
    collapsed = sorted(ratio[ratio < 0.5].index.tolist())
    out = {
        "std_ratio_median": median_ratio,
        "std_ratio_min": float(ratio.min()),
        "std_ratio_max": float(ratio.max()),
        "n_features_below_half": len(collapsed),
        "n_features": int(len(ratio)),
        "collapsed_features": collapsed[:10],
        "variance_collapse_suspected": bool(median_ratio < 0.5),
    }
    if out["variance_collapse_suspected"]:
        log.warning(
            "합성자료 분산 붕괴: 변수별 표준편차 비 중앙값 %.3f (< 0.5), "
            "%d/%d개 변수가 실제의 절반 미만이다. 증강이 소수 클래스의 '중심 한 점'만 "
            "복제하고 있어 분류 성능 개선을 기대할 수 없다 "
            "(synthetic_data_risk.md §3.1). VAE 손실 균형(recon/kl reduction·beta)을 점검하라.",
            median_ratio, out["n_features_below_half"], out["n_features"],
        )
    return out
