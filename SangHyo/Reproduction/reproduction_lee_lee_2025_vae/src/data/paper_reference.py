"""논문이 보고한 수치를 그대로 담는 상수 모듈.

사용자 지시 10: 논문의 잘못된 가능성이 있는 수치를 코드에서 임의로 수정하지 않는다.
따라서 표(정본)와 본문(오기 의심)의 값을 **둘 다** 보유하고, 결과표에서 병기한다.
"""

from __future__ import annotations

__all__ = [
    "NOT_REPORTED",
    "TABLE3_COHORT",
    "SECTION51_AFTER_OUTLIER",
    "TABLE5_SPLIT",
    "DERIVED_SPLIT_FACTS",
    "TABLE6_WIDE_DEEP",
    "BODY_TEXT_WIDE_DEEP_CLAIM",
    "FIGURE3_F1",
    "BODY_TEXT_FIGURE3_F1",
    "RECONSTRUCTED_EVAL_SETS",
    "VAE_REPORTED",
    "CLASSIFIER_REPORTED",
]

#: 논문이 해당 조합을 보고하지 않았음을 나타내는 센티널.
#: 결과표에서 공란이 아니라 "미보고(not reported)"로 출력된다.
NOT_REPORTED = "not_reported"


# --------------------------------------------------------------------------------------
# 표 3 — 분류별 라이프로그 개수 (§3.2)  [실측과 완전 일치]
# --------------------------------------------------------------------------------------
TABLE3_COHORT: dict[str, dict[str, int]] = {
    "n_person": {"CN": 111, "MCI": 51, "Dem": 12},
    "n_record": {"CN": 7_737, "MCI": 3_661, "Dem": 785},
}

# --------------------------------------------------------------------------------------
# §5.1 — 이상치 제거 후 행 수
# --------------------------------------------------------------------------------------
SECTION51_AFTER_OUTLIER: dict[str, int] = {"CN": 7_075, "MCI": 3_374, "Dem": 515}

# --------------------------------------------------------------------------------------
# 표 5 — 최종 학습 데이터 (원문 열 이름 그대로: Train / Test / Vaild)
# --------------------------------------------------------------------------------------
TABLE5_SPLIT: dict[str, dict[str, int]] = {
    "Train": {"CN": 5_660, "MCI": 2_699, "Dem": 4_412},
    "Test": {"CN": 707, "MCI": 337, "Dem": 51},
    "Vaild": {"CN": 708, "MCI": 338, "Dem": 52},
}

#: 표 5에서 산술적으로 유도되는 사실 (report_inconsistencies.md I-5).
#: 논문이 직접 보고한 값이 아니므로 "derived"임을 이름으로 표시한다.
DERIVED_SPLIT_FACTS: dict[str, int] = {
    "dem_real_train_rows": 412,      # 515 - 51 - 52
    "dem_synthetic_rows": 4_000,     # 4412 - 412
    "dem_augmentation_multiple": 10,  # 4412 / 412 ≈ 10.7배 (실제 배수는 아래 float)
}
DERIVED_SPLIT_FACTS["dem_synthetic_ratio_to_real"] = 4_000 / 412  # ≈ 9.71

# --------------------------------------------------------------------------------------
# 표 6 — Wide & Deep 증강 전/후  [정본. 내부 산술 정합 검증 완료]
# --------------------------------------------------------------------------------------
TABLE6_WIDE_DEEP: dict[str, dict[str, dict[str, float]]] = {
    "without_augmentation": {
        "CN": {"precision": 0.9107, "recall": 0.9223, "f1": 0.9165},
        "MCI": {"precision": 0.8398, "recall": 0.8373, "f1": 0.8385},
        "Dem": {"precision": 0.9070, "recall": 0.7647, "f1": 0.8298},
        "macro_avg": {"precision": 0.8858, "recall": 0.8414, "f1": 0.8616},
    },
    "with_vae": {
        "CN": {"precision": 0.9332, "recall": 0.8501, "f1": 0.8897},
        "MCI": {"precision": 0.7340, "recall": 0.8843, "f1": 0.8022},
        "Dem": {"precision": 0.9333, "recall": 0.8235, "f1": 0.8750},
        "macro_avg": {"precision": 0.8668, "recall": 0.8526, "f1": 0.8556},
    },
}

#: §5.2 본문의 서술. 표 6과 Dem F1 / macro F1이 **뒤바뀌어** 있다.
#: (report_inconsistencies.md I-3 — 표 6이 정본임을 산술로 확인)
BODY_TEXT_WIDE_DEEP_CLAIM: dict[str, object] = {
    "dem_f1": 0.8556,
    "avg_f1": 0.875,
    "note": (
        "본문 §5.2의 값. 표 6과 Dem F1(0.8750)·macro F1(0.8556)이 교환되어 있다. "
        "표 6은 F1 = 2PR/(P+R)과 macro 평균이 모두 정합하므로 표 6이 정본이다."
    ),
}

# --------------------------------------------------------------------------------------
# 그림 3 — 4개 모델 클래스별 F1 (그림의 표기를 그대로 전사)
# 조건 미표기이나 W&D 값이 표 6의 증강 후와 동일하므로 "증강 후"로 판단 (I-12).
# XGBoost MCI는 그림 0.7501 / 본문 0.7581로 충돌하며, 그림의 macro 0.8103은
# 본문값 0.7581을 쓸 때에만 클래스별 단순 평균과 맞는다.
# --------------------------------------------------------------------------------------
FIGURE3_F1: dict[str, dict[str, float]] = {
    "XGBoost": {"CN": 0.8914, "MCI": 0.7501, "Dem": 0.7816, "macro_avg": 0.8103},
    "DNN": {"CN": 0.8958, "MCI": 0.7770, "Dem": 0.7527, "macro_avg": 0.8085},
    "TabNet": {"CN": 0.8762, "MCI": 0.7485, "Dem": 0.7391, "macro_avg": 0.7879},
    "WideDeep": {"CN": 0.8897, "MCI": 0.8022, "Dem": 0.8750, "macro_avg": 0.8556},
}

BODY_TEXT_FIGURE3_F1: dict[str, dict[str, float]] = {
    "XGBoost": {"MCI": 0.7581},
}

#: 표 6의 precision·recall에서 역산한 평가셋 크기 (report_inconsistencies.md I-4).
#: 두 조건의 N이 다르다 = 동일 평가셋 비교가 아니다.
RECONSTRUCTED_EVAL_SETS: dict[str, dict[str, object]] = {
    "without_augmentation": {
        "n_actual": {"CN": 708, "MCI": 338, "Dem": 51},
        "n_predicted": {"CN": 717, "MCI": 337, "Dem": 43},
        "true_positives": {"CN": 653, "MCI": 283, "Dem": 39},
        "n_total": 1_097,
        "accuracy": 975 / 1_097,
    },
    "with_vae": {
        "n_actual": {"CN": 707, "MCI": 337, "Dem": 51},
        "n_predicted": {"CN": 644, "MCI": 406, "Dem": 45},
        "true_positives": {"CN": 601, "MCI": 298, "Dem": 42},
        "n_total": 1_095,
        "accuracy": 941 / 1_095,
    },
    "note": (
        "증강 전 N=1097, 증강 후 N=1095로 평가셋이 다르다. "
        "각 조건에서 실제 합계 = 예측 합계로 혼동행렬 주변합이 닫힌다."
    ),
}

# --------------------------------------------------------------------------------------
# VAE 보고 설정 (§5.1, 그림 2)
# --------------------------------------------------------------------------------------
VAE_REPORTED: dict[str, object] = {
    "encoder_hidden": (512, 256),
    "decoder_hidden": (256, 512),
    "latent_dim_body_text": 500,   # §5.1
    "latent_dim_figure2": 50,      # 그림 2 (PDF에서 직접 판독)
    "batch_normalization": True,
    "dropout": 0.3,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "loss": "reconstruction + KL (weighted sum, weight not reported)",
    "reconstruction_error": 2e-4,  # 척도 미보고 (I-15)
    "sampling": "z ~ N(0, I) -> decoder",
    "epochs": NOT_REPORTED,
    "batch_size": NOT_REPORTED,
    "beta": NOT_REPORTED,
    "seed": NOT_REPORTED,
    "fit_scope": NOT_REPORTED,     # 전체 Dem인지 train Dem인지 (I-9)
}

# --------------------------------------------------------------------------------------
# 분류기 보고 설정 (§5.1)
# --------------------------------------------------------------------------------------
CLASSIFIER_REPORTED: dict[str, dict[str, object]] = {
    "XGBoost": {
        "objective": "multi:softmax",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": NOT_REPORTED,
        "seed": NOT_REPORTED,
    },
    "DNN": {
        "hidden": (512, 256, 128, 64, 32),
        "activation": "relu",
        "l2_regularization": True,   # 계수 미보고
        "batch_normalization": True,
        "dropout": 0.5,
        "output": "softmax",
        "l2_coefficient": NOT_REPORTED,
        "epochs": NOT_REPORTED,
        "batch_size": NOT_REPORTED,
        "learning_rate": NOT_REPORTED,
    },
    "TabNet": {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": NOT_REPORTED,
        "lambda_sparse": NOT_REPORTED,
        "epochs": NOT_REPORTED,
        "batch_size": NOT_REPORTED,
    },
    "WideDeep": {
        "deep_hidden": (256, 128, 64),
        "activation": "relu",
        "dropout": 0.3,
        "wide_input": NOT_REPORTED,   # 무엇이 들어가는지 전혀 기재 없음 (Q12)
        "epochs": NOT_REPORTED,
        "batch_size": NOT_REPORTED,
        "learning_rate": NOT_REPORTED,
    },
    "_common": {
        "loss": "cross_entropy",
        "optimizer": "adam",
        "early_stopping": NOT_REPORTED,
        "seed": NOT_REPORTED,
    },
}
