"""데이터 계약 (contract).

논문 표 1·2의 변수 목록과 실측 데이터 프로파일을 상수로 보유한다.
로더가 매 실행마다 이 계약을 검증하며, 어긋나면 조용히 고치지 않고 ``SchemaError``를 던진다.
(사용자 지시 8: 원 논문과 데이터가 충돌하면 조용히 수정하지 말고 불일치를 문서화하라.)
"""

from __future__ import annotations

import re

__all__ = [
    "SchemaError",
    "PAPER_ACTIVITY_FEATURES",
    "PAPER_SLEEP_FEATURES",
    "PAPER_FEATURES",
    "EXCLUDED_ACTIVITY_COLUMNS",
    "EXCLUDED_SLEEP_COLUMNS",
    "FORBIDDEN_COLUMNS",
    "FORBIDDEN_PATTERNS",
    "CLASS_ORDER",
    "CLASS_TO_CODE",
    "CODE_TO_CLASS",
    "DATA_CONTRACT",
    "KNOWN_DUPLICATE_COLUMN_PAIRS",
    "SIGNED_FEATURES",
    "NON_NEGATIVE_FEATURES",
    "NON_INTEGER_FEATURES",
    "INTEGER_VALUED_FEATURES",
    "SCORE_FEATURES",
    "SYNTHETIC_SUBJECT_SENTINEL",
    "assert_no_forbidden_features",
]


class SchemaError(RuntimeError):
    """데이터가 논문·실측 계약과 어긋날 때."""


# --------------------------------------------------------------------------------------
# 논문 표 1 — 활동 변수 22개 (실측: 컬럼명 100% 일치)
# --------------------------------------------------------------------------------------
PAPER_ACTIVITY_FEATURES: tuple[str, ...] = (
    "activity_average_met",
    "activity_cal_active",
    "activity_cal_total",
    "activity_daily_movement",
    "activity_high",
    "activity_inactive",
    "activity_low",
    "activity_medium",
    "activity_met_min_high",
    "activity_met_min_inactive",
    "activity_met_min_low",
    "activity_met_min_medium",
    "activity_rest",
    "activity_score",
    "activity_score_meet_daily_targets",
    "activity_score_move_every_hour",
    "activity_score_recovery_time",
    "activity_score_stay_active",
    "activity_score_training_frequency",
    "activity_score_training_volume",
    "activity_steps",
    "activity_total",
)

# 논문 §3.1이 제외한다고 서술한 활동 컬럼.
EXCLUDED_ACTIVITY_COLUMNS: tuple[str, ...] = (
    "activity_class_5min",          # 5분 활동 로그
    "activity_day_end",             # 활동 종료 시간
    "activity_day_start",           # 활동 시작 시간
    "activity_met_1min",            # 1분 MET 로그
    "activity_inactivity_alerts",   # 비활동 알람 횟수
    "activity_non_wear",            # 미착용 시간
)

# --------------------------------------------------------------------------------------
# 논문 표 2 — 수면 변수 24개 (실측: 컬럼명 100% 일치)
# --------------------------------------------------------------------------------------
PAPER_SLEEP_FEATURES: tuple[str, ...] = (
    "sleep_awake",
    "sleep_breath_average",
    "sleep_deep",
    "sleep_duration",
    "sleep_efficiency",
    "sleep_hr_average",
    "sleep_hr_lowest",
    "sleep_light",
    "sleep_midpoint_at_delta",
    "sleep_midpoint_time",
    "sleep_onset_latency",
    "sleep_rem",
    "sleep_restless",
    "sleep_rmssd",
    "sleep_score",
    "sleep_score_alignment",
    "sleep_score_deep",
    "sleep_score_disturbances",
    "sleep_score_efficiency",
    "sleep_score_latency",
    "sleep_score_rem",
    "sleep_score_total",
    "sleep_temperature_delta",
    "sleep_temperature_deviation",
)

# 논문 §3.1은 앞 6개만 예시하고 "등"으로 마무리했다.
# sleep_rmssd_5min·sleep_total은 표 2에 없으므로 제외가 확실하다 (32 - 8 = 24).
EXCLUDED_SLEEP_COLUMNS: tuple[str, ...] = (
    "sleep_bedtime_end",
    "sleep_bedtime_start",
    "sleep_hr_5min",
    "sleep_hypnogram_5min",
    "sleep_is_longest",
    "sleep_period_id",
    "sleep_rmssd_5min",   # 논문 미명시 (unresolved_questions.md Q6)
    "sleep_total",        # 논문 미명시 (unresolved_questions.md Q6)
)

PAPER_FEATURES: tuple[str, ...] = PAPER_ACTIVITY_FEATURES + PAPER_SLEEP_FEATURES


# --------------------------------------------------------------------------------------
# 금지 변수 (사용자 지시 12·13)
# --------------------------------------------------------------------------------------
#: 피험자 식별자 / 타깃 / MMSE·진단 파생. feature 프레임에 절대 들어가면 안 된다.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "EMAIL",
        "SAMPLE_EMAIL",
        "DIAG_NM",
        "DIAG_SEQ",
        "DOCTOR_NM",
        "MMSE_NUM",
        "MMSE_KIND",
        "TOTAL",
        "Q12_TOTAL",
    }
)

#: MMSE 문항(Q01, Q11_1, Q16_3 …)과 CONVERT 파생 컬럼을 잡아내는 패턴.
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Q\d+(_\d+)?$", re.IGNORECASE),
    re.compile(r"^CONVERT\(", re.IGNORECASE),
    re.compile(r"mmse", re.IGNORECASE),
    re.compile(r"diag", re.IGNORECASE),
    re.compile(r"email", re.IGNORECASE),
)


def assert_no_forbidden_features(columns) -> None:
    """feature 컬럼 목록에 금지 변수가 없는지 확인한다.

    Raises:
        SchemaError: 금지 변수가 하나라도 포함된 경우.
    """
    bad: list[str] = []
    for col in columns:
        if col in FORBIDDEN_COLUMNS:
            bad.append(f"{col} (금지 목록)")
            continue
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(col):
                bad.append(f"{col} (패턴 {pat.pattern})")
                break
    if bad:
        raise SchemaError(
            "입력 feature에 금지 변수가 포함되었다 (사용자 지시 12·13):\n  - "
            + "\n  - ".join(bad)
        )


# --------------------------------------------------------------------------------------
# 클래스
# --------------------------------------------------------------------------------------
CLASS_ORDER: tuple[str, ...] = ("CN", "MCI", "Dem")
CLASS_TO_CODE: dict[str, int] = {name: i for i, name in enumerate(CLASS_ORDER)}
CODE_TO_CLASS: dict[int, str] = {i: name for name, i in CLASS_TO_CODE.items()}

#: 합성행의 subject 자리에 들어가는 센티널. 실제 피험자 ID를 절대 부여하지 않는다.
#: (사용자 지시 15·16, synthetic_data_risk.md §2)
SYNTHETIC_SUBJECT_SENTINEL = "__SYNTHETIC__"


# --------------------------------------------------------------------------------------
# 실측 데이터 계약 (2026-08-02, Training+Validation 합본 기준)
# --------------------------------------------------------------------------------------
DATA_CONTRACT: dict[str, object] = {
    "n_rows": 12_183,          # 논문 §3.1 "12,183건"과 일치
    "n_subjects": 174,         # 논문 §3.1 "174명"과 일치
    "n_features": 46,          # 활동 22 + 수면 24
    "n_activity_columns": 28,  # 논문 §3.1 "활동성 변수 28개"와 일치
    "n_sleep_columns": 32,     # 논문 §3.1은 33개라고 기재 → report_inconsistencies.md I-13
    "records_per_class": {"CN": 7_737, "MCI": 3_661, "Dem": 785},
    "subjects_per_class": {"CN": 111, "MCI": 51, "Dem": 12},
    "n_missing_cells": 0,
    "n_duplicate_rows": 0,
}

#: 실측상 원소 단위로 완전히 동일한 컬럼쌍. 논문의 "46개 변수"는 실질 45개다.
#: (paper_data_mapping.md §4, unresolved_questions.md Q7)
KNOWN_DUPLICATE_COLUMN_PAIRS: tuple[tuple[str, str], ...] = (
    ("sleep_temperature_delta", "sleep_temperature_deviation"),
)

#: 실측상 음수가 존재하는 변수. 나머지 43개는 항상 >= 0이다.
SIGNED_FEATURES: frozenset[str] = frozenset(
    {"sleep_midpoint_at_delta", "sleep_temperature_delta", "sleep_temperature_deviation"}
)
NON_NEGATIVE_FEATURES: tuple[str, ...] = tuple(
    f for f in PAPER_FEATURES if f not in SIGNED_FEATURES
)

#: 실측상 비정수 값을 갖는 변수 5개. 나머지 41개는 정수값만 관측된다.
NON_INTEGER_FEATURES: frozenset[str] = frozenset(
    {
        "activity_average_met",
        "sleep_breath_average",
        "sleep_hr_average",
        "sleep_temperature_delta",
        "sleep_temperature_deviation",
    }
)
INTEGER_VALUED_FEATURES: tuple[str, ...] = tuple(
    f for f in PAPER_FEATURES if f not in NON_INTEGER_FEATURES
)

#: 관측 범위가 [0, 100] 또는 [1, 100]인 점수형 변수. VAE 생성값 검사에 쓴다.
SCORE_FEATURES: tuple[str, ...] = tuple(
    f for f in PAPER_FEATURES if "_score" in f
)
