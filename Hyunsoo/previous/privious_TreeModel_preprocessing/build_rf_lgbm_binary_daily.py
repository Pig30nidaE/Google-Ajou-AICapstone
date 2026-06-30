"""
RF/LGBM용 activity/sleep 비연속형 일 단위 CSV 생성.

전체 흐름:
1. train/val 라벨 파일에서 환자별 정답값을 만듭니다.
2. activity와 sleep 원천 데이터를 각각 하루 단위 표로 정리합니다.
3. RF/LGBM에 넣기 어려운 긴 시계열, 날짜, ID, 라벨 컬럼은 제외합니다.
4. 결측률이 높은 feature를 제거하고 남은 결측치는 train 중앙값으로 채웁니다.
5. 최종 CSV를 저장하고, 어떤 기준으로 전처리했는지 guide 문서에 남깁니다.

용어 정리:
- feature: 모델이 입력으로 보는 설명 변수입니다.
- binary_class: 모델이 맞혀야 하는 정답값입니다. 0=정상, 1=인지저하/치매 계열.
- split: 원본 폴더가 Training인지 Validation인지 표시한 참고 정보입니다.

산출물:
- rf_lgbm_activity_discrete.csv
- rf_lgbm_sleep_discrete.csv
- rf_lgbm_discrete_dataset_guide.md
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# ===== 기본 설정 =====
SCRIPT_DIR = Path(__file__).resolve().parent
TIMEZONE = "Asia/Seoul"

# 모델 입력값이 아니라 샘플을 설명하거나 정답을 담는 컬럼입니다.
META = ["patient_id", "sample_date", "split", "binary_class"]
OUTPUTS = {
    "activity": "rf_lgbm_activity_discrete.csv",
    "sleep": "rf_lgbm_sleep_discrete.csv",
    "guide": "rf_lgbm_discrete_dataset_guide.md",
}

# 진단명을 모델이 이해할 수 있는 숫자 라벨로 바꿉니다. 0=정상, 1=인지저하/치매 계열.
LABEL_MAP = {"CN": 0, "NORMAL": 0, "MCI": 1, "DEMENTIA": 1, "DEM": 1, "AD": 1}

# 원본 파일마다 컬럼명이 조금씩 달라질 수 있어서 후보 이름을 준비합니다.
ID_CANDIDATES = ["user_id", "subject_id", "participant_id", "patient_id", "sample_email", "email", "SAMPLE_EMAIL", "EMAIL"]
LABEL_CANDIDATES = ["DIAG_NM", "diagnosis", "diag_nm", "label", "class", "target"]

# RF/LGBM용 하루 단위 feature를 고를 때 제외할 단어입니다.
EXCLUDE_KEYWORDS = [
    "5min", "1min", "timestamp", "datetime", "date", "time", "start", "end",
    "convert(", "using utf8", "email", "id", "period", "diag", "doctor",
    "sample", "split", "class", "label",
]


def args() -> argparse.Namespace:
    """실행할 때 바꿀 수 있는 옵션을 정의합니다."""
    p = argparse.ArgumentParser(description="논문 RF/LGBM 비연속형 변수 처리 설명에 맞춘 재현형 전처리 CSV 생성")
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--missing-threshold", type=float, default=0.50)
    p.add_argument("--no-missing-feature-drop", action="store_true")
    return p.parse_args()


def data_root() -> Path:
    """현재 위치 주변에서 Training/Validation 데이터 폴더가 있는 곳을 찾습니다."""
    roots = [Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd() / "data", SCRIPT_DIR.parent / "data"]
    roots += [p for p in Path.cwd().iterdir() if p.is_dir()]
    for root in dict.fromkeys(p.resolve() for p in roots):
        if (root / "1.Training").exists() and (root / "2.Validation").exists():
            return root
    raise FileNotFoundError("1.Training / 2.Validation 폴더를 찾지 못했습니다.")


DATA_ROOT = data_root()

# 코드에서는 train/val이라고 부르고, 실제 폴더명은 1.Training/2.Validation을 사용합니다.
ROOTS = {"train": DATA_ROOT / "1.Training", "val": DATA_ROOT / "2.Validation"}


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """CSV 파일의 문자 인코딩이 제각각일 수 있어 여러 방식으로 읽어봅니다."""
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV 읽기 실패: {path}") from last_error


def csv_in(folder: Path, stem: str) -> Path:
    """폴더 안에서 이름에 stem이 들어간 CSV 파일 하나를 찾습니다."""
    files = sorted({p for pat in ("*.csv", "*.csv.part*", "*.CSV", "*.CSV.part*") for p in folder.glob(pat)})
    files = [p for p in files if stem.lower() in p.name.lower()] or files
    if not files:
        raise FileNotFoundError(f"CSV 파일을 찾지 못했습니다: {folder}")
    if len(files) > 1:
        print(f"[WARN] 여러 CSV 후보 발견, 첫 번째 사용: {[p.name for p in files]}")
    return files[0]


def pick_col(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """데이터셋마다 다른 컬럼명 중에서 목적에 맞는 컬럼을 자동으로 고릅니다."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        hit = next((c for c in df.columns if cand.lower() in c.lower()), None)
        if hit:
            return hit
    raise KeyError(f"{purpose} 컬럼 자동 탐지 실패. columns={list(df.columns)}")


def local_dt(s: pd.Series) -> pd.Series:
    """원본 시각을 한국 시간 기준 날짜/시간으로 맞춥니다."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(TIMEZONE)


def numeric(s: pd.Series) -> pd.Series:
    """문자처럼 들어온 숫자를 실제 숫자로 바꾸고, 바꿀 수 없는 값은 결측치로 둡니다."""
    return pd.to_numeric(s, errors="coerce")


def is_sequence_like(s: pd.Series) -> bool:
    """값 안에 /, [], {} 등이 자주 보이면 하루 요약값이 아니라 시계열 컬럼으로 판단합니다."""
    sample = s.dropna().astype(str).head(200)
    return False if sample.empty else sample.str.contains(r"[/,\[\]\{\}]", regex=True).mean() > 0.05


def feature_cols(df: pd.DataFrame) -> list[str]:
    """RF/LGBM에 넣을 수 있는 하루 단위 숫자 feature만 골라냅니다."""
    cols = []
    for col in df.columns:
        name = col.lower()
        if name.startswith("_") or name in {"patient_id", "sample_date"}:
            continue
        if any(k in name for k in EXCLUDE_KEYWORDS) or is_sequence_like(df[col]):
            continue
        if numeric(df[col]).notna().any():
            cols.append(col)
    return cols


def load_labels(split: str) -> pd.DataFrame:
    """진단 라벨을 읽어 환자별 0/1 정답값을 만듭니다."""
    df = read_csv(csv_in(ROOTS[split] / "라벨링데이터" / "3.인지기능", "label"))
    df = df[[pick_col(df, ID_CANDIDATES, "환자 ID"), pick_col(df, LABEL_CANDIDATES, "진단 라벨")]]
    df.columns = ["patient_id", "diagnosis"]
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()
    df["binary_class"] = df["diagnosis"].str.upper().map(LABEL_MAP)
    unknown = sorted(df.loc[df["binary_class"].isna(), "diagnosis"].unique())
    assert not unknown, f"알 수 없는 진단 라벨: {unknown}"
    assert (df.groupby("patient_id")["binary_class"].nunique() <= 1).all(), "같은 환자에 서로 다른 라벨이 있습니다."
    return df.sort_values(["patient_id", "diagnosis"]).drop_duplicates("patient_id").reset_index(drop=True)


def load_activity(split: str) -> pd.DataFrame:
    """활동 원천 데이터를 환자-날짜별 하루 요약표로 바꿉니다."""
    df = read_csv(csv_in(ROOTS[split] / "원천데이터" / "1.걸음걸이", "activity"))
    df = df.rename(columns={pick_col(df, ID_CANDIDATES, "환자 ID"): "patient_id"})
    ts = local_dt(df[pick_col(df, ["activity_day_start", "activity_start_time", "timestamp", "datetime", "activity_day_end", "time"], "활동 타임스탬프")])
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["sample_date"] = ts.dt.date

    # 새벽 0~3시는 전날 활동의 연장으로 보고 날짜를 전날로 붙입니다.
    df.loc[ts.dt.hour < 4, "sample_date"] = (ts[ts.dt.hour < 4] - pd.Timedelta(days=1)).dt.date
    df["sample_date"] = df["sample_date"].astype(str)

    feats = feature_cols(df)
    assert feats, "선택된 activity 이산형 피처가 없습니다."
    df[feats] = df[feats].apply(numeric)

    # 평균 성격의 지표는 mean, 시간/횟수/총량 성격의 지표는 sum으로 하루 값을 만듭니다.
    agg = {c: ("mean" if any(k in c.lower() for k in ("score", "average", "efficiency", "met")) else "sum") for c in feats}
    out = df.dropna(subset=["patient_id", "sample_date"]).groupby(["patient_id", "sample_date"], as_index=False).agg(agg)
    assert out.duplicated(["patient_id", "sample_date"]).sum() == 0, "activity 환자-날짜 중복이 있습니다."
    return out


def load_sleep(split: str) -> pd.DataFrame:
    """수면 원천 데이터를 환자-수면종료일별 하루 요약표로 바꿉니다."""
    df = read_csv(csv_in(ROOTS[split] / "원천데이터" / "2.수면", "sleep"))
    df = df.rename(columns={pick_col(df, ID_CANDIDATES, "환자 ID"): "patient_id"})
    end_col = pick_col(df, ["sleep_bedtime_end", "sleep_end_time", "bedtime_end", "end_time", "end"], "수면 종료 시각")
    end_dt = local_dt(df[end_col])
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["sample_date"] = end_dt.dt.date.astype(str)

    if "sleep_duration" in df.columns:
        duration = numeric(df["sleep_duration"])
    else:
        # sleep_duration 컬럼이 없으면 시작/종료 시각 차이로 수면 시간을 계산합니다.
        start_col = pick_col(df, ["sleep_bedtime_start", "sleep_start_time", "bedtime_start", "start_time", "start"], "수면 시작 시각")
        duration = (end_dt - local_dt(df[start_col])).dt.total_seconds()
    df["_duration"] = duration

    # ID, 날짜, 수면시간이 없거나 0초 이하/24시간 초과인 기록은 비정상 기록으로 제외합니다.
    df = df.dropna(subset=["patient_id", "sample_date", "_duration"])
    df = df[(df["_duration"] > 0) & (df["_duration"] <= 24 * 60 * 60)].copy()

    feats = feature_cols(df)
    if "sleep_duration" in df.columns and "sleep_duration" not in feats:
        feats.append("sleep_duration")
    assert feats, "선택된 sleep 이산형 피처가 없습니다."
    df[feats] = df[feats].apply(numeric)

    # 같은 날짜에 여러 수면 기록이 있으면 가장 긴 기록을 대표값으로 사용합니다.
    out = df.sort_values("_duration", ascending=False).drop_duplicates(["patient_id", "sample_date"])
    out = out[["patient_id", "sample_date", *feats]].reset_index(drop=True)
    assert out.duplicated(["patient_id", "sample_date"]).sum() == 0, "sleep 환자-날짜 중복이 있습니다."
    return out


def attach_labels(df: pd.DataFrame, labels: pd.DataFrame, split: str) -> pd.DataFrame:
    """하루 요약 데이터에 환자별 정답 라벨과 split 정보를 붙입니다."""
    out = df.merge(labels[["patient_id", "binary_class"]], on="patient_id", how="inner")
    out["split"] = split
    return out[[*feature_cols(out), *META]]


def build_raw(kind: str, labels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """activity 또는 sleep 데이터를 train/val 모두 처리해 하나의 표로 합칩니다."""
    loader = load_activity if kind == "activity" else load_sleep
    daily = {s: loader(s) for s in ("train", "val")}
    return pd.concat([
        attach_labels(daily["train"], labels["train"], "train"),
        attach_labels(daily["val"], labels["val"], "val"),
    ], ignore_index=True)


def impute(df: pd.DataFrame, drop_features: bool, threshold: float) -> tuple[pd.DataFrame, dict[str, object]]:
    """결측률이 높은 feature를 제거하고 남은 결측치는 train 중앙값으로 채웁니다."""
    feats = [c for c in df.columns if c not in META]
    df = df.copy()
    df[feats] = df[feats].apply(numeric)
    before = int(df[feats].isna().sum().sum())

    # feature 제거 기준도 train만 보고 정합니다. val 정보를 사용하면 평가가 과하게 좋아질 수 있습니다.
    missing_ratio = df.loc[df["split"] == "train", feats].isna().mean()
    dropped = missing_ratio[missing_ratio >= threshold].index.tolist() if drop_features else []
    feats = [c for c in feats if c not in dropped]

    # 논문에 결측치 처리 방식이 명시되어 있지 않아 구현상 선택한 처리입니다.
    # 중앙값은 평균보다 극단값의 영향을 덜 받아 결측치 대체에 자주 쓰입니다.
    medians = df.loc[df["split"] == "train", feats].median(numeric_only=True)
    assert not medians.isna().any(), f"train median 계산 불가 피처: {medians[medians.isna()].index.tolist()}"
    df = df[[*feats, *META]]
    df[feats] = df[feats].fillna(medians)
    after = int(df[feats].isna().sum().sum())
    assert after == 0, "결측치 처리 후에도 결측이 남아 있습니다."
    return df.sort_values(["split", "patient_id", "sample_date"]).reset_index(drop=True), {
        "features_before": len(missing_ratio),
        "features_after": len(feats),
        "missing_before": before,
        "missing_after": after,
        "dropped": dropped,
    }


def validate(df: pd.DataFrame, expected_splits: set[str]) -> None:
    """최종 CSV가 모델 입력으로 쓰기에 안전한지 기본 조건을 검사합니다."""
    feats = [c for c in df.columns if c not in META]
    train_ids = set(df.loc[df["split"] == "train", "patient_id"])
    other_ids = set(df.loc[df["split"] != "train", "patient_id"])
    assert set(df["split"].unique()) == expected_splits, f"split 값 오류: {set(df['split'].unique())}"
    assert df[["patient_id", "sample_date", "binary_class"]].notna().all().all(), "필수 메타 컬럼에 결측이 있습니다."
    assert not (train_ids & other_ids), "train과 val 간 patient_id 중복이 있습니다."
    assert df.duplicated(["patient_id", "sample_date"]).sum() == 0, "patient_id + sample_date 중복이 있습니다."
    assert df["binary_class"].isin([0, 1]).all(), "binary_class가 0/1이 아닙니다."
    assert not [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])], "비수치형 feature가 남아 있습니다."


def save_checked(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """CSV를 저장한 뒤 다시 읽어 저장 과정에서 행/열이 깨지지 않았는지 확인합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    reloaded = pd.read_csv(path)
    assert reloaded.shape == df.shape, f"저장/재로드 후 shape 불일치: {path}"
    assert reloaded.duplicated(["patient_id", "sample_date"]).sum() == 0, f"재로드 중복 발견: {path}"
    return reloaded


def summary(df: pd.DataFrame, miss: dict[str, object]) -> dict[str, object]:
    """저장 결과를 사람이 확인하기 쉬운 통계로 요약합니다."""
    feats = [c for c in df.columns if c not in META]
    train_ids = set(df.loc[df["split"] == "train", "patient_id"])
    other_ids = set(df.loc[df["split"] != "train", "patient_id"])
    return {
        "shape": df.shape,
        "rows": df["split"].value_counts().sort_index(),
        "patients": df.groupby("split")["patient_id"].nunique(),
        "labels": pd.crosstab(df["split"], df["binary_class"]),
        "patient_overlap": bool(train_ids & other_ids),
        "date_duplicates": int(df.duplicated(["patient_id", "sample_date"]).sum()),
        "features": len(feats),
        **miss,
    }


def print_summary(name: str, path: Path, info: dict[str, object]) -> None:
    """전처리가 끝난 뒤 콘솔에 핵심 요약을 출력합니다."""
    print(f"\n[{name}] {path.name}")
    print(f"shape: {info['shape']}")
    print(f"split별 row 수:\n{info['rows'].to_string()}")
    print(f"split별 patient 수:\n{info['patients'].to_string()}")
    print(f"split별 binary_class 분포:\n{info['labels'].to_string()}")
    print(f"train과 val 간 patient_id 중복 여부: {info['patient_overlap']}")
    print(f"patient_id + sample_date 중복 개수: {info['date_duplicates']}")
    print(f"최종 feature 수: {info['features']}")
    print(f"결측치 처리 전후 결측 개수: {info['missing_before']} -> {info['missing_after']}")
    print(f"결측률 기준 제거 피처 수: {len(info['dropped'])}")


def md_table(df: pd.DataFrame) -> str:
    """guide 문서에 넣을 간단한 Markdown 표를 만듭니다."""
    rows = [[str(x) for x in df.columns], ["---"] * len(df.columns)]
    rows += df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def guide(results: dict[str, tuple[pd.DataFrame, dict[str, object], Path]], drop_features: bool, threshold: float, path: Path) -> None:
    """모델팀이 산출물 구조와 전처리 기준을 확인할 수 있는 설명 문서를 씁니다."""
    def dist(df: pd.DataFrame) -> pd.DataFrame:
        """split별 행 수와 환자 수를 셉니다."""
        return df.groupby("split", as_index=False).agg(rows=("patient_id", "size"), patients=("patient_id", "nunique"))

    def labels(df: pd.DataFrame) -> pd.DataFrame:
        """split별 0/1 라벨 개수를 표로 만듭니다."""
        out = pd.crosstab(df["split"], df["binary_class"]).reset_index()
        for col in (0, 1):
            if col not in out:
                out[col] = 0
        return out[["split", 0, 1]]

    lines = [
        "# RF/LGBM Discrete Dataset Guide",
        "",
        f"- 생성일: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "- 설명: 논문 RF/LGBM 비연속형 변수 처리 설명에 맞춘 재현형 전처리 CSV",
        "",
        "## X에서 제외할 컬럼",
        "",
        "`patient_id`, `sample_date`, `split`, `binary_class`는 모델 입력 X에서 제외합니다. `binary_class`만 y로 사용합니다.",
        "",
        "## 결측치와 split",
        "",
        "- 논문 미기재: 결측치 처리 방식, median imputation, 결측률 50% 이상 피처 제거.",
        "- 구현상 선택: train 기준 median을 fit하고 val에는 transform만 적용.",
        f"- 구현상 선택: 결측률 피처 제거 적용 여부 `{drop_features}`, threshold `{threshold:.2f}`.",
        "- `split` 컬럼은 원본 `1.Training`/`2.Validation` 출처를 표시하는 참고 컬럼입니다.",
        "- 논문 RF/LGBM 재현 모드에서는 `split=train`만 사용해 70/30 row-level stratified split과 Random Search를 수행합니다.",
        "- 임상 일반화 성능 확인 모드에서는 전체 CSV를 사용하되 `patient_id` 기준 GroupKFold/GroupShuffleSplit을 사용합니다.",
        "",
        "## 전처리 원칙",
        "",
        "- RF/LGBM용 이산형 변수만 남깁니다.",
        "- 제외: `5min`, `1min` 연속형 시계열, timestamp/datetime/date/time/start/end, list/array, id/email/sample/label/class/diag 계열 메타 컬럼.",
        "- Sleep `sample_date`는 수면 종료일 기준입니다.",
        "- Activity는 일 단위 집계이며, `score`/`average`/`efficiency`/`met` 계열은 mean, 나머지 시간/횟수/총량 계열은 sum입니다.",
        "",
        "- 원본 `1.Training`은 `train`, `2.Validation`은 `val`로 유지합니다.",
        "",
    ]

    for name, (df, miss, csv_path) in results.items():
        feats = [c for c in df.columns if c not in META]
        lines += [
            f"## {name.title()}",
            "",
            f"- 파일: `{csv_path.name}`",
            f"- rows: {len(df)}",
            f"- patients: {df['patient_id'].nunique()}",
            f"- features: {len(feats)}",
            f"- missing before/after: {miss['missing_before']} -> {miss['missing_after']}",
            f"- dropped features: {', '.join(miss['dropped']) or '없음'}",
            "",
            "### Split Distribution",
            "",
            md_table(dist(df)),
            "",
            "### Label Distribution",
            "",
            md_table(labels(df)),
            "",
            "### Features",
            "",
            "\n".join(f"- `{c}`" for c in feats),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """RF/LGBM 전처리 전체 순서를 실제로 실행하는 진입점입니다."""
    a = args()
    output_dir = a.output_dir.resolve()
    drop_features = not a.no_missing_feature_drop
    print("논문 RF/LGBM 비연속형 변수 처리 설명에 맞춘 재현형 전처리 CSV 생성")
    print(f"DATA_ROOT: {DATA_ROOT}")

    # 1. train/val 라벨을 읽고 환자별 정답값을 준비합니다.
    labels = {"train": load_labels("train"), "val": load_labels("val")}
    assert (pd.concat(labels.values()).groupby("patient_id")["binary_class"].nunique() <= 1).all(), "원본 train/val 라벨 충돌"

    results = {}
    for name in ("activity", "sleep"):
        # 2. activity와 sleep을 각각 하루 단위 원본 표로 정리합니다.
        raw = build_raw(name, labels)

        # 3. 결측 feature 제거/중앙값 대체를 적용합니다.
        final, miss = impute(raw, drop_features, a.missing_threshold)

        # 4. 저장 전에 중복, 라벨, 숫자형 feature 등 기본 품질을 확인합니다.
        validate(final, {"train", "val"})
        csv_path = output_dir / OUTPUTS[name]

        # 5. CSV로 저장하고 다시 읽어 저장 결과를 확인합니다.
        final = save_checked(final, csv_path)
        results[name] = (final, miss, csv_path)
        print_summary(name, csv_path, summary(final, miss))

    # 6. 산출 CSV를 어떻게 만들었는지 설명하는 guide 문서를 생성합니다.
    guide_path = output_dir / OUTPUTS["guide"]
    guide(results, drop_features, a.missing_threshold, guide_path)
    print(f"\nguide saved: {guide_path}")
    print("완료")


if __name__ == "__main__":
    main()
