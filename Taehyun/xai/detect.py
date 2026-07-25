"""모델 타입 자동 감지

- "tree"           : sklearn 호환 트리 분류기 (LGBM, RF, XGBoost, CatBoost 등)
- "keras_sequence" : tf.keras.Model (Sequential / Functional)

ShapAnalyzer에 model_type을 명시하면 감지 단계는 건너뜀.
"""

from __future__ import annotations

from typing import Any


# 트리 계열 클래스 화이트리스트
# - module path 검사만으로는 누락되는 케이스(예: lightgbm.sklearn.LGBMClassifier) 보완용
TREE_CLASS_NAMES = {
    "LGBMClassifier",
    "LGBMRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
}


def detect_model_type(model: Any) -> str:
    """모델 인스턴스를 보고 "tree" 또는 "keras_sequence" 반환

    - 알 수 없는 타입이면 TypeError
    - 호출자는 ShapAnalyzer(..., model_type=...)로 강제 지정 가능
    """
    cls_name = type(model).__name__

    # 1) 클래스명 화이트리스트 매칭
    if cls_name in TREE_CLASS_NAMES:
        return "tree"

    # 2) 모듈 경로 prefix 매칭 (서드파티 트리 모델 커버)
    mod = type(model).__module__ or ""
    if mod.startswith("lightgbm") or mod.startswith("xgboost") or mod.startswith("catboost"):
        return "tree"
    if mod.startswith("sklearn.ensemble") or mod.startswith("sklearn.tree"):
        return "tree"

    # 3) Keras 모델 duck-type 검사
    if _is_keras_model(model):
        return "keras_sequence"

    raise TypeError(
        f"Cannot auto-detect model_type for {cls_name} (module={mod}). "
        "Pass model_type='tree' or model_type='keras_sequence' explicitly."
    )


def _is_keras_model(model: Any) -> bool:
    """tensorflow를 직접 import하지 않고 Keras 모델 여부 판별

    - MRO를 순회하며 keras/tensorflow/tf_keras 모듈에서 정의된
      Model/Sequential/Functional 베이스를 찾으면 True
    """
    for base in type(model).__mro__:
        mod = base.__module__ or ""
        if mod.startswith("keras.") or mod.startswith("tensorflow.") or mod.startswith("tf_keras."):
            if base.__name__ in {"Model", "Sequential", "Functional"}:
                return True
    return False
