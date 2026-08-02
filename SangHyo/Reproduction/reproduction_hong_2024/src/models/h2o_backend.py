"""Optional H2O AutoML backend.

Hong et al. §4.2 names the tool and the version explicitly -- "H2O, a Python (H2O
version 3.46.0.1) AutoML library" -- so this backend exists to follow the paper
where the environment allows it.  It is never required: ``run.py`` falls back to
the scikit-learn / XGBoost path whenever ``h2o`` is not importable, and the
result records which backend actually ran.

Two cautions are wired in rather than left to the reader:

1. AutoML must only ever see the training fold.  ``fit_automl`` takes one frame,
   and the engine passes the training split; there is no validation-frame
   argument that could accidentally be handed the outer test set.
2. AutoML picks a model family *and* its hyperparameters from the data it is
   given, so under experiment C it belongs inside the inner CV, not around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REQUIRED_VERSION = "3.46.0.1"


class H2ONotAvailable(RuntimeError):
    """Raised when the H2O backend is requested but cannot be used."""


@dataclass
class H2OConfig:
    max_models: int = 20
    max_runtime_secs: int = 600
    include_algos: tuple[str, ...] = ("GLM", "DRF", "XGBoost", "GBM")
    nfolds: int = 5
    seed: int = 42
    sort_metric: str = "AUC"

    def describe(self) -> dict[str, Any]:
        return dict(self.__dict__)


def is_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("h2o") is not None


def installed_version() -> str | None:
    try:
        import h2o

        return str(getattr(h2o, "__version__", None))
    except Exception:  # pragma: no cover - only on a broken install
        return None


def ensure_available(*, require_exact_version: bool = False) -> str:
    if not is_available():
        raise H2ONotAvailable(
            "h2o is not installed. Either `pip install h2o==3.46.0.1` or set "
            "models.baseline_backend: sklearn in the config."
        )
    version = installed_version()
    if require_exact_version and version != REQUIRED_VERSION:
        raise H2ONotAvailable(
            f"the paper used H2O {REQUIRED_VERSION}, this environment has {version}. "
            "Set models.h2o.require_exact_version: false to proceed anyway; the "
            "result will record the version that actually ran."
        )
    return version or "unknown"


def start(*, max_mem_size: str = "4G", nthreads: int = -1) -> None:
    import h2o

    h2o.init(max_mem_size=max_mem_size, nthreads=nthreads)
    h2o.no_progress()


def shutdown() -> None:
    try:
        import h2o

        h2o.cluster().shutdown()
    except Exception:  # pragma: no cover - shutdown is best-effort
        pass


def fit_automl(
    X: np.ndarray,
    y: np.ndarray,
    config: H2OConfig,
    *,
    feature_names: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run AutoML on the training fold only.

    Note the deliberate absence of a ``validation`` parameter: H2O's own
    ``nfolds`` cross-validation runs *inside* this training frame, so no held-out
    split can reach it.
    """
    import h2o
    import pandas as pd
    from h2o.automl import H2OAutoML

    version = ensure_available()
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(np.asarray(X), columns=names)
    frame["target"] = np.asarray(y).astype(int)

    h2o_frame = h2o.H2OFrame(frame)
    h2o_frame["target"] = h2o_frame["target"].asfactor()

    automl = H2OAutoML(
        max_models=config.max_models,
        max_runtime_secs=config.max_runtime_secs,
        include_algos=list(config.include_algos),
        nfolds=config.nfolds,
        seed=config.seed,
        sort_metric=config.sort_metric,
    )
    automl.train(x=names, y="target", training_frame=h2o_frame)

    leader = automl.leader
    meta = {
        "backend": "h2o",
        "h2o_version": version,
        "matches_paper_version": version == REQUIRED_VERSION,
        "leader_model_id": str(leader.model_id),
        "leader_algo": str(leader.algo),
        "n_train_rows": int(len(X)),
        **config.describe(),
    }
    return leader, meta


def predict_proba(model: Any, X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    import h2o
    import pandas as pd

    if len(X) == 0:
        return np.empty(0, dtype=np.float64)
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    frame = h2o.H2OFrame(pd.DataFrame(np.asarray(X), columns=names))
    predictions = model.predict(frame).as_data_frame()
    column = "p1" if "p1" in predictions.columns else predictions.columns[-1]
    return predictions[column].to_numpy(dtype=np.float64)
