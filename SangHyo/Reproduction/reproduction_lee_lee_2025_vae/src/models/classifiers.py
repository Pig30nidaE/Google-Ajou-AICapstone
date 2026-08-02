"""4개 비교 모델 (논문 §4.2·§5.1).

논문이 보고한 하이퍼파라미터는 기본값에 반영하고, 미보고 항목은
``assumptions.md`` E절의 가정값을 쓴다. ``paper_reported_keys``가 둘을 구분한다.

무거운 의존성(xgboost, torch, pytorch_tabnet)은 모두 **지연 임포트**한다.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import BaseClassifier, TorchTrainingMixin

log = logging.getLogger(__name__)

__all__ = ["XGBoostClassifier", "DNNClassifier", "TabNetClassifier", "WideDeepClassifier"]


class XGBoostClassifier(BaseClassifier):
    """논문 보고: multi:softmax, max_depth=6, learning_rate=0.1.

    확률 출력이 필요하므로 ``multi:softprob``을 쓴다 (학습은 동일, 출력만 확률).
    assumptions.md E-1.
    """

    name = "XGBoost"
    paper_reported_keys = ("max_depth", "learning_rate", "objective")

    DEFAULTS = {
        "objective": "multi:softprob",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "min_child_weight": 1,
        "tree_method": "hist",
    }

    def fit(self, X, y, *, sample_weight=None, eval_set=None):
        from xgboost import XGBClassifier

        params = {**self.DEFAULTS, **self.params}
        params.pop("class_weight", None)
        self._model = XGBClassifier(
            num_class=self.n_classes, random_state=self.seed, **params
        )
        kwargs = {}
        if eval_set is not None and len(eval_set[0]):
            kwargs["eval_set"] = [(eval_set[0], eval_set[1])]
            kwargs["verbose"] = False
        self._model.fit(X, y, sample_weight=sample_weight, **kwargs)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class DNNClassifier(BaseClassifier, TorchTrainingMixin):
    """논문 보고: 512-256-128-64-32, ReLU, L2, BatchNorm, Dropout 0.5, softmax."""

    name = "DNN"
    paper_reported_keys = ("hidden", "dropout", "batch_norm", "activation")

    DEFAULTS = {
        "hidden": (512, 256, 128, 64, 32),
        "dropout": 0.5,
        "batch_norm": True,
        "activation": "relu",
        "l2": 1e-4,          # 논문은 "L2 정규화" 사실만 보고, 계수 미보고
        "lr": 1e-3,
        "epochs": 200,
        "batch_size": 128,
        "patience": 20,
    }

    def _build(self, d_in: int):
        from torch import nn

        p = {**self.DEFAULTS, **self.params}
        layers, prev = [], d_in
        for h in p["hidden"]:
            layers.append(nn.Linear(prev, h))
            if p["batch_norm"]:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if p["dropout"] > 0:
                layers.append(nn.Dropout(p["dropout"]))
            prev = h
        layers.append(nn.Linear(prev, self.n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y, *, sample_weight=None, eval_set=None):
        p = {**self.DEFAULTS, **self.params}
        self._train_torch(
            self._build(X.shape[1]), X, y,
            eval_set=eval_set, sample_weight=sample_weight,
            lr=p["lr"], epochs=p["epochs"], batch_size=p["batch_size"],
            weight_decay=p["l2"], patience=p["patience"],
            class_weight=self.params.get("class_weight"),
        )
        return self

    def predict_proba(self, X):
        return self._predict_proba_torch(X)


class WideDeepClassifier(BaseClassifier, TorchTrainingMixin):
    """논문 보고: Deep 256-128-64, ReLU, Dropout 0.3.

    ⚠️ **Wide 컴포넌트의 입력은 논문에 전혀 기재되지 않았다**
    (unresolved_questions.md Q12). 여기서는 46개 원 특성의 단일 선형 층으로 가정한다.
    교차특성은 만들지 않으며, ``wide_crosses``를 지정하면 결과에
    ``assumed_wide_crosses`` 태그가 붙는다 (assumptions.md E-4).
    """

    name = "WideDeep"
    paper_reported_keys = ("deep_hidden", "dropout", "activation")

    DEFAULTS = {
        "deep_hidden": (256, 128, 64),
        "dropout": 0.3,
        "activation": "relu",
        "wide_features": "all",   # all | none | [컬럼 인덱스 리스트]
        "wide_crosses": (),       # 기본값: 교차특성 없음
        "combine": "sum",         # sum | concat_then_linear
        "batch_norm": False,      # 논문 미보고
        "lr": 1e-3,
        "epochs": 200,
        "batch_size": 128,
        "patience": 20,
        "l2": 0.0,
    }

    def _build(self, d_in: int):
        import torch
        from torch import nn

        p = {**self.DEFAULTS, **self.params}
        if p["wide_crosses"]:
            log.warning(
                "wide_crosses가 지정되었다. 이는 논문 미보고 설정이며 "
                "'원 논문 방식'이 아니다 (assumptions.md E-4)."
            )
        n_classes, deep_hidden = self.n_classes, p["deep_hidden"]

        class WideDeep(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_wide = p["wide_features"] != "none"
                self.wide = nn.Linear(d_in, n_classes) if self.use_wide else None
                layers, prev = [], d_in
                for h in deep_hidden:
                    layers.append(nn.Linear(prev, h))
                    if p["batch_norm"]:
                        layers.append(nn.BatchNorm1d(h))
                    layers.append(nn.ReLU())
                    if p["dropout"] > 0:
                        layers.append(nn.Dropout(p["dropout"]))
                    prev = h
                self.deep = nn.Sequential(*layers)
                if p["combine"] == "sum":
                    self.deep_out = nn.Linear(prev, n_classes)
                    self.head = None
                else:
                    self.deep_out = None
                    self.head = nn.Linear(prev + (n_classes if self.use_wide else 0), n_classes)

            def forward(self, x):
                d = self.deep(x)
                if self.head is None:
                    logits = self.deep_out(d)
                    if self.use_wide:
                        logits = logits + self.wide(x)
                    return logits
                parts = [d] + ([self.wide(x)] if self.use_wide else [])
                return self.head(torch.cat(parts, dim=1))

        return WideDeep()

    def fit(self, X, y, *, sample_weight=None, eval_set=None):
        p = {**self.DEFAULTS, **self.params}
        self._train_torch(
            self._build(X.shape[1]), X, y,
            eval_set=eval_set, sample_weight=sample_weight,
            lr=p["lr"], epochs=p["epochs"], batch_size=p["batch_size"],
            weight_decay=p["l2"], patience=p["patience"],
            class_weight=self.params.get("class_weight"),
        )
        return self

    def predict_proba(self, X):
        return self._predict_proba_torch(X)

    def describe(self):
        d = super().describe()
        d["wide_input_note"] = (
            "논문은 Wide 컴포넌트 입력을 보고하지 않았다. "
            "46개 원 특성의 선형 층으로 가정했다 (assumptions.md E-4)."
        )
        return d


class TabNetClassifier(BaseClassifier):
    """논문 보고: n_d = n_a = 64, n_steps = 5."""

    name = "TabNet"
    paper_reported_keys = ("n_d", "n_a", "n_steps")

    DEFAULTS = {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.3,
        "lambda_sparse": 1e-3,
        "n_independent": 2,
        "n_shared": 2,
        "lr": 2e-2,
        "max_epochs": 200,
        "patience": 20,
        "batch_size": 1024,
        "virtual_batch_size": 128,
    }

    def fit(self, X, y, *, sample_weight=None, eval_set=None):
        from pytorch_tabnet.tab_model import TabNetClassifier as _TabNet

        p = {**self.DEFAULTS, **self.params}
        model_keys = {"n_d", "n_a", "n_steps", "gamma", "lambda_sparse", "n_independent", "n_shared"}
        self._model = _TabNet(
            **{k: p[k] for k in model_keys},
            optimizer_params={"lr": p["lr"]},
            seed=self.seed,
            verbose=0,
        )
        kwargs = {}
        if eval_set is not None and len(eval_set[0]):
            kwargs["eval_set"] = [(np.asarray(eval_set[0]), np.asarray(eval_set[1]))]
            kwargs["patience"] = p["patience"]
        self._model.fit(
            np.asarray(X), np.asarray(y),
            max_epochs=p["max_epochs"],
            batch_size=min(p["batch_size"], max(len(y), 1)),
            virtual_batch_size=min(p["virtual_batch_size"], max(len(y), 1)),
            weights=0,
            **kwargs,
        )
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(np.asarray(X))
