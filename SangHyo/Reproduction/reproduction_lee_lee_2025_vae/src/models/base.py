"""분류기 공통 인터페이스와 torch 학습 루프.

핵심 규칙 (사용자 지시 8절): **early stopping은 train 내부 validation에서만.**
outer test로 early stopping하지 않는다. 내부 validation은 피험자 단위로 나누며
합성행을 포함하지 않는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["BaseClassifier", "InternalValidation", "make_internal_validation", "TorchTrainingMixin"]


@dataclass
class InternalValidation:
    """train 내부 validation 인덱스."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    subjects: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))


def _stratified_subject_sample(
    y: np.ndarray,
    subject: np.ndarray,
    real: np.ndarray,
    subs: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> set[str]:
    """클래스별로 층화해 validation 피험자를 뽑는다.

    무작위 표집이면 train fold에 Dem 피험자가 8명뿐이라 20% 표집에서 Dem이 통째로
    빠지는 일이 실제로 생긴다(실측: 20개 seed 중 3건). 그러면

    * early stopping이 존재하지 않는 클래스를 무시한 채 이뤄지고,
    * 클래스 수를 확인하는 metric(예: log loss)이 예외로 죽는다.

    각 클래스에서 최소 1명(단, 그 클래스 피험자가 2명 이상일 때)을 보장한다.
    """
    subject_label: dict[str, int] = {}
    for s, label in zip(subject[real], y[real]):
        subject_label.setdefault(str(s), int(label))

    n_val_total = max(1, int(round(len(subs) * fraction)))
    by_class: dict[int, list[str]] = {}
    for s in subs:
        by_class.setdefault(subject_label[str(s)], []).append(str(s))

    val_subs: set[str] = set()
    for label in sorted(by_class):
        pool = sorted(by_class[label])
        if len(pool) < 2:
            # 1명뿐이면 train에서 빼면 그 클래스를 학습할 수 없다. 그대로 둔다.
            continue
        k = max(1, int(round(len(pool) * fraction)))
        k = min(k, len(pool) - 1)          # 최소 1명은 train에 남긴다
        val_subs.update(rng.choice(np.array(pool), size=k, replace=False).tolist())

    # 층화 결과가 목표 비율에 못 미치면 남은 피험자에서 무작위로 채운다.
    remaining = sorted(set(map(str, subs)) - val_subs)
    if len(val_subs) < n_val_total and remaining:
        extra = min(n_val_total - len(val_subs), len(remaining) - 1)
        if extra > 0:
            val_subs.update(rng.choice(np.array(remaining), size=extra, replace=False).tolist())
    return val_subs


def make_internal_validation(
    y: np.ndarray,
    subject: np.ndarray,
    is_synthetic: np.ndarray,
    *,
    fraction: float = 0.2,
    split_by: str = "subject",
    seed: int = 42,
) -> InternalValidation:
    """train 안에서 early stopping용 validation을 떼어낸다.

    합성행은 validation에 넣지 않는다 (synthetic_data_risk.md §2 금지 6).
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    if fraction <= 0:
        return InternalValidation(np.arange(n), np.array([], dtype=int))

    real = ~np.asarray(is_synthetic, dtype=bool)
    if split_by == "subject":
        subs = np.array(sorted({str(s) for s in subject[real]}))
        if len(subs) >= 5:
            val_subs = _stratified_subject_sample(y, subject, real, subs, fraction, rng)
            val_mask = np.array([str(s) in val_subs for s in subject]) & real
            if val_mask.any() and (~val_mask).sum() > 0:
                return InternalValidation(
                    np.flatnonzero(~val_mask), np.flatnonzero(val_mask), subject[val_mask]
                )
        log.warning("내부 validation을 피험자 단위로 나눌 표본이 부족하다. 행 단위로 대체한다.")

    real_idx = np.flatnonzero(real)
    perm = rng.permutation(real_idx)
    n_val = max(1, int(round(len(perm) * fraction)))
    val_idx = perm[:n_val]
    val_set = set(val_idx.tolist())
    train_idx = np.array([i for i in range(n) if i not in val_set])
    return InternalValidation(train_idx, val_idx, subject[val_idx])


class BaseClassifier:
    """모든 분류기가 만족해야 하는 계약."""

    name = "base"
    #: ``True``인 모델만 학습행 일부를 early-stopping validation으로 분리한다.
    #: XGBoost처럼 eval_set을 받더라도 실제 early stopping을 하지 않는 모델에서
    #: 학습행 20%를 이유 없이 버리는 문제를 막는다.
    uses_early_stopping = False
    #: 논문에서 보고된 하이퍼파라미터 키 (결과표에서 "가정값"과 구분해 표시한다).
    paper_reported_keys: tuple[str, ...] = ()

    def __init__(self, params: dict | None = None, *, seed: int = 42) -> None:
        self.params = dict(params or {})
        self.seed = seed
        self.n_classes = 3
        self.fit_log: dict = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "BaseClassifier":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def describe(self) -> dict:
        return {
            "model": self.name,
            "params": self.params,
            "paper_reported_keys": list(self.paper_reported_keys),
            "assumed_keys": sorted(set(self.params) - set(self.paper_reported_keys)),
        }


class TorchTrainingMixin:
    """DNN·Wide&Deep 공용 학습 루프."""

    def _train_torch(
        self,
        module,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set,
        sample_weight,
        lr: float,
        epochs: int,
        batch_size: int,
        weight_decay: float,
        patience: int,
        class_weight: dict | None = None,
    ):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = module.to(device)

        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        yt = torch.tensor(np.asarray(y, dtype=np.int64))
        wt = (
            torch.tensor(np.asarray(sample_weight, dtype=np.float32))
            if sample_weight is not None
            else torch.ones(len(y), dtype=torch.float32)
        )
        drop_last = len(y) % batch_size == 1  # BatchNorm 방어
        loader = DataLoader(
            TensorDataset(Xt, yt, wt),
            batch_size=min(batch_size, len(y)),
            shuffle=True,
            drop_last=drop_last,
        )

        cw = None
        if class_weight:
            cw = torch.tensor(
                [class_weight.get(i, 1.0) for i in range(self.n_classes)], dtype=torch.float32
            ).to(device)
        opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

        val_X = val_y = None
        if eval_set is not None and len(eval_set[0]):
            val_X = torch.tensor(np.asarray(eval_set[0], dtype=np.float32)).to(device)
            val_y = torch.tensor(np.asarray(eval_set[1], dtype=np.int64)).to(device)

        best, best_state, bad, best_epoch = float("inf"), None, 0, -1
        history: list[dict] = []
        for epoch in range(epochs):
            module.train()
            tot, nb = 0.0, 0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                opt.zero_grad()
                logits = module(xb)
                loss_vec = torch.nn.functional.cross_entropy(
                    logits, yb, weight=cw, reduction="none"
                )
                loss = (loss_vec * wb).mean()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
            rec = {"epoch": epoch, "train_loss": tot / max(nb, 1)}

            if val_X is not None:
                module.eval()
                with torch.no_grad():
                    vloss = float(
                        torch.nn.functional.cross_entropy(module(val_X), val_y, weight=cw)
                    )
                rec["val_loss"] = vloss
                if vloss < best - 1e-6:
                    best, bad, best_epoch = vloss, 0, epoch
                    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        history.append(rec)
                        log.info("%s early stopping @ epoch %d (best %d)", self.name, epoch, best_epoch)
                        break
            history.append(rec)

        if best_state is not None:
            module.load_state_dict(best_state)
        self.fit_log = {"history": history, "best_epoch": best_epoch, "n_epochs_run": len(history)}
        self._module, self._device = module, device
        return module

    def _predict_proba_torch(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._module.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(self._device)
            return torch.softmax(self._module(xt), dim=1).cpu().numpy()
