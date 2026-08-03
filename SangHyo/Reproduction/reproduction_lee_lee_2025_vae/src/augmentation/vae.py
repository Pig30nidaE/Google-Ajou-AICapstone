"""표 형식 VAE (논문 §5.1, 그림 2).

구조 (논문 보고):

    Encoder: d -> 512 -> 256 -> (mu, logvar) in R^L
    Decoder: L -> 256 -> 512 -> d
    각 은닉층: Linear -> BatchNorm1d -> ReLU -> Dropout(0.3)   (순서는 가정, D-1)
    손실: reconstruction + beta * KL,  Adam(lr=1e-4)

latent 차원 L은 본문 500 / 그림 2는 50으로 충돌한다 (report_inconsistencies.md I-2).
어느 쪽도 기본값으로 정하지 않고 config에서 받는다.

torch는 **지연 임포트**한다. 이 저장소의 dry-run·unit test는 torch 없이 동작해야 한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["VAEConfig", "TabularVAE", "VAETrainer", "VAETrainingLog"]


@dataclass
class VAEConfig:
    """VAE 설정. 논문 보고값은 기본값에 반영하고, 미보고 항목은 assumptions.md D절 가정값."""

    input_dim: int
    latent_dim: int                       # 논문 충돌: 500(본문) / 50(그림 2)
    encoder_hidden: tuple[int, ...] = (512, 256)   # 논문 보고
    decoder_hidden: tuple[int, ...] = (256, 512)   # 논문 보고
    dropout: float = 0.3                  # 논문 보고
    batch_norm: bool = True               # 논문 보고
    learning_rate: float = 1e-4           # 논문 보고
    beta: float = 1.0                     # 미보고 → 가정 (D-2)
    epochs: int = 300                     # 미보고 → 가정 (D-3)
    batch_size: int = 64                  # 미보고 → 가정 (D-3)
    recon_reduction: str = "mean_per_feature"  # mean_per_feature | sum
    # reconstruction과 KL을 각각 feature/latent 차원으로 정규화한다. ``sum``은
    # latent_dim=50/500에 따라 beta의 실효값을 10배 바꾸므로 민감도 분석용 opt-in이다.
    kl_reduction: str = "mean"                 # mean | sum
    output_activation: str = "linear"          # linear | sigmoid
    layer_order: str = "linear_bn_relu_dropout"
    early_stopping: bool = True
    patience: int = 30
    min_delta: float = 1e-5
    val_fraction: float = 0.2
    val_split_by: str = "subject"          # subject | row
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str | None = None

    @classmethod
    def from_dict(cls, d: dict, *, input_dim: int, seed: int = 42) -> "VAEConfig":
        d = dict(d or {})
        es = d.pop("early_stopping", {}) or {}
        if isinstance(es, bool):
            es = {"enabled": es}
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["input_dim"] = input_dim
        kwargs.setdefault("seed", seed)
        kwargs["early_stopping"] = bool(es.get("enabled", True))
        if "patience" in es:
            kwargs["patience"] = int(es["patience"])
        if "min_delta" in es:
            kwargs["min_delta"] = float(es["min_delta"])
        if "monitor" in es:
            pass  # 현재는 val_total_loss 고정
        if "latent_dim" not in kwargs:
            raise ValueError("vae.latent_dim은 명시해야 한다 (논문 본문 500 / 그림 2 50, I-2)")
        for key in ("encoder_hidden", "decoder_hidden"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        cfg = cls(**kwargs)
        if cfg.recon_reduction not in {"mean_per_feature", "sum"}:
            raise ValueError(
                "vae.recon_reduction은 'mean_per_feature' 또는 'sum'이어야 한다"
            )
        if cfg.kl_reduction not in {"mean", "sum"}:
            raise ValueError("vae.kl_reduction은 'mean' 또는 'sum'이어야 한다")
        return cfg

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class VAETrainingLog:
    """epoch별 손실. 재구성 손실과 KL을 **따로** 기록한다 (사용자 요구)."""

    train_total: list[float] = field(default_factory=list)
    train_recon: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_total: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False
    #: 논문 §5.1의 "재구성 오차 0.0002"와 대조하기 위해 두 척도 모두 기록한다 (I-15).
    final_recon_mse_scaled_space: float = float("nan")
    final_recon_mse_raw_space: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "train_total": self.train_total,
            "train_recon": self.train_recon,
            "train_kl": self.train_kl,
            "val_total": self.val_total,
            "val_recon": self.val_recon,
            "val_kl": self.val_kl,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "final_recon_mse_scaled_space": self.final_recon_mse_scaled_space,
            "final_recon_mse_raw_space": self.final_recon_mse_raw_space,
        }


def _build_mlp(torch_nn, dims, *, dropout: float, batch_norm: bool):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch_nn.Linear(dims[i], dims[i + 1]))
        if batch_norm:
            layers.append(torch_nn.BatchNorm1d(dims[i + 1]))
        layers.append(torch_nn.ReLU())
        if dropout > 0:
            layers.append(torch_nn.Dropout(dropout))
    return torch_nn.Sequential(*layers)


def build_vae(cfg: VAEConfig):
    """torch 모듈을 만든다 (지연 임포트)."""
    import torch
    from torch import nn

    class TabularVAEModule(nn.Module):
        def __init__(self, c: VAEConfig):
            super().__init__()
            self.cfg = c
            enc_dims = (c.input_dim, *c.encoder_hidden)
            self.encoder = _build_mlp(nn, enc_dims, dropout=c.dropout, batch_norm=c.batch_norm)
            self.fc_mu = nn.Linear(enc_dims[-1], c.latent_dim)
            self.fc_logvar = nn.Linear(enc_dims[-1], c.latent_dim)
            dec_dims = (c.latent_dim, *c.decoder_hidden)
            self.decoder = _build_mlp(nn, dec_dims, dropout=c.dropout, batch_norm=c.batch_norm)
            self.fc_out = nn.Linear(dec_dims[-1], c.input_dim)

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def decode(self, z):
            h = self.decoder(z)
            out = self.fc_out(h)
            if self.cfg.output_activation == "sigmoid":
                out = torch.sigmoid(out)
            return out

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    return TabularVAEModule(cfg)


def vae_loss(recon, x, mu, logvar, cfg: VAEConfig):
    """재구성 손실과 KL을 **분리해서** 반환한다."""
    import torch

    if cfg.recon_reduction == "sum":
        recon_loss = torch.nn.functional.mse_loss(recon, x, reduction="sum") / x.shape[0]
    else:  # mean_per_feature
        recon_loss = torch.nn.functional.mse_loss(recon, x, reduction="mean")
    # latent 차원에 대해 합한 뒤 배치 평균. 기본 ``mean``은 latent 차원으로도 나눠
    # latent_dim=50/500 사이에서 beta의 의미가 바뀌지 않게 한다 (D-2).
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl_per_sample.mean()
    if cfg.kl_reduction == "mean":
        kl = kl / mu.shape[1]
    return recon_loss, kl


class TabularVAE:
    """학습·생성·체크포인트를 감싼 래퍼."""

    def __init__(self, cfg: VAEConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.log = VAETrainingLog()
        self._source_subjects: list = []
        self._device = None

    # ------------------------------------------------------------------
    def _resolve_device(self):
        import torch

        if self.cfg.device != "auto":
            return torch.device(self.cfg.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X: np.ndarray,
        *,
        subjects: np.ndarray | None = None,
    ) -> "TabularVAE":
        """VAE를 학습한다.

        Args:
            X: (n, d) 학습 행렬. **현재 fold의 실제 Dem 기록만** 넘겨야 한다.
            subjects: (n,) 피험자 ID. early stopping용 내부 validation을
                피험자 단위로 나누는 데 쓰인다 (평가 fold는 절대 포함되지 않는다).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self._device = self._resolve_device()
        self.model = build_vae(self.cfg).to(self._device)
        if subjects is not None:
            self._source_subjects = sorted({str(s) for s in subjects})

        tr_idx, val_idx = self._internal_split(len(X), subjects)
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        train_ds = TensorDataset(Xt[tr_idx])
        # batch_size=1이면 BatchNorm이 실패하므로 drop_last로 방어한다.
        drop_last = len(tr_idx) % self.cfg.batch_size == 1
        loader = DataLoader(
            train_ds, batch_size=min(self.cfg.batch_size, len(tr_idx)),
            shuffle=True, drop_last=drop_last,
        )
        val_t = Xt[val_idx].to(self._device) if len(val_idx) else None

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        best, best_state, bad = float("inf"), None, 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            tot = rec = kld = 0.0
            n_batch = 0
            for (xb,) in loader:
                xb = xb.to(self._device)
                opt.zero_grad()
                recon, mu, logvar = self.model(xb)
                r, k = vae_loss(recon, xb, mu, logvar, self.cfg)
                loss = r + self.cfg.beta * k
                loss.backward()
                opt.step()
                tot += float(loss); rec += float(r); kld += float(k); n_batch += 1
            n_batch = max(n_batch, 1)
            self.log.train_total.append(tot / n_batch)
            self.log.train_recon.append(rec / n_batch)
            self.log.train_kl.append(kld / n_batch)

            if val_t is not None and len(val_t) > 1:
                self.model.eval()
                with torch.no_grad():
                    recon, mu, logvar = self.model(val_t)
                    r, k = vae_loss(recon, val_t, mu, logvar, self.cfg)
                    v = float(r + self.cfg.beta * k)
                self.log.val_total.append(v)
                self.log.val_recon.append(float(r))
                self.log.val_kl.append(float(k))
                if self.cfg.early_stopping:
                    if v < best - self.cfg.min_delta:
                        best, bad = v, 0
                        best_state = {kk: vv.detach().clone() for kk, vv in self.model.state_dict().items()}
                        self.log.best_epoch = epoch
                    else:
                        bad += 1
                        if bad >= self.cfg.patience:
                            self.log.stopped_early = True
                            log.info("VAE early stopping @ epoch %d (best %d)", epoch, self.log.best_epoch)
                            break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.log.final_recon_mse_scaled_space = self._reconstruction_mse(X)
        if self.cfg.checkpoint_dir:
            self.save_checkpoint(Path(self.cfg.checkpoint_dir) / "vae_best.pt")
        return self

    def _internal_split(self, n: int, subjects: np.ndarray | None):
        """early stopping용 내부 validation. 피험자 단위 분리를 기본으로 한다."""
        rng = np.random.default_rng(self.cfg.seed)
        if not self.cfg.early_stopping or self.cfg.val_fraction <= 0:
            return np.arange(n), np.array([], dtype=int)
        if self.cfg.val_split_by == "subject" and subjects is not None:
            uniq = np.array(sorted({str(s) for s in subjects}))
            if len(uniq) >= 3:
                n_val = max(1, int(round(len(uniq) * self.cfg.val_fraction)))
                val_subs = set(rng.choice(uniq, size=n_val, replace=False).tolist())
                mask = np.array([str(s) in val_subs for s in subjects])
                if mask.any() and (~mask).sum() > 1:
                    return np.flatnonzero(~mask), np.flatnonzero(mask)
            log.warning(
                "VAE 학습 피험자가 %d명뿐이라 피험자 단위 내부 validation이 불안정하다. "
                "행 단위로 대체한다 (unresolved_questions.md Q8).",
                len(uniq),
            )
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * self.cfg.val_fraction)))
        return idx[n_val:], idx[:n_val]

    def _reconstruction_mse(self, X: np.ndarray) -> float:
        import torch

        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(self._device)
            recon, _, _ = self.model(xt)
            return float(torch.nn.functional.mse_loss(recon, xt, reduction="mean"))

    def sample(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """표준정규에서 latent를 뽑아 디코딩한다 (논문 §5.1 생성 방식)."""
        import torch

        if self.model is None:
            raise RuntimeError("fit을 먼저 호출하라")
        g = torch.Generator(device="cpu").manual_seed(self.cfg.seed if seed is None else seed)
        z = torch.randn(n, self.cfg.latent_dim, generator=g).to(self._device)
        self.model.eval()
        with torch.no_grad():
            return self.model.decode(z).cpu().numpy()

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "source_subjects": self._source_subjects,
                "log": self.log.to_dict(),
            },
            path,
        )
        log.info("VAE checkpoint -> %s", path)

    def load_checkpoint(self, path: str | Path) -> "TabularVAE":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        self._device = self._resolve_device()
        self.model = build_vae(self.cfg).to(self._device)
        self.model.load_state_dict(blob["state_dict"])
        self._source_subjects = blob.get("source_subjects", [])
        return self

    @property
    def source_subjects(self) -> list:
        """학습에 사용된 원본 subject ID (provenance·감사용)."""
        return list(self._source_subjects)


class VAETrainer:  # pragma: no cover - 얇은 별칭
    """이름 호환용 별칭."""

    def __new__(cls, cfg: VAEConfig) -> TabularVAE:  # type: ignore[misc]
        return TabularVAE(cfg)
