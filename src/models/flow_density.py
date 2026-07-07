"""
Phase 4: normalizing-flow density model over the Phase-2 feature vector.

The flow learns p(features | Gaussian, isotropic LambdaCDM) from the Phase-3
null-sim table. Anomaly score for any patch — real or simulated — is then
-log p(features) under the trained flow. High score = statistically unusual
relative to the null ensemble, with no prior assumption about what the anomaly
should look like (template-free, per CLAUDE.md Sec. 2).

Two backend implementations are exposed under a single interface:

  - `MAFDensity`  : masked-autoregressive flow via `normflows` (production)
  - `GaussianDensity` : whitened multivariate Gaussian (a diagnostic baseline
    and a safe fallback that the unit tests can run without pulling torch)

Both implement `.fit(X_train, X_val)` and `.anomaly_score(X)`. The MAF is the
one the paper's results come from; the Gaussian baseline lets us confirm the
whole downstream pipeline runs before torch is installed and lets us quantify
how much the flow buys us over a linear baseline.

**Train/val split MUST partition by sim_id, not by patch.** Patches from the
same Gaussian realization share large-scale correlated modes; splitting by
patch leaks that structure and gives falsely low validation NLL. `split_by_sim`
enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Feature-table preparation
# ---------------------------------------------------------------------------


def split_by_sim(rows: Sequence[dict], val_frac: float = 0.2,
                  seed: int = 0) -> tuple[list[dict], list[dict]]:
    """
    Partition rows into (train, val) by sim_id, not by patch. Every sim_id lives
    entirely in one split — this is the leak-avoiding split described above.
    """
    sim_ids = sorted({r["sim_id"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(sim_ids)
    n_val = max(1, int(round(len(sim_ids) * val_frac)))
    val_ids = set(sim_ids[:n_val])
    train_rows = [r for r in rows if r["sim_id"] not in val_ids]
    val_rows = [r for r in rows if r["sim_id"] in val_ids]
    return train_rows, val_rows


def rows_to_matrix(rows: Sequence[dict], feature_names: Sequence[str]) -> np.ndarray:
    """
    Stack the named feature columns into an (n_patches, n_features) float array.
    Rows containing NaN in any selected feature are dropped — cleaner than
    imputing, and NaNs here reflect legitimate feature-extraction failures
    (e.g. wavelet scale finer than projection resolution) that we do not want
    the flow to pretend it can score.
    """
    mat = np.array([[float(r[k]) for k in feature_names] for r in rows], dtype=float)
    finite = np.all(np.isfinite(mat), axis=1)
    return mat[finite]


@dataclass
class Standardizer:
    """Per-feature whitening — fit on training data, apply to any input."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return cls(mean=mu, std=sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


# ---------------------------------------------------------------------------
# Baseline: whitened multivariate Gaussian
# ---------------------------------------------------------------------------


class GaussianDensity:
    """
    Diagnostic baseline: p(features) as a full-covariance Gaussian in whitened
    coordinates. Two roles:
      - Lets the whole Phase 5 / 6 / 7 pipeline run before torch is installed.
      - Gives an interpretable NLL benchmark the flow must beat to earn its complexity.
    """

    def __init__(self) -> None:
        self.scaler: Optional[Standardizer] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._log_norm: float = 0.0

    def fit(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None) -> "GaussianDensity":
        self.scaler = Standardizer.fit(X_train)
        Z = self.scaler.transform(X_train)
        cov = np.cov(Z, rowvar=False)
        d = cov.shape[0]
        # Small jitter for numerical stability with limited samples.
        cov = cov + 1e-6 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
        assert sign > 0, "training covariance is not positive definite"
        self._cov_inv = np.linalg.inv(cov)
        self._log_norm = 0.5 * (d * np.log(2.0 * np.pi) + logdet)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self._cov_inv is not None, "call .fit first"
        Z = self.scaler.transform(X)
        quad = np.einsum("ij,jk,ik->i", Z, self._cov_inv, Z)
        return 0.5 * quad + self._log_norm  # = -log p(X)


# ---------------------------------------------------------------------------
# Production: MAF via `normflows` (falls back with a clear error if missing)
# ---------------------------------------------------------------------------


class MAFDensity:
    """
    Masked autoregressive flow density estimator. Thin wrapper around normflows
    (or nflows) so we can swap either without disturbing the interface. Trains
    on whitened features and returns -log p as `anomaly_score`.

    The flow is imported lazily so this file can be imported and unit-tested
    in envs that only have numpy + scipy.
    """

    def __init__(self, n_layers: int = 6, hidden: int = 64, lr: float = 1e-3,
                  n_epochs: int = 200, batch: int = 256, device: str = "cpu",
                  seed: int = 0, verbose: bool = True,
                  log_every: int = 10) -> None:
        self.n_layers = n_layers
        self.hidden = hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch = batch
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.log_every = log_every
        self.scaler: Optional[Standardizer] = None
        self._model = None  # normflows.NormalizingFlow
        self._torch = None

    def _build(self, dim: int):
        try:
            import torch
            import normflows as nf
        except ImportError as e:  # pragma: no cover — imported only when torch installed
            raise RuntimeError(
                "MAFDensity requires torch + normflows. Install the hawkingml env "
                "from environment.yml, or use GaussianDensity as a fallback."
            ) from e
        torch.manual_seed(self.seed)
        flows = []
        for _ in range(self.n_layers):
            flows.append(nf.flows.MaskedAffineAutoregressive(dim, self.hidden))
            flows.append(nf.flows.Permute(dim, mode="swap"))
        base = nf.distributions.DiagGaussian(dim)
        model = nf.NormalizingFlow(base, flows).to(self.device)
        self._torch = torch
        return model

    def fit(self, X_train: np.ndarray,
             X_val: Optional[np.ndarray] = None) -> "MAFDensity":
        # Line-by-line diagnostics so any hang has a unique last-log-line
        # signature — critical on macOS where torch/normflows CAN block on
        # threading init with no error message.
        _p = lambda msg: print(msg, flush=True) if self.verbose else None
        _p(f"[MAFDensity] fit() entered; whitening features ...")
        self.scaler = Standardizer.fit(X_train)
        Ztr = self.scaler.transform(X_train)
        Zva = self.scaler.transform(X_val) if X_val is not None else None
        dim = Ztr.shape[1]
        _p(f"[MAFDensity] Building flow (dim={dim}) — importing torch/normflows ...")
        self._model = self._build(dim)
        _p(f"[MAFDensity] Flow built; assembling optimizer ...")
        torch = self._torch

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        X = torch.tensor(Ztr, dtype=torch.float32, device=self.device)
        best_val = np.inf
        _p = lambda msg: print(msg, flush=True) if self.verbose else None
        _p(f"[MAFDensity] Training {self.n_layers}-layer MAF (hidden={self.hidden}) "
           f"for {self.n_epochs} epochs on {Ztr.shape[0]} patches × "
           f"{Ztr.shape[1]} features ({self.device})")

        # Diagnostic: time the first forward+backward pass so we know per-batch cost
        # BEFORE launching the full training loop. If this hangs for > a few sec on
        # 600 samples × 32 features, the flow backend has a CPU pathology and the
        # right fix is to swap to a coupling-based backend (RealNVP), not to wait.
        import time
        t0 = time.perf_counter()
        _p("[MAFDensity] Timing warm-up batch ...")
        first_batch = X[:min(self.batch, X.shape[0])]
        loss = -self._model.log_prob(first_batch).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        dt = time.perf_counter() - t0
        n_batches_per_epoch = max(1, (X.shape[0] + self.batch - 1) // self.batch)
        est_epoch_s = dt * n_batches_per_epoch
        _p(f"[MAFDensity] Warm-up batch: {dt:.2f}s. "
           f"Estimated {est_epoch_s:.1f}s per epoch → "
           f"{est_epoch_s * self.n_epochs / 60:.1f} min total. Starting training.")

        for epoch in range(self.n_epochs):
            idx = torch.randperm(X.shape[0], device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, X.shape[0], self.batch):
                batch = X[idx[start:start + self.batch]]
                loss = -self._model.log_prob(batch).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            epoch_loss /= max(1, n_batches)
            val_nll = float("nan")
            if Zva is not None:
                with torch.no_grad():
                    val_x = torch.tensor(Zva, dtype=torch.float32, device=self.device)
                    val_nll = -self._model.log_prob(val_x).mean().item()
                if val_nll < best_val:
                    best_val = val_nll
            if self.verbose and (epoch % self.log_every == 0 or epoch == self.n_epochs - 1):
                _p(f"[MAFDensity] epoch {epoch:4d}/{self.n_epochs}  "
                   f"train_nll={epoch_loss:+.4f}  val_nll={val_nll:+.4f}  "
                   f"best_val={best_val:+.4f}")
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None and self.scaler is not None, "call .fit first"
        torch = self._torch
        with torch.no_grad():
            Z = self.scaler.transform(X)
            x = torch.tensor(Z, dtype=torch.float32, device=self.device)
            logp = self._model.log_prob(x).cpu().numpy()
        return -logp
