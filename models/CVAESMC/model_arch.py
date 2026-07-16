import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


Array = np.ndarray


"""
A Novel CVAE-Based Sequential Monte Carlo Framework for Dynamic Soft Sensor Applications
Wenxin Sun, Weili Xiong, et al. IEEE TII 2023

This project version keeps the existing training contract:
    train forward -> dict used by exp/losses.py::CVAESMC_Loss
    eval forward  -> point prediction

It also adds an optional online SMC sampler for delayed lab-result calibration.
"""


def standardize_apply(x: Array, mean: Array, std: Array) -> Array:
    return (x - mean) / (std + 1e-8)


def inverse_standardize(x: Array, mean: Array, std: Array) -> Array:
    return x * (std + 1e-8) + mean


def monte_carlo_sampling(mu: torch.Tensor, logvar: torch.Tensor, num_samples: int) -> torch.Tensor:
    std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 5.0))
    eps = torch.randn((num_samples,) + tuple(std.shape), device=std.device, dtype=std.dtype)
    return mu.unsqueeze(0) + eps * std.unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int, activation: str, num_samples: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, max(hidden_dim // 2, z_dim * 2))
        self.fc_mean_var = nn.Linear(max(hidden_dim // 2, z_dim * 2), z_dim * 2)
        try:
            self.activation = getattr(nn, activation)()
        except AttributeError as exc:
            raise NameError(f"Invalid activation name '{activation}'. Please check activation name") from exc
        self.num_samples = num_samples

    def forward(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.activation(self.fc1(x_enc))
        h = self.activation(self.fc2(h))
        stats = self.fc_mean_var(h)
        mu_posterior, logvar_posterior = torch.chunk(stats, 2, dim=-1)
        logvar_posterior = torch.clamp(logvar_posterior, -10.0, 5.0)
        z_posterior = monte_carlo_sampling(mu_posterior, logvar_posterior, self.num_samples)
        return z_posterior, mu_posterior, logvar_posterior


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        C_out: int,
        z_dim: int,
        hidden_dim: int,
        num_samples: int,
        activation: str,
    ):
        super().__init__()
        try:
            act = getattr(nn, activation)
        except AttributeError as exc:
            raise NameError(f"Invalid activation name '{activation}'. Please check activation name") from exc

        self.net = nn.Sequential(
            nn.Linear(input_dim + z_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, C_out)),
            act(),
            nn.Linear(max(hidden_dim // 2, C_out), C_out),
        )
        self.logvar_dec = nn.Parameter(torch.zeros(1, C_out), requires_grad=True)
        self.num_samples = num_samples
        self.C_out = C_out

    def forward(self, z_t: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_t: [num_samples, B, z_dim]
        x_t: [B, x_dim]
        return:
            mean_dec: [num_samples, B, C_out]
            logvar_dec: [1, C_out], broadcastable in the loss
        """
        num_samples, B, _ = z_t.shape
        x_rep = x_t.unsqueeze(0).expand(num_samples, B, x_t.shape[-1])
        dec_in = torch.cat([x_rep, z_t], dim=-1).reshape(num_samples * B, -1)
        mean_dec = self.net(dec_in).reshape(num_samples, B, self.C_out)
        return mean_dec, torch.clamp(self.logvar_dec, -10.0, 5.0)


@dataclass
class StateSpec:
    na: int
    nb: int
    nd: int
    u_dim: int
    y_dim: int

    @property
    def u_lag_count(self) -> int:
        return self.nb - self.nd + 1

    @property
    def y_lag_count(self) -> int:
        return self.na + 1

    @property
    def x_dim(self) -> int:
        return self.u_lag_count * self.u_dim + self.y_lag_count * self.y_dim


class CVAESMCSSampling:
    """Online SMC sampler for multistep soft-sensor prediction.

    Use this after the CVAE model has been trained. It keeps N possible y-trajectories,
    propagates them by p(y_{t+1}|x_t), and resamples when delayed lab measurements arrive.
    """

    def __init__(
        self,
        model: nn.Module,
        spec: StateSpec,
        x_mean: Array,
        x_std: Array,
        y_mean: Array,
        y_std: Array,
        n_particles: int = 100,
        measurement_std: float | Array = 0.1,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.spec = spec
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self.n_particles = n_particles
        self.measurement_std = np.asarray(measurement_std, dtype=np.float32).reshape(1, -1)
        self.device = device
        self.particles_y: Optional[Array] = None
        self.trajectories: List[Array] = []

    def initialize(self, initial_y_window: Array, jitter_std: float = 0.0) -> None:
        y0 = np.asarray(initial_y_window, dtype=np.float32)
        if y0.ndim == 1:
            y0 = y0[:, None]
        expected = (self.spec.na + 1, self.spec.y_dim)
        if y0.shape != expected:
            raise ValueError(f"initial_y_window must have shape {expected}, got {y0.shape}")

        particles = np.repeat(y0[None, :, :], self.n_particles, axis=0)
        if jitter_std > 0:
            particles += np.random.randn(*particles.shape).astype(np.float32) * jitter_std
        self.particles_y = particles
        self.trajectories = [particles[:, -1, :].copy()]

    def _make_particle_x(self, u_history: Array) -> Array:
        if self.particles_y is None:
            raise RuntimeError("Call initialize() before predict_step().")

        u_history = np.asarray(u_history, dtype=np.float32)
        if u_history.ndim == 1:
            u_history = u_history[:, None]
        expected = (self.spec.u_lag_count, self.spec.u_dim)
        if u_history.shape != expected:
            raise ValueError(f"u_history must have shape {expected}, got {u_history.shape}")

        u_flat = np.tile(u_history.reshape(1, -1), (self.n_particles, 1))
        y_flat = self.particles_y.reshape(self.n_particles, -1)
        return np.concatenate([u_flat, y_flat], axis=1)

    def predict_step(self, u_history: Array, lab_results: Optional[Dict[int, Array]] = None) -> Array:
        x_raw = self._make_particle_x(u_history)
        x_scaled = standardize_apply(x_raw, self.x_mean, self.x_std)
        xb = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y_scaled = self.model.sample(xb, num_samples=1, add_noise=True)[0].cpu().numpy()
        y_next = inverse_standardize(y_scaled, self.y_mean, self.y_std).astype(np.float32)

        assert self.particles_y is not None
        self.particles_y = np.concatenate([self.particles_y[:, 1:, :], y_next[:, None, :]], axis=1)
        self.trajectories.append(y_next.copy())

        if lab_results:
            self._resample(lab_results)
        return y_next

    def _resample(self, lab_results: Dict[int, Array]) -> None:
        traj = np.stack(self.trajectories, axis=1)
        logw = np.zeros(self.n_particles, dtype=np.float64)
        sigma = self.measurement_std.astype(np.float64)
        for time_index, z in lab_results.items():
            if time_index < 0 or time_index >= traj.shape[1]:
                continue
            z_arr = np.asarray(z, dtype=np.float64).reshape(1, -1)
            err = (traj[:, time_index, :].astype(np.float64) - z_arr) / sigma
            logw += -0.5 * np.sum(err * err, axis=1)

        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum() + 1e-12
        idx = np.random.choice(self.n_particles, size=self.n_particles, replace=True, p=w)

        assert self.particles_y is not None
        self.particles_y = self.particles_y[idx]
        self.trajectories = [arr[idx] for arr in self.trajectories]

    def estimate(self, kind: str = "mean", trim_sigma: float = 3.0) -> Array:
        latest = self.trajectories[-1]
        if kind == "mean":
            return latest.mean(axis=0)
        if kind == "median":
            return np.median(latest, axis=0)
        if kind == "midrange":
            mu, sd = latest.mean(axis=0), latest.std(axis=0) + 1e-8
            keep = np.all(np.abs(latest - mu) <= trim_sigma * sd, axis=1)
            vals = latest[keep] if keep.any() else latest
            return 0.5 * (vals.min(axis=0) + vals.max(axis=0))
        raise ValueError("kind must be 'mean', 'median', or 'midrange'")

    def interval(self, q_low: float = 0.1, q_high: float = 0.9) -> Tuple[Array, Array]:
        latest = self.trajectories[-1]
        return np.quantile(latest, q_low, axis=0), np.quantile(latest, q_high, axis=0)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task = configs.task
        self.C_in = configs.C_in
        self.C_out = configs.C_out
        self.z_dim = configs.z_dim
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_samples = configs.num_samples
        self.output_type = configs.output_type

        self.u_dim = self.C_in - self.C_out
        if self.u_dim <= 0:
            raise ValueError("C_in must include input variables plus target variables, so C_in > C_out")

        # With the existing data layout, x_enc has seq_len rows and the last C_out columns are y.
        # For pred_len=1 and seq_len=5 this gives the paper-style 4 u-lags + 5 y-lags.
        self.u_lag_count = max(1, self.seq_len - self.pred_len)
        self.y_lag_count = self.seq_len
        self.x_dim = self.u_lag_count * self.u_dim + self.y_lag_count * self.C_out
        self.encoder_input_dim = self.x_dim + self.C_out

        self.encoder = Encoder(
            input_dim=self.encoder_input_dim,
            z_dim=self.z_dim,
            hidden_dim=configs.hidden_dim,
            activation=configs.activation,
            num_samples=self.num_samples,
        )
        self.decoder = Decoder(
            input_dim=self.x_dim,
            C_out=self.C_out,
            z_dim=self.z_dim,
            hidden_dim=configs.hidden_dim,
            num_samples=self.num_samples,
            activation=configs.activation,
        )

        self.spec = StateSpec(
            na=self.y_lag_count - 1,
            nb=self.u_lag_count - 1,
            nd=0,
            u_dim=self.u_dim,
            y_dim=self.C_out,
        )

    def _build_transition_input(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Build x_t = [u lags, y lags] from [B, seq_len, C_in]."""
        B, T, D = x_enc.shape
        if D < self.C_out:
            raise ValueError(f"x_enc last dimension {D} is smaller than C_out {self.C_out}")
        if T < self.y_lag_count or T < self.u_lag_count:
            raise ValueError(
                f"x_enc seq_len {T} is too short for u_lag_count={self.u_lag_count}, "
                f"y_lag_count={self.y_lag_count}"
            )

        u_hist = x_enc[:, -self.u_lag_count :, : self.u_dim].reshape(B, -1)
        y_hist = x_enc[:, -self.y_lag_count :, -self.C_out :].reshape(B, -1)
        return torch.cat([u_hist, y_hist], dim=-1)

    def _select_target(self, batch_y: torch.Tensor) -> torch.Tensor:
        if batch_y.ndim == 2:
            return batch_y[:, -self.C_out :]
        return batch_y[:, -1, -self.C_out :].reshape(batch_y.shape[0], self.C_out)

    def _select_target_sequence(self, batch_y: torch.Tensor) -> torch.Tensor:
        if batch_y.ndim == 2:
            return batch_y[:, -self.C_out :].unsqueeze(1)
        return batch_y[:, -self.pred_len :, -self.C_out :]

    def _select_future_u(self, x_enc: torch.Tensor, batch_y: Optional[torch.Tensor]) -> torch.Tensor:
        last_u = x_enc[:, -1:, : self.u_dim].expand(-1, self.pred_len, -1)
        if batch_y is None or batch_y.ndim != 3 or batch_y.shape[-1] < self.u_dim:
            return last_u
        return batch_y[:, -self.pred_len :, : self.u_dim]

    def _roll_history(self, history: torch.Tensor, u_next: torch.Tensor, y_next: torch.Tensor) -> torch.Tensor:
        next_row = torch.cat([u_next, y_next], dim=-1).unsqueeze(1)
        return torch.cat([history[:, 1:, :], next_row], dim=1)

    @torch.no_grad()
    def sample(self, dec_inp: torch.Tensor, num_samples: int = 1, add_noise: bool = True) -> torch.Tensor:
        B = dec_inp.shape[0]
        z_prior = torch.randn(num_samples, B, self.z_dim, device=dec_inp.device, dtype=dec_inp.dtype)
        mean_dec, logvar_dec = self.decoder(z_prior, dec_inp)
        if add_noise:
            std = torch.exp(0.5 * logvar_dec).unsqueeze(0)
            mean_dec = mean_dec + std * torch.randn_like(mean_dec)
        return mean_dec

    @torch.no_grad()
    def generate(self, dec_inp: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        y_pred = self.sample(dec_inp, num_samples or self.num_samples, add_noise=True)

        if self.output_type == "mean":
            return y_pred.mean(dim=0)
        if self.output_type == "median":
            return y_pred.median(dim=0).values
        if self.output_type == "sample":
            return y_pred[0]
        raise ValueError("output_type must be one of ['mean', 'median', 'sample']")

    def short_term_forecasting(self, x_enc: torch.Tensor, batch_y: torch.Tensor, flag: str = "train"):
        if flag == "train":
            history = x_enc
            future_u = self._select_future_u(x_enc, batch_y)
            target_y = self._select_target_sequence(batch_y)
            mean_dec_steps = []
            mu_steps = []
            logvar_steps = []
            logvar_dec = None

            for step in range(self.pred_len):
                dec_inp = self._build_transition_input(history)
                y_next = target_y[:, step, :]
                enc_inp = torch.cat([dec_inp, y_next], dim=-1)
                z_posterior, mu_posterior, logvar_posterior = self.encoder(enc_inp)
                mean_dec, logvar_dec = self.decoder(z_posterior, dec_inp)
                mean_dec_steps.append(mean_dec)
                mu_steps.append(mu_posterior)
                logvar_steps.append(logvar_posterior)
                history = self._roll_history(history, future_u[:, step, :], y_next)

            return {
                "mu_posterior": torch.stack(mu_steps, dim=1),
                "logvar_posterior": torch.stack(logvar_steps, dim=1),
                "mean_dec": torch.stack(mean_dec_steps, dim=2),
                "logvar_dec": logvar_dec,
            }

        history = x_enc
        future_u = self._select_future_u(x_enc, batch_y)
        preds = []

        for step in range(self.pred_len):
            dec_inp = self._build_transition_input(history)
            y_next = self.generate(dec_inp, self.num_samples)
            preds.append(y_next)
            history = self._roll_history(history, future_u[:, step, :], y_next)

        return torch.stack(preds, dim=1)

    def soft_sensor(self, x_enc: torch.Tensor, batch_y: Optional[torch.Tensor] = None, flag: str = "train"):
        if flag == "train" and batch_y is not None:
            return self.short_term_forecasting(x_enc, batch_y, flag=flag)
        dec_inp = self._build_transition_input(x_enc)
        return self.generate(dec_inp, self.num_samples)

    def build_smc_sampler(
        self,
        x_mean: Array,
        x_std: Array,
        y_mean: Array,
        y_std: Array,
        n_particles: Optional[int] = None,
        measurement_std: float | Array = 0.1,
        device: Optional[str] = None,
    ) -> CVAESMCSSampling:
        return CVAESMCSSampling(
            model=self,
            spec=self.spec,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            n_particles=n_particles or self.num_samples,
            measurement_std=measurement_std,
            device=device or next(self.parameters()).device.type,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag="train"):
        if self.task == "short_term_forecasting":
            return self.short_term_forecasting(x_enc, batch_y, flag=flag)
        if self.task == "soft_sensor":
            return self.soft_sensor(x_enc, batch_y=batch_y, flag=flag)
        raise ValueError(f"Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor")
