"""Hybrid NECO and energy scoring for pixel-wise OOD detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange


@dataclass
class DetectorStats:
    """Fitted detector statistics."""

    mu: torch.Tensor
    basis: torch.Tensor
    neco_mean: torch.Tensor
    neco_std: torch.Tensor
    energy_mean: torch.Tensor
    energy_std: torch.Tensor


class HybridNECOEnergyOOD:
    """Single-forward pixel-wise detector combining NECO geometry and energy."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        num_components: int = 16,
        device: str | torch.device = "cuda",
        temperature: float = 1.0,
        eps: float = 1e-8,
        feature_module_name: str = "outc",
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.model = model
        self.num_classes = num_classes
        self.num_components = num_components
        self.device = torch.device(device)
        self.temperature = temperature
        self.eps = eps
        self.feature_module_name = feature_module_name

        self.model.to(self.device)
        self.model.eval()
        self._cached_feat: torch.Tensor | None = None
        self._hook = self._register_feature_hook(feature_module_name)
        self.stats: DetectorStats | None = None

    def _register_feature_hook(self, module_name: str) -> Any:
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            raise ValueError(f"Model has no module named {module_name!r} for feature capture.")
        return module.register_forward_pre_hook(self._capture_decoder_feat)

    def _capture_decoder_feat(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        del module
        self._cached_feat = inputs[0].detach()

    def _forward_logits_and_feats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._cached_feat = None
        logits = self.model(x)
        if self._cached_feat is None:
            raise RuntimeError("Feature hook did not capture decoder features.")
        return logits, self._cached_feat

    def _valid_flatten(
        self,
        logits: torch.Tensor,
        feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_logits = rearrange(logits, "b c h w -> (b h w) c")
        flat_feats = rearrange(feats, "b d h w -> (b h w) d")
        flat_labels = labels.flatten()
        valid = (flat_labels != 255) & (flat_labels < self.num_classes)
        return flat_logits[valid], flat_feats[valid]

    @torch.no_grad()
    def fit(self, id_loader: Any, max_pixels: int = 250_000) -> DetectorStats:
        """Fit PCA basis and normalization statistics from ID pixels."""

        all_feats: list[torch.Tensor] = []
        all_energy: list[torch.Tensor] = []
        seen_pixels = 0

        self.model.eval()
        for batch_data in id_loader:
            images = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device).clone()
            labels[labels >= self.num_classes] = 255
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            logits, feats = self._forward_logits_and_feats(images)
            valid_logits, valid_feats = self._valid_flatten(logits, feats, labels)
            if valid_feats.numel() == 0:
                continue

            energy = self.temperature * torch.logsumexp(valid_logits / self.temperature, dim=1)
            all_feats.append(valid_feats)
            all_energy.append(energy)
            seen_pixels += valid_feats.shape[0]
            if seen_pixels >= max_pixels:
                break

        if not all_feats:
            raise RuntimeError("No valid ID pixels were found while fitting the detector.")

        feat_mat = torch.cat(all_feats, dim=0)[:max_pixels]
        energy_vec = torch.cat(all_energy, dim=0)[:max_pixels]

        mu = feat_mat.mean(dim=0, keepdim=True)
        centered = feat_mat - mu
        q = min(self.num_components, centered.shape[1], centered.shape[0] - 1)
        if q < 1:
            raise RuntimeError("At least two valid feature vectors are required to fit PCA.")

        _, _, v = torch.pca_lowrank(centered, q=q)
        basis = v[:, :q]
        proj = centered @ basis
        neco = torch.norm(proj, dim=1) / (torch.norm(centered, dim=1) + self.eps)

        self.stats = DetectorStats(
            mu=mu,
            basis=basis,
            neco_mean=neco.mean(),
            neco_std=neco.std().clamp_min(self.eps),
            energy_mean=energy_vec.mean(),
            energy_std=energy_vec.std().clamp_min(self.eps),
        )
        return self.stats

    @torch.no_grad()
    def score_maps(self, x: torch.Tensor, alpha: float = 0.6) -> dict[str, torch.Tensor]:
        """Return NECO, energy, ID, and OOD maps for a batch."""

        if self.stats is None:
            raise RuntimeError("Call fit() or load_state_dict() before score_maps().")

        x = x.to(self.device)
        logits, feats = self._forward_logits_and_feats(x)
        batch, _, height, width = logits.shape

        feat_flat = rearrange(feats, "b d h w -> (b h w) d")
        centered = feat_flat - self.stats.mu
        proj = centered @ self.stats.basis

        neco_flat = torch.norm(proj, dim=1) / (torch.norm(centered, dim=1) + self.eps)
        energy_map = self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        energy_flat = energy_map.flatten()

        neco_z = (neco_flat - self.stats.neco_mean) / self.stats.neco_std
        energy_z = (energy_flat - self.stats.energy_mean) / self.stats.energy_std
        id_score_flat = alpha * neco_z + (1.0 - alpha) * energy_z
        ood_score_flat = -id_score_flat

        return {
            "neco": neco_flat.view(batch, height, width),
            "energy": energy_flat.view(batch, height, width),
            "id_score": id_score_flat.view(batch, height, width),
            "ood_score": ood_score_flat.view(batch, height, width),
        }

    def state_dict(self) -> dict[str, torch.Tensor | float | int | str]:
        """Serialize fitted detector state."""

        if self.stats is None:
            raise RuntimeError("Detector is not fitted.")
        return {
            "num_classes": self.num_classes,
            "num_components": self.num_components,
            "temperature": self.temperature,
            "eps": self.eps,
            "feature_module_name": self.feature_module_name,
            "mu": self.stats.mu.detach().cpu(),
            "basis": self.stats.basis.detach().cpu(),
            "neco_mean": self.stats.neco_mean.detach().cpu(),
            "neco_std": self.stats.neco_std.detach().cpu(),
            "energy_mean": self.stats.energy_mean.detach().cpu(),
            "energy_std": self.stats.energy_std.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load fitted detector statistics."""

        self.num_classes = int(state.get("num_classes", self.num_classes))
        self.num_components = int(state.get("num_components", self.num_components))
        self.temperature = float(state.get("temperature", self.temperature))
        self.eps = float(state.get("eps", self.eps))
        self.stats = DetectorStats(
            mu=state["mu"].to(self.device),
            basis=state["basis"].to(self.device),
            neco_mean=state["neco_mean"].to(self.device),
            neco_std=state["neco_std"].to(self.device),
            energy_mean=state["energy_mean"].to(self.device),
            energy_std=state["energy_std"].to(self.device),
        )

    def cleanup(self) -> None:
        """Remove the feature hook."""

        if self._hook is not None:
            self._hook.remove()
            self._hook = None
