"""Shift-operator concept bottleneck"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class EnrollmentBaseline(nn.Module):
    def __init__(self, n_speakers: int, h_dim: int):
        super().__init__()
        self.register_buffer("baselines", torch.zeros(n_speakers, h_dim))
        self.register_buffer("counts", torch.zeros(n_speakers))

    @torch.no_grad()
    def fit_from_arrays(self, h: torch.Tensor, speaker_idx: torch.Tensor) -> None:
        self.baselines.zero_()
        self.counts.zero_()
        self.baselines.index_add_(0, speaker_idx, h.to(self.baselines.dtype))
        self.counts.index_add_(0, speaker_idx, torch.ones_like(speaker_idx, dtype=self.counts.dtype))
        seen = self.counts > 0
        self.baselines[seen] /= self.counts[seen].unsqueeze(1)
        if (~seen).any():
            self.baselines[~seen] = self.baselines[seen].mean(dim=0, keepdim=True)

    def forward(self, speaker_idx: torch.Tensor) -> torch.Tensor:
        return self.baselines[speaker_idx]


class ShiftOperatorCBM(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        h_dim: int,
        n_concepts: int,
        n_emotions: int,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = encoder
        self.concept_head = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_concepts),
            nn.Sigmoid(),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(n_concepts, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_emotions),
        )

    def forward(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor,
        intervene_concepts: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        delta = h - baseline
        if scale is not None:
            delta = delta * scale
        concepts = self.concept_head(delta)
        if intervene_concepts is not None:
            keep = torch.isnan(intervene_concepts)
            concepts = torch.where(keep, concepts, intervene_concepts)
        return {
            "h": h,
            "delta": delta,
            "concepts": concepts,
            "emotion_logits": self.emotion_head(concepts),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D_in, H, n_spk, n_c, n_emo = 64, 40, 32, 8, 6, 6

    encoder = nn.Sequential(nn.Linear(D_in, H), nn.Tanh())
    model = ShiftOperatorCBM(encoder, h_dim=H, n_concepts=n_c, n_emotions=n_emo).eval()
    enroll = EnrollmentBaseline(n_speakers=n_spk, h_dim=H)

    spk = torch.randint(0, n_spk, (B,))
    spk_signature = torch.randn(n_spk, D_in)
    x = spk_signature[spk]

    with torch.no_grad():
        enroll.fit_from_arrays(model.encoder(x), spk)
        out = model(x, baseline=enroll(spk))
    assert out["emotion_logits"].shape == (B, n_emo)
    assert out["concepts"].shape == (B, n_c)
    residual = out["delta"].abs().max().item()
    assert residual < 1e-5, f"speaker-only signal must vanish in delta, got {residual}"

    intervention = torch.full((B, n_c), float("nan"))
    intervention[:, 0] = 1.0
    with torch.no_grad():
        out_i = model(x, baseline=enroll(spk), intervene_concepts=intervention)
    assert torch.allclose(out_i["concepts"][:, 0], torch.ones(B))
    assert torch.allclose(out_i["concepts"][:, 1:], out["concepts"][:, 1:])

    print(f"operator_model smoke test OK (max |delta| on speaker-only input = {residual:.2e})")
