"""CRNN bottleneck with the speaker subtraction at one of five depths"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

N_POSITIONS = 5
_CONV_BLOCK_SIZE = 4


def _utt_profile(a: torch.Tensor) -> torch.Tensor:
    return a.mean(dim=-1) if a.dim() == 4 else a


def _apply(a: torch.Tensor, m: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if a.dim() == 4:
        return a - w.view(-1, 1, 1, 1) * m.unsqueeze(-1)
    return a - w.view(-1, 1) * m


class MultiPositionCRNN(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        conv = encoder.conv
        self.block1 = nn.Sequential(*list(conv)[0:_CONV_BLOCK_SIZE])
        self.block2 = nn.Sequential(*list(conv)[_CONV_BLOCK_SIZE:2 * _CONV_BLOCK_SIZE])
        self.block3 = nn.Sequential(*list(conv)[2 * _CONV_BLOCK_SIZE:3 * _CONV_BLOCK_SIZE])
        self.gru = encoder.gru
        self.proj = encoder.proj

    def forward(self, x: torch.Tensor,
                baselines: Optional[List[Optional[torch.Tensor]]] = None,
                gate: Optional[torch.Tensor] = None,
                taps_for: Optional[set] = None) -> Dict[str, torch.Tensor]:
        b = baselines or [None] * N_POSITIONS
        g = gate if gate is not None else x.new_zeros(N_POSITIONS)
        want = set(range(N_POSITIONS)) if taps_for is None else set(taps_for)
        taps: List[Optional[torch.Tensor]] = []

        z = x
        for l, block in enumerate([None, self.block1, self.block2, self.block3]):
            if block is not None:
                z = block(z)
            taps.append(_utt_profile(z).detach() if l in want else None)
            if b[l] is not None:
                z = _apply(z, b[l], g[l].expand(z.shape[0]))

        z = z.mean(dim=2).transpose(1, 2)
        out, _ = self.gru(z)
        h = self.proj(out.mean(dim=1))
        taps.append(h.detach())
        if b[4] is not None:
            h = _apply(h, b[4], g[4].expand(h.shape[0]))
        return {"h": h, "taps": taps}


class PositionGatedCBM(nn.Module):
    def __init__(self, encoder: nn.Module, h_dim: int, n_concepts: int,
                 n_emotions: int, dropout: float = 0.25,
                 fixed_position: Optional[int] = None, tau: float = 1.0):
        super().__init__()
        self.encoder = MultiPositionCRNN(encoder)
        self.fixed_position = fixed_position
        self.tau = tau
        if fixed_position is None:
            self.pi = nn.Parameter(torch.zeros(N_POSITIONS))
        else:
            onehot = torch.zeros(N_POSITIONS)
            onehot[fixed_position] = 1.0
            self.register_buffer("pi", onehot)
        self.concept_head = nn.Sequential(
            nn.Linear(h_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, n_concepts), nn.Sigmoid(),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(n_concepts, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_emotions),
        )

    def active_positions(self) -> List[int]:
        if self.fixed_position is not None:
            return [self.fixed_position]
        return list(range(N_POSITIONS))

    def gate(self) -> torch.Tensor:
        if self.fixed_position is not None:
            return self.pi
        return F.softmax(self.pi / self.tau, dim=0)

    def forward(self, x: torch.Tensor, baselines: List[Optional[torch.Tensor]],
                intervene_concepts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        g = self.gate()
        want = set(self.active_positions()) | {N_POSITIONS - 1}
        enc = self.encoder(x, baselines=baselines, gate=g, taps_for=want)
        h = enc["h"]
        concepts = self.concept_head(h)
        if intervene_concepts is not None:
            keep = torch.isnan(intervene_concepts)
            concepts = torch.where(keep, concepts, intervene_concepts)
        return {"h": h, "h_raw": enc["taps"][N_POSITIONS - 1], "delta": h,
                "concepts": concepts,
                "emotion_logits": self.emotion_head(concepts),
                "gate": g, "taps": enc["taps"]}


@torch.no_grad()
def fit_position_baselines(model: PositionGatedCBM, loader, device: str,
                           n_speakers: int) -> List[Optional[torch.Tensor]]:
    was_training = model.training
    model.eval()
    prev = getattr(model, "_baselines", None)
    active = set(model.active_positions())
    sums: List[Optional[torch.Tensor]] = [None] * N_POSITIONS
    counts = torch.zeros(n_speakers, device=device)
    for batch in loader:
        x = batch["x"].to(device)
        spk = batch["speaker_local"].to(device)
        b = [None if prev is None or prev[l] is None else prev[l][spk]
             for l in range(N_POSITIONS)]
        taps = model.encoder(x, baselines=b, gate=model.gate(), taps_for=active)["taps"]
        for l, t in enumerate(taps):
            if l not in active:
                continue
            t = t.float()
            if sums[l] is None:
                sums[l] = torch.zeros((n_speakers,) + t.shape[1:], device=device)
            sums[l].index_add_(0, spk, t)
        counts.index_add_(0, spk, torch.ones_like(spk, dtype=counts.dtype))
    seen = counts > 0
    denom = counts.clamp(min=1)
    out: List[Optional[torch.Tensor]] = []
    for l in range(N_POSITIONS):
        if sums[l] is None:
            out.append(None)
            continue
        m = sums[l] / denom.view((-1,) + (1,) * (sums[l].dim() - 1))
        if (~seen).any():
            m[~seen] = m[seen].mean(dim=0, keepdim=True)
        out.append(m)
    model._baselines = out
    if was_training:
        model.train()
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import main_egemaps_baseline_deviation_cbm as nc

    torch.manual_seed(0)
    B, M, T, H, S = 8, 64, 100, 192, 4
    enc = nc.CRNNEncoder(n_mels=M, h_dim=H, dropout=0.0).eval()
    x = torch.randn(B, 1, M, T)
    spk = torch.randint(0, S, (B,))

    with torch.no_grad():
        plain = enc(x)
    for pos in range(N_POSITIONS):
        m = PositionGatedCBM(nc.CRNNEncoder(n_mels=M, h_dim=H, dropout=0.0),
                             h_dim=H, n_concepts=6, n_emotions=6,
                             fixed_position=pos).eval()
        m.encoder.block1.load_state_dict(nn.Sequential(*list(enc.conv)[0:4]).state_dict())
        m.encoder.block2.load_state_dict(nn.Sequential(*list(enc.conv)[4:8]).state_dict())
        m.encoder.block3.load_state_dict(nn.Sequential(*list(enc.conv)[8:12]).state_dict())
        m.encoder.gru.load_state_dict(enc.gru.state_dict())
        m.encoder.proj.load_state_dict(enc.proj.state_dict())
        with torch.no_grad():
            zero = [torch.zeros((S,) + t.shape[1:]) for t in
                    m.encoder(x, baselines=None, gate=m.gate())["taps"]]
            out = m(x, baselines=[z[spk] for z in zero])
        assert torch.allclose(out["h"], plain, atol=1e-5), f"pos {pos} zero-baseline drift"
        assert abs(float(out["gate"].sum()) - 1.0) < 1e-6
        print(f"pos {pos}: taps ok, gate {out['gate'].tolist()}")
    print("position_gate smoke test OK")
