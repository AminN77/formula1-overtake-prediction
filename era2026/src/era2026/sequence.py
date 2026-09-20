"""
A recurrent hazard model over battle episodes.

Every other model here sees a battle-lap as an independent row with
hand-computed history (gap_delta_1, gap_mean_3, closing_laps, battle_lap). This
one sees the episode as a sequence and learns its own summary of the past.

The comparison is deliberately fair: identical features, identical target,
identical walk-forward protocol. The only thing that changes is whether the
temporal structure is hand-engineered or learned.

The expectation is that it loses. Roughly 300 positive events in an early
training fold is not enough to learn a temporal encoder that hand-crafted
aggregates already capture with zero parameters, and gradient-boosted trees are
hard to beat on heterogeneous tabular data. The experiment is run and reported
because a measured negative result is worth more than an untested assumption,
and because the one setting where it could win is pretraining the encoder on
the pre-2026 era, which this module supports.

Causality: a GRU is unidirectional and the hazard head reads the hidden state at
lap t, so lap t's prediction depends only on laps up to t. No future leaks in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from era2026.features import FEATURE_TIERS

DEFAULT_HIDDEN = 32
DEFAULT_EPOCHS = 60
DEFAULT_LR = 3e-3


def to_sequences(frame: pd.DataFrame, columns: list[str]) -> tuple:
    """Group rows into per-episode sequences, padded to the longest episode."""
    episodes = []
    for episode_id, group in frame.groupby("episode_id", sort=False):
        ordered = group.sort_values("lap_number")
        episodes.append((
            ordered[columns].astype(float).fillna(0.0).to_numpy(np.float32),
            ordered["label"].to_numpy(np.float32),
            ordered.index.to_numpy(),
        ))
    if not episodes:
        return np.zeros((0, 1, len(columns)), np.float32), np.zeros((0, 1), np.float32), \
               np.zeros((0, 1), np.float32), np.zeros((0, 1), np.int64)

    longest = max(len(x) for x, _, _ in episodes)
    n = len(episodes)
    X = np.zeros((n, longest, len(columns)), np.float32)
    y = np.zeros((n, longest), np.float32)
    mask = np.zeros((n, longest), np.float32)
    index = np.full((n, longest), -1, np.int64)
    for i, (features, labels, original) in enumerate(episodes):
        length = len(features)
        X[i, :length] = features
        y[i, :length] = labels
        mask[i, :length] = 1.0
        index[i, :length] = original
    return X, y, mask, index


@dataclass
class RecurrentHazard:
    """GRU encoder over the episode, linear hazard head on each step."""

    name: str = "gru"
    columns: list[str] = field(default_factory=lambda: list(FEATURE_TIERS))
    hidden: int = DEFAULT_HIDDEN
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    seed: int = 42
    pretrained_state: dict | None = None
    freeze_encoder: bool = False
    model_: object | None = field(default=None, repr=False)
    mean_: np.ndarray | None = field(default=None, repr=False)
    std_: np.ndarray | None = field(default=None, repr=False)

    def _build(self, n_features: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(self, n_in, hidden):
                super().__init__()
                self.gru = nn.GRU(n_in, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out).squeeze(-1)

        return Net(n_features, self.hidden)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecurrentHazard":
        import torch
        from torch import nn

        frame = X.copy()
        frame["label"] = np.asarray(y, dtype=float)
        seq, target, mask, _ = to_sequences(frame, self.columns)

        flat = seq.reshape(-1, seq.shape[-1])[mask.reshape(-1) > 0]
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        seq = (seq - self.mean_) / self.std_
        seq *= mask[..., None]

        model = self._build(seq.shape[-1])
        if self.pretrained_state:
            model.load_state_dict(self.pretrained_state, strict=False)
            if self.freeze_encoder:
                for p in model.gru.parameters():
                    p.requires_grad = False

        positives = target[mask > 0].sum()
        negatives = mask.sum() - positives
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        optimiser = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=self.lr, weight_decay=1e-4
        )

        xb = torch.from_numpy(seq)
        yb = torch.from_numpy(target)
        mb = torch.from_numpy(mask)

        model.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            logits = model(xb)
            loss = (loss_fn(logits, yb) * mb).sum() / mb.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        self.model_ = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        frame = X.copy()
        if "label" not in frame.columns:
            frame["label"] = 0.0
        seq, _, mask, index = to_sequences(frame, self.columns)
        seq = (seq - self.mean_) / self.std_
        seq *= mask[..., None]

        with torch.no_grad():
            logits = self.model_(torch.from_numpy(seq)).numpy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        out = pd.Series(0.0, index=X.index)
        flat_index, flat_p = index.reshape(-1), probabilities.reshape(-1)
        keep = flat_index >= 0
        out.loc[flat_index[keep]] = flat_p[keep]
        return out.to_numpy()

    def encoder_state(self) -> dict:
        return {k: v.clone() for k, v in self.model_.state_dict().items() if k.startswith("gru")}


def pretrain_on_legacy(columns: list[str], hidden: int = DEFAULT_HIDDEN,
                       epochs: int = DEFAULT_EPOCHS, seed: int = 42) -> dict:
    """Learn the encoder on pre-2026 episodes, then hand it to the in-era model.

    This is the one configuration where a sequence model plausibly beats trees:
    the encoder learns what a closing battle looks like from four seasons, while
    only the small hazard head is exposed to the fourteen-race in-era sample.
    """
    from era2026.transfer import LEGACY_TARGET, load_legacy

    legacy = load_legacy()
    if "pit_stop_involved" in legacy.columns:
        legacy = legacy[~legacy["pit_stop_involved"].astype(bool)]

    usable = [c for c in columns if c in legacy.columns]
    legacy = legacy.copy()
    legacy["label"] = legacy[LEGACY_TARGET].astype(int)
    legacy["episode_id"] = (
        legacy["race_name"].astype(str) + "|" + legacy["year"].astype(str) + "|"
        + legacy["attacker"].astype(str) + ">" + legacy["defender"].astype(str)
    )
    for column in columns:
        if column not in legacy.columns:
            legacy[column] = 0.0

    model = RecurrentHazard(name="pretrain", columns=columns, hidden=hidden,
                            epochs=epochs, seed=seed)
    model.fit(legacy, legacy["label"])
    return model.encoder_state()
