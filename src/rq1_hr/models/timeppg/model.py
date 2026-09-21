"""Faithful PyTorch port/adaptation of the official TimePPG-Big float model.

Upstream: https://github.com/eml-eda/q-ppg
Original file: precision_search/model/TimePPG_float.py
License: Apache-2.0. Required RQ1 adaptations are documented in README.md.
"""

from __future__ import annotations

from math import ceil

import torch
from torch import Tensor, nn


TIMEPPG_BIG_CHANNELS = (32, 32, 63, 64, 64, 121, 122, 104, 76, 82, 61)
TIMEPPG_DILATIONS = (2, 2, 1, 4, 4, 8, 8)
TIMEPPG_RECEPTIVE_FIELDS = (5, 5, 5, 9, 9, 17, 17)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, receptive_field: int, dilation: int):
        super().__init__()
        kernel = ceil(receptive_field / dilation)
        padding = ((kernel - 1) * dilation + 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int, padding: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=padding, bias=False),
            nn.AvgPool1d(2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class Regressor(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU6(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class TimePPGBigPPGOnly(nn.Module):
    """TimePPG-Big topology adapted to one 512-sample PPG input."""

    def __init__(self, input_samples: int = 512):
        super().__init__()
        ch = TIMEPPG_BIG_CHANNELS
        dil = TIMEPPG_DILATIONS
        rf = TIMEPPG_RECEPTIVE_FIELDS
        self.features = nn.Sequential(
            TemporalConvBlock(1, ch[0], rf[0], dil[0]),
            TemporalConvBlock(ch[0], ch[1], rf[1], dil[1]),
            ConvBlock(ch[1], ch[2], stride=1, padding=2),
            TemporalConvBlock(ch[2], ch[3], rf[3], dil[3]),
            TemporalConvBlock(ch[3], ch[4], rf[4], dil[4]),
            ConvBlock(ch[4], ch[5], stride=2, padding=2),
            TemporalConvBlock(ch[5], ch[6], rf[5], dil[5]),
            TemporalConvBlock(ch[6], ch[7], rf[6], dil[6]),
            ConvBlock(ch[7], ch[8], stride=4, padding=4),
        )
        self.features.eval()
        with torch.no_grad():
            feature_count = int(self.features(torch.zeros(2, 1, input_samples)).flatten(1).shape[1])
        self.features.train()
        self.regression = nn.Sequential(
            Regressor(feature_count, ch[9]),
            Regressor(ch[9], ch[10]),
            nn.Linear(ch[10], 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("TimePPG expects shape (batch, 1, samples)")
        return self.regression(self.features(values).flatten(1)).squeeze(-1)


class LogCoshLoss(nn.Module):
    """Numerically stable equivalent of the upstream Keras LogCosh loss."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        error = prediction - target
        return torch.mean(error + torch.nn.functional.softplus(-2.0 * error) - torch.log(error.new_tensor(2.0)))
