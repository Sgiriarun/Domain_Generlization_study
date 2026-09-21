"""Training utilities for the reference-faithful TimePPG baseline."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = absolute_error_sum = squared_error_sum = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_count = targets.numel()
            error = predictions.detach() - targets
            loss_sum += float(loss.detach()) * batch_count
            absolute_error_sum += float(torch.abs(error).sum())
            squared_error_sum += float(torch.square(error).sum())
            count += batch_count
    if count == 0:
        raise ValueError("empty data loader")
    return {
        "loss": loss_sum / count,
        "mae_bpm": absolute_error_sum / count,
        "rmse_bpm": math.sqrt(squared_error_sum / count),
    }


def predict(
    model: nn.Module, loader: DataLoader, *, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    references, predictions, indices = [], [], []
    with torch.no_grad():
        for inputs, targets, row_indices in loader:
            outputs: Tensor = model(inputs.to(device)).cpu()
            references.append(targets.numpy())
            predictions.append(outputs.numpy())
            indices.append(row_indices.numpy())
    return np.concatenate(references), np.concatenate(predictions), np.concatenate(indices)

