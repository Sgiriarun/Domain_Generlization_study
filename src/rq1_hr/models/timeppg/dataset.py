"""Manifest-backed input dataset for the TimePPG baseline."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rq1_hr.data.windows import zscore_window


class ManifestPPGDataset(Dataset[tuple[Tensor, Tensor, int]]):
    """Read frozen signal slices lazily and normalize each window independently."""

    def __init__(self, manifest: pd.DataFrame, *, cache_size: int = 128):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self._signals: OrderedDict[str, np.ndarray] = OrderedDict()
        required = {
            "signal_path", "start_sample_64hz", "end_sample_64hz",
            "channel_index", "hr_bpm",
        }
        missing = required.difference(self.manifest.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def _signal(self, path: str) -> np.ndarray:
        if path not in self._signals:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            self._signals[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            self._signals.move_to_end(path)
            while len(self._signals) > self.cache_size:
                self._signals.popitem(last=False)
        return self._signals[path]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        row = self.manifest.iloc[index]
        signal = self._signal(str(row.signal_path))
        values = signal[
            int(row.start_sample_64hz):int(row.end_sample_64hz),
            int(row.channel_index),
        ]
        normalized = zscore_window(values)
        return (
            torch.from_numpy(normalized.copy()).unsqueeze(0),
            torch.tensor(float(row.hr_bpm), dtype=torch.float32),
            int(index),
        )


def attach_split_roles(
    manifest: pd.DataFrame, subjects: pd.DataFrame, role_column: str
) -> pd.DataFrame:
    """Attach one frozen split role and assert a many-window-to-one-subject join."""
    if role_column not in subjects.columns:
        raise KeyError(role_column)
    roles = subjects[["dataset", "subject_id", role_column]]
    result = manifest.merge(
        roles, on=["dataset", "subject_id"], how="left", validate="many_to_one"
    )
    if result[role_column].isna().any():
        raise RuntimeError("some windows have no subject split role")
    return result
