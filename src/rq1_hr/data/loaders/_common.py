"""Small shared helpers for source-format loaders."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rq1_hr.data.schema import SampledSignal


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required dataset file not found: {path}")
    return path


def clean_wfdb_name(name: str) -> str:
    """Remove punctuation that some WFDB headers append to channel names."""
    return name.strip().rstrip(",")


def signal_from_columns(
    values: np.ndarray,
    indices: Iterable[int],
    *,
    fs_hz: float,
    names: Iterable[str],
    units: Iterable[str | None],
) -> SampledSignal:
    idx = tuple(indices)
    return SampledSignal(
        values=np.asarray(values)[:, idx],
        fs_hz=fs_hz,
        channel_names=tuple(names),
        units=tuple(units),
    )


def parse_tagged_comments(comments: Iterable[str]) -> dict[str, str]:
    """Parse WFDB comments containing ``<key>: value`` fields."""
    text = " ".join(comments)
    matches = list(re.finditer(r"<([^>]+)>:\s*", text))
    fields: dict[str, str] = {}
    for pos, match in enumerate(matches):
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
        fields[match.group(1).strip()] = text[match.end() : end].strip()
    return fields


def load_pickle(path: Path) -> dict[str, Any]:
    """Load a trusted local dataset pickle with Python-2 compatibility."""
    import pickle

    with require_file(path).open("rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    if not isinstance(value, dict):
        raise ValueError(f"expected a dictionary in {path}, got {type(value).__name__}")
    return value
