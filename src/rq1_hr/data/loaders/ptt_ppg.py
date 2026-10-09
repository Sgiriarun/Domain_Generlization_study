"""Loader for the Pulse Transit Time PPG WFDB records."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import wfdb

from rq1_hr.data.schema import EventSeries, PPGRecord
from rq1_hr.data.loaders._common import clean_wfdb_name, parse_tagged_comments, require_file, signal_from_columns


def list_ptt_ppg_records(root: Path) -> tuple[str, ...]:
    root = Path(root)
    return tuple(sorted(path.stem for path in root.glob("s[0-9]*_*.hea")))


def load_ptt_ppg(root: Path, record_id: str) -> PPGRecord:
    """Load one PTT-PPG activity record with all six PPG channels."""
    root = Path(root)
    header = require_file(root / f"{record_id}.hea")
    annotation = require_file(root / f"{record_id}.atr")
    record = wfdb.rdrecord(str(header.with_suffix("")))
    ann = wfdb.rdann(str(header.with_suffix("")), "atr")
    names = tuple(clean_wfdb_name(name) for name in record.sig_name)
    units = tuple(record.units)
    ppg_idx = [names.index(f"pleth_{i}") for i in range(1, 7)]
    ecg_idx = names.index("ecg")
    aux_idx = [i for i in range(len(names)) if i not in ppg_idx and i != ecg_idx]
    match = re.fullmatch(r"s(\d+)_(sit|walk|run)", record_id)
    if match is None:
        raise ValueError(f"invalid PTT-PPG record id: {record_id!r}")

    return PPGRecord(
        dataset="PTT-PPG", record_id=record_id, subject_id=f"s{int(match.group(1))}",
        ppg=signal_from_columns(record.p_signal, ppg_idx, fs_hz=record.fs, names=[names[i] for i in ppg_idx], units=[units[i] for i in ppg_idx]),
        ecg=signal_from_columns(record.p_signal, [ecg_idx], fs_hz=record.fs, names=[names[ecg_idx]], units=[units[ecg_idx]]),
        rpeaks=EventSeries(np.asarray(ann.sample), record.fs, "rpeaks", "manual_verified_annotation"),
        recording_condition=match.group(2),
        extra_signals={"sensor_aux": signal_from_columns(record.p_signal, aux_idx, fs_hz=record.fs, names=[names[i] for i in aux_idx], units=[units[i] for i in aux_idx])},
        source_files=(header, root / f"{record_id}.dat", annotation),
        metadata=parse_tagged_comments(record.comments),
    )
