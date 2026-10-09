"""Loader for the BIDMC PPG and Respiration Dataset (WFDB format)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import wfdb

from rq1_hr.data.schema import PPGRecord, SampledValues
from rq1_hr.data.loaders._common import (
    clean_wfdb_name,
    parse_tagged_comments,
    require_file,
    signal_from_columns,
)


def list_bidmc_records(root: Path) -> tuple[str, ...]:
    root = Path(root)
    return tuple(sorted(path.stem for path in root.glob("bidmc[0-9][0-9].hea")))


def load_bidmc(root: Path, record_id: str) -> PPGRecord:
    """Load one BIDMC record without preprocessing its signals."""
    root = Path(root)
    waveform_header = require_file(root / f"{record_id}.hea")
    numeric_header = require_file(root / f"{record_id}n.hea")
    waveform = wfdb.rdrecord(str(waveform_header.with_suffix("")))
    numeric = wfdb.rdrecord(str(numeric_header.with_suffix("")))

    names = tuple(clean_wfdb_name(name) for name in waveform.sig_name)
    units = tuple(waveform.units)
    ppg_index = names.index("PLETH")
    ecg_index = names.index("II")
    ppg = signal_from_columns(
        waveform.p_signal, [ppg_index], fs_hz=waveform.fs,
        names=[names[ppg_index]], units=[units[ppg_index]],
    )
    ecg = signal_from_columns(
        waveform.p_signal, [ecg_index], fs_hz=waveform.fs,
        names=[names[ecg_index]], units=[units[ecg_index]],
    )

    numeric_names = tuple(clean_wfdb_name(name) for name in numeric.sig_name)
    hr_index = numeric_names.index("HR")
    provided_hr = SampledValues(
        values=np.asarray(numeric.p_signal)[:, hr_index],
        fs_hz=numeric.fs,
        name="monitor_hr",
        unit=numeric.units[hr_index],
    )

    metadata = parse_tagged_comments(waveform.comments)
    source_match = re.search(r"/matched/([^/]+)/", metadata.get("source", ""))
    if source_match is None:
        raise ValueError(f"cannot recover MIMIC patient identity for {record_id}")

    other_indices = [i for i in range(len(names)) if i not in (ppg_index, ecg_index)]
    extras = {
        "waveform_aux": signal_from_columns(
            waveform.p_signal, other_indices, fs_hz=waveform.fs,
            names=[names[i] for i in other_indices],
            units=[units[i] for i in other_indices],
        )
    }
    return PPGRecord(
        dataset="BIDMC",
        record_id=record_id,
        subject_id=source_match.group(1),
        ppg=ppg,
        ecg=ecg,
        provided_hr=provided_hr,
        recording_condition="ICU",
        extra_signals=extras,
        source_files=(waveform_header, root / f"{record_id}.dat", numeric_header, root / f"{record_id}n.dat"),
        metadata=metadata,
    )
