"""Loader for WESAD subject pickle files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rq1_hr.data.schema import PPGRecord, SampledSignal, SampledValues
from rq1_hr.data.loaders._common import load_pickle, require_file


def list_wesad_records(root: Path) -> tuple[str, ...]:
    root = Path(root)
    return tuple(sorted((p.name for p in root.glob("S[0-9]*") if (p / f"{p.name}.pkl").is_file()), key=lambda x: int(x[1:])))


def load_wesad(root: Path, record_id: str) -> PPGRecord:
    """Load one WESAD subject and preserve every array on its native clock."""
    source = require_file(Path(root) / record_id / f"{record_id}.pkl")
    data = load_pickle(source)
    chest, wrist = data["signal"]["chest"], data["signal"]["wrist"]

    ppg = SampledSignal(wrist["BVP"], 64, ("BVP",), (None,))
    ecg = SampledSignal(chest["ECG"], 700, ("ECG",), (None,))
    conditions = SampledValues(
        np.asarray(data["label"]), 700, "protocol_label",
        value_map={0: "not_defined", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation", 5: "not_defined", 6: "not_defined", 7: "not_defined"},
    )
    extras = {
        "wrist_acc": SampledSignal(wrist["ACC"], 32, ("ACC_X", "ACC_Y", "ACC_Z"), (None, None, None)),
        "wrist_eda": SampledSignal(wrist["EDA"], 4, ("EDA",), (None,)),
        "wrist_temp": SampledSignal(wrist["TEMP"], 4, ("TEMP",), (None,)),
        "chest_acc": SampledSignal(chest["ACC"], 700, ("ACC_X", "ACC_Y", "ACC_Z"), (None, None, None)),
        "chest_resp": SampledSignal(chest["Resp"], 700, ("RESP",), (None,)),
    }
    return PPGRecord(
        dataset="WESAD", record_id=record_id, subject_id=str(data["subject"]),
        ppg=ppg, ecg=ecg, conditions=conditions,
        recording_condition="laboratory protocol", extra_signals=extras,
        source_files=(source,),
    )
