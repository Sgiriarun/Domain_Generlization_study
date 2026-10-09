"""Loader for PPG-DaLiA subject pickle files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rq1_hr.data.schema import EventSeries, PPGRecord, SampledSignal, SampledValues
from rq1_hr.data.loaders._common import load_pickle, require_file


def list_ppg_dalia_records(root: Path) -> tuple[str, ...]:
    root = Path(root)
    return tuple(sorted((p.name for p in root.glob("S[0-9]*") if (p / f"{p.name}.pkl").is_file()), key=lambda x: int(x[1:])))


def load_ppg_dalia(root: Path, record_id: str) -> PPGRecord:
    """Load one PPG-DaLiA subject without altering source signals or labels."""
    source = require_file(Path(root) / record_id / f"{record_id}.pkl")
    data = load_pickle(source)
    chest, wrist = data["signal"]["chest"], data["signal"]["wrist"]
    questionnaire = dict(data.get("questionnaire", {}))
    raw_rpeaks = np.asarray(data["rpeaks"])
    unique_rpeaks = np.unique(raw_rpeaks)
    questionnaire["source_rpeak_count"] = int(raw_rpeaks.size)
    questionnaire["source_rpeak_duplicate_count"] = int(
        raw_rpeaks.size - unique_rpeaks.size
    )

    return PPGRecord(
        dataset="PPG-DaLiA",
        record_id=record_id,
        subject_id=str(data["subject"]),
        ppg=SampledSignal(wrist["BVP"], 64, ("BVP",), (None,)),
        ecg=SampledSignal(chest["ECG"], 700, ("ECG",), (None,)),
        rpeaks=EventSeries(unique_rpeaks, 700, "rpeaks", "dataset_annotation"),
        provided_hr=SampledValues(np.asarray(data["label"]), 0.5, "provided_hr", "bpm"),
        conditions=SampledValues(
            np.asarray(data["activity"]).reshape(-1), 4, "activity_label",
            value_map={
                0: "no_activity", 1: "baseline", 2: "stairs",
                3: "table_soccer", 4: "cycling", 5: "driving",
                6: "lunch", 7: "walking", 8: "working",
            },
        ),
        recording_condition="daily-life activities",
        extra_signals={
            "wrist_acc": SampledSignal(wrist["ACC"], 32, ("ACC_X", "ACC_Y", "ACC_Z"), (None, None, None)),
            "wrist_eda": SampledSignal(wrist["EDA"], 4, ("EDA",), (None,)),
            "wrist_temp": SampledSignal(wrist["TEMP"], 4, ("TEMP",), (None,)),
            "chest_acc": SampledSignal(chest["ACC"], 700, ("ACC_X", "ACC_Y", "ACC_Z"), (None, None, None)),
            "chest_resp": SampledSignal(chest["Resp"], 700, ("RESP",), (None,)),
        },
        source_files=(source,),
        metadata=questionnaire,
    )
