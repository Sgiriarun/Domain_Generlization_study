"""Canonical in-memory data contract for the RQ1 datasets.

This module contains structures only. It deliberately performs no filtering,
resampling, peak detection, HR derivation, windowing, or quality rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class SchemaError(ValueError):
    """Raised when a record violates a structural schema invariant."""


def _positive_finite(value: float, field_name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise SchemaError(f"{field_name} must be finite and > 0; got {value!r}")
    return value


def _finite(value: float, field_name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise SchemaError(f"{field_name} must be finite; got {value!r}")
    return value


def _identifier(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise SchemaError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class SampledSignal:
    """A regularly sampled signal on its native clock.

    ``values`` always has shape ``(n_samples, n_channels)``. A one-dimensional
    input is converted to ``(n_samples, 1)`` as a zero-copy view when possible.
    Non-finite values are allowed here because detecting and reporting signal
    quality is a later phase, not a loading/schema responsibility.
    """

    values: NDArray[Any] = field(repr=False, compare=False)
    fs_hz: float
    channel_names: tuple[str, ...]
    units: tuple[str | None, ...] = ()
    t0_s: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise SchemaError(
                f"signal values must be 1D or 2D, received shape {values.shape}"
            )
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise SchemaError(f"signal values must be non-empty; got {values.shape}")

        channel_names = tuple(_identifier(x, "channel name") for x in self.channel_names)
        if len(channel_names) != values.shape[1]:
            raise SchemaError(
                f"received {len(channel_names)} channel names for "
                f"{values.shape[1]} channels"
            )

        units = tuple(self.units) if self.units else (None,) * values.shape[1]
        if len(units) != values.shape[1]:
            raise SchemaError(
                f"received {len(units)} units for {values.shape[1]} channels"
            )

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "fs_hz", _positive_finite(self.fs_hz, "fs_hz"))
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "t0_s", _finite(self.t0_s, "t0_s"))

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration represented by the samples, using an exclusive end time."""
        return self.n_samples / self.fs_hz

    @property
    def end_time_s(self) -> float:
        return self.t0_s + self.duration_s

    def channel(self, name: str) -> NDArray[Any]:
        """Return a one-dimensional, non-copying view of a named channel."""
        try:
            index = self.channel_names.index(name)
        except ValueError as exc:
            available = ", ".join(self.channel_names)
            raise KeyError(f"unknown channel {name!r}; available: {available}") from exc
        return self.values[:, index]


@dataclass(frozen=True, slots=True)
class EventSeries:
    """Sorted event indices expressed on a named native signal clock."""

    sample_indices: NDArray[np.integer[Any]] = field(repr=False, compare=False)
    fs_hz: float
    name: str
    source: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.sample_indices)
        if indices.ndim != 1:
            raise SchemaError(f"event indices must be 1D; got {indices.shape}")
        if not np.issubdtype(indices.dtype, np.integer):
            raise SchemaError("event indices must use an integer dtype")
        indices = indices.astype(np.int64, copy=False)
        if indices.size and indices[0] < 0:
            raise SchemaError("event indices must be non-negative")
        if indices.size > 1 and np.any(np.diff(indices) <= 0):
            raise SchemaError("event indices must be unique and strictly increasing")
        object.__setattr__(self, "sample_indices", indices)
        object.__setattr__(self, "fs_hz", _positive_finite(self.fs_hz, "fs_hz"))
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        object.__setattr__(self, "source", _identifier(self.source, "event source"))

    @property
    def times_s(self) -> NDArray[np.float64]:
        return self.sample_indices.astype(np.float64) / self.fs_hz


@dataclass(frozen=True, slots=True)
class SampledValues:
    """A regularly sampled one-dimensional target or condition sequence."""

    values: NDArray[Any] = field(repr=False, compare=False)
    fs_hz: float
    name: str
    unit: str | None = None
    t0_s: float = 0.0
    value_map: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 1 or values.size == 0:
            raise SchemaError(
                f"sampled values must be a non-empty 1D array; got {values.shape}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "fs_hz", _positive_finite(self.fs_hz, "fs_hz"))
        object.__setattr__(self, "name", _identifier(self.name, "series name"))
        object.__setattr__(self, "t0_s", _finite(self.t0_s, "t0_s"))
        object.__setattr__(self, "value_map", dict(self.value_map))

    @property
    def duration_s(self) -> float:
        return self.values.size / self.fs_hz


@dataclass(frozen=True, slots=True)
class PPGRecord:
    """One source recording represented without preprocessing.

    ``subject_id`` is the leakage-control identity. It may differ from
    ``record_id`` (notably for BIDMC, where 53 records map to 46 patients).
    """

    dataset: str
    record_id: str
    subject_id: str
    ppg: SampledSignal
    ecg: SampledSignal | None = None
    rpeaks: EventSeries | None = None
    provided_hr: SampledValues | None = None
    conditions: SampledValues | None = None
    recording_condition: str | None = None
    extra_signals: Mapping[str, SampledSignal] = field(default_factory=dict)
    source_files: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _identifier(self.dataset, "dataset"))
        object.__setattr__(self, "record_id", _identifier(self.record_id, "record_id"))
        object.__setattr__(self, "subject_id", _identifier(self.subject_id, "subject_id"))
        if self.recording_condition is not None:
            object.__setattr__(
                self,
                "recording_condition",
                _identifier(self.recording_condition, "recording_condition"),
            )
        if self.rpeaks is not None:
            if self.ecg is None:
                raise SchemaError("rpeaks require an ECG signal")
            if not np.isclose(self.rpeaks.fs_hz, self.ecg.fs_hz):
                raise SchemaError("rpeak and ECG sampling rates must match")
            if (
                self.rpeaks.sample_indices.size
                and self.rpeaks.sample_indices[-1] >= self.ecg.n_samples
            ):
                raise SchemaError("an rpeak index lies outside the ECG signal")

        extras = dict(self.extra_signals)
        if any(not str(name).strip() for name in extras):
            raise SchemaError("extra signal names must not be empty")
        object.__setattr__(self, "extra_signals", extras)
        object.__setattr__(
            self, "source_files", tuple(Path(path) for path in self.source_files)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.record_id}"

    def summary(self) -> dict[str, Any]:
        """Return lightweight provenance without copying signal arrays."""
        return {
            "dataset": self.dataset,
            "record_id": self.record_id,
            "subject_id": self.subject_id,
            "recording_condition": self.recording_condition,
            "ppg_samples": self.ppg.n_samples,
            "ppg_channels": self.ppg.channel_names,
            "ppg_fs_hz": self.ppg.fs_hz,
            "ppg_duration_s": self.ppg.duration_s,
            "ecg_samples": None if self.ecg is None else self.ecg.n_samples,
            "ecg_fs_hz": None if self.ecg is None else self.ecg.fs_hz,
            "rpeak_count": None if self.rpeaks is None else self.rpeaks.sample_indices.size,
            "source_files": tuple(str(path) for path in self.source_files),
        }

    def describe(self, preview_samples: int = 5) -> str:
        """Return a readable, bounded summary for interactive inspection.

        This method deliberately reports structure and a short PPG preview only.
        It never prints complete signal arrays or performs preprocessing.
        """
        if preview_samples < 0:
            raise ValueError("preview_samples must be >= 0")

        def signal_line(label: str, signal: SampledSignal) -> str:
            channels = ", ".join(signal.channel_names)
            return (
                f"  {label:<12} shape={signal.values.shape!s:<18} "
                f"fs={signal.fs_hz:g} Hz  duration={signal.duration_s:.3f} s  "
                f"channels=[{channels}]"
            )

        lines = [
            f"PPGRecord  {self.key}",
            f"  subject      {self.subject_id}",
            f"  condition    {self.recording_condition or 'not provided'}",
            signal_line("PPG", self.ppg),
        ]
        if self.ecg is not None:
            lines.append(signal_line("ECG", self.ecg))
        else:
            lines.append("  ECG          not provided")

        if self.rpeaks is None:
            lines.append("  R-peaks      not provided")
        else:
            lines.append(
                f"  R-peaks      count={self.rpeaks.sample_indices.size}  "
                f"clock={self.rpeaks.fs_hz:g} Hz  source={self.rpeaks.source}"
            )

        if self.provided_hr is None:
            lines.append("  provided HR  not provided")
        else:
            lines.append(
                f"  provided HR  shape={self.provided_hr.values.shape}  "
                f"fs={self.provided_hr.fs_hz:g} Hz  unit={self.provided_hr.unit or 'not provided'}"
            )

        if self.conditions is not None:
            lines.append(
                f"  labels       name={self.conditions.name}  "
                f"shape={self.conditions.values.shape}  fs={self.conditions.fs_hz:g} Hz"
            )

        if self.extra_signals:
            lines.append("  auxiliary")
            for name, signal in self.extra_signals.items():
                lines.append(signal_line(name, signal))

        if preview_samples:
            preview = np.array2string(
                self.ppg.values[:preview_samples],
                precision=4,
                suppress_small=False,
                threshold=preview_samples * self.ppg.n_channels,
            )
            lines.append(f"  PPG preview  first {min(preview_samples, self.ppg.n_samples)} rows:\n{preview}")

        if self.source_files:
            lines.append("  source files")
            lines.extend(f"    - {path}" for path in self.source_files)
        return "\n".join(lines)
