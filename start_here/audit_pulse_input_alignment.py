"""Read-only comparison of the current window list with saved Pulse-PPG inputs.

Run from the repository root with the project environment active. The script
never changes a manifest or cache and does not waive the byte-hash safeguard.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/models/pulseppg'))
from run_full_linear_probe import batch_windows  # noqa: E402


def main() -> None:
    manifest = pd.read_csv(ROOT/'reports/phase7_frozen_dataset/main_window_manifest.csv', low_memory=False)
    predictions = pd.read_csv(
        ROOT/'reports/phase12_pulseppg/02_frozen_linear_probe/within_test_predictions.csv',
        low_memory=False,
    )
    key = ['dataset', 'record_id', 'subject_id', 'window_index']
    indices = predictions.embedding_index.to_numpy(dtype=np.int64)
    if len(predictions) != len(manifest) or not np.array_equal(np.sort(indices), np.arange(len(manifest))):
        raise RuntimeError('Saved feature indices are not a one-to-one list of all current windows')
    mapped = manifest.iloc[indices].reset_index(drop=True)
    for column in key:
        if not mapped[column].astype(str).equals(predictions[column].astype(str)):
            raise RuntimeError(f'Window identity differs at {column}')
    if not np.allclose(mapped.hr_bpm, predictions.reference_hr_bpm, rtol=0, atol=1e-9):
        raise RuntimeError('Reference HR differs between saved predictions and current windows')

    meta = json.loads((ROOT/'artifacts/pulseppg/phase12_inputs_8s_50hz/metadata.json').read_text())
    saved = np.load(ROOT/'artifacts/pulseppg/phase12_inputs_8s_50hz'/meta['matrix'],
                    mmap_mode='r', allow_pickle=False)
    if saved.shape != (len(manifest), 400):
        raise RuntimeError(f'Saved input matrix has unexpected shape: {saved.shape}')
    largest_difference = 0.0
    mismatched_rows = 0
    for start in range(0, len(manifest), 512):
        stop = min(start + 512, len(manifest))
        current = batch_windows(manifest.iloc[start:stop])
        difference = np.max(np.abs(current - saved[start:stop]), axis=1)
        largest_difference = max(largest_difference, float(difference.max()))
        mismatched_rows += int(np.count_nonzero(difference > 1e-5))
        if stop % 16384 < 512 or stop == len(manifest):
            print(f'Checked {stop:,}/{len(manifest):,} windows', flush=True)
    print(json.dumps({
        'windows': len(manifest), 'identity_and_reference_mismatches': 0,
        'prepared_signal_rows_different_above_1e-5': mismatched_rows,
        'maximum_prepared_signal_difference': largest_difference,
        'note': 'This tests row alignment. It does not make the original byte hashes equal.',
    }, indent=2))
    if mismatched_rows:
        raise RuntimeError('Current prepared PPG windows do not match the saved Pulse-PPG input cache')


if __name__ == '__main__':
    main()
