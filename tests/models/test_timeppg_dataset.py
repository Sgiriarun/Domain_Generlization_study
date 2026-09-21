import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rq1_hr.models.timeppg.dataset import ManifestPPGDataset, attach_split_roles


def test_manifest_dataset_loads_expected_tensor(tmp_path):
    path = tmp_path / "signal.npy"
    np.save(path, np.arange(600, dtype=np.float32)[:, None])
    manifest = pd.DataFrame([{
        "signal_path": str(path), "start_sample_64hz": 10,
        "end_sample_64hz": 522, "channel_index": 0, "hr_bpm": 75.0,
    }])
    values, target, index = ManifestPPGDataset(manifest)[0]
    assert values.shape == (1, 512)
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(values.std(correction=0)) == pytest.approx(1.0, abs=1e-6)
    assert target.item() == 75.0
    assert index == 0


def test_attach_split_roles_rejects_unmatched_subject():
    manifest = pd.DataFrame([{"dataset": "A", "subject_id": "S1"}])
    subjects = pd.DataFrame([{"dataset": "A", "subject_id": "S2", "fold": 0}])
    with pytest.raises(RuntimeError):
        attach_split_roles(manifest, subjects, "fold")
