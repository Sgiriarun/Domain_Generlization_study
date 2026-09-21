#!/usr/bin/env python3
"""Small source-only execution check; never reports target-test performance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_partial_finetuning import (  # noqa: E402
    LODO_ROLES, fit_trial, head_checkpoint_path, joined_roles, load_inputs,
    manifest_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    frame = manifest_frame(); inputs = load_inputs(frame)
    role = joined_roles(frame, LODO_ROLES, ["lodo_PTT-PPG_role"])["lodo_PTT-PPG_role"].to_numpy()
    source_train = np.flatnonzero(role == "source_train")[:48]
    source_validation = np.flatnonzero(role == "source_validation")[:16]
    train = np.zeros(len(frame), dtype=bool); train[source_train] = True
    validation = np.zeros(len(frame), dtype=bool); validation[source_validation] = True
    checkpoint = torch.load(head_checkpoint_path("lodo", "PTT-PPG", None, 17),
                            map_location="cpu", weights_only=False)
    _, record = fit_trial(inputs, frame.hr_bpm.to_numpy(float), frame.dataset.to_numpy(),
                          train, validation, checkpoint, 3e-5, 17,
                          torch.device(args.device), 8, 1, 1)
    assert record["best_epoch"] == 1
    print("PASS: final-block update, source validation and checkpoint selection execute on", args.device)


if __name__ == "__main__":
    main()
