#!/usr/bin/env python3
"""Freeze three-fold outer tests with source-only validation subjects."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from rq1_hr.data.windows import deterministic_subject_folds


SEED = 17
INPUT = Path("reports/phase7_frozen_dataset/subject_splits.csv")
OUTPUT = Path("reports/phase9_timeppg/three_fold_subject_roles.csv")


def stable_order(subjects: list[str], label: str) -> list[str]:
    return sorted(subjects, key=lambda value: hashlib.sha256(f"{SEED}:{label}:{value}".encode()).digest())


def main() -> None:
    source = pd.read_csv(INPUT)[["dataset", "subject_id"]].drop_duplicates()
    parts = []
    for dataset, group in source.groupby("dataset", sort=True):
        result = group.copy()
        outer = deterministic_subject_folds(result.subject_id, n_folds=3, seed=SEED)
        result["within3_outer_fold"] = result.subject_id.map(outer)
        for test_fold in range(3):
            column = f"within3_test_fold_{test_fold}_role"
            test_mask = result.within3_outer_fold.eq(test_fold)
            source_subjects = result.loc[~test_mask, "subject_id"].tolist()
            # Approximately 20% of all subjects, chosen only from non-test subjects.
            validation_count = max(1, round(len(result) * 0.20))
            validation = set(stable_order(source_subjects, f"{dataset}:test{test_fold}")[:validation_count])
            result[column] = "train"
            result.loc[result.subject_id.isin(validation), column] = "validation"
            result.loc[test_mask, column] = "test"
        parts.append(result)
    splits = pd.concat(parts, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUTPUT, index=False)
    rows = []
    for dataset, group in splits.groupby("dataset"):
        for fold in range(3):
            column = f"within3_test_fold_{fold}_role"
            for role, count in group[column].value_counts().items():
                rows.append({"dataset": dataset, "test_fold": fold, "role": role, "subjects": count})
    summary = pd.DataFrame(rows).sort_values(["dataset", "test_fold", "role"])
    summary.to_csv(OUTPUT.with_name("three_fold_role_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
