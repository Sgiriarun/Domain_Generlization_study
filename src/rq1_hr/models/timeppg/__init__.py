"""TimePPG-derived PPG-only supervised baseline."""

from rq1_hr.models.timeppg.dataset import ManifestPPGDataset, attach_split_roles
from rq1_hr.models.timeppg.model import LogCoshLoss, TimePPGBigPPGOnly
from rq1_hr.models.timeppg.trainer import predict, run_epoch

__all__ = [
    "LogCoshLoss",
    "ManifestPPGDataset",
    "TimePPGBigPPGOnly",
    "attach_split_roles",
    "predict",
    "run_epoch",
]
