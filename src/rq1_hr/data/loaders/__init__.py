"""Dataset-specific loader implementations."""
"""Dataset-specific adapters into the canonical :class:`PPGRecord` schema."""

from rq1_hr.data.loaders.bidmc import list_bidmc_records, load_bidmc
from rq1_hr.data.loaders.ppg_dalia import list_ppg_dalia_records, load_ppg_dalia
from rq1_hr.data.loaders.ptt_ppg import list_ptt_ppg_records, load_ptt_ppg
from rq1_hr.data.loaders.wesad import list_wesad_records, load_wesad

__all__ = [
    "list_bidmc_records",
    "list_ppg_dalia_records",
    "list_ptt_ppg_records",
    "list_wesad_records",
    "load_bidmc",
    "load_ppg_dalia",
    "load_ptt_ppg",
    "load_wesad",
]
