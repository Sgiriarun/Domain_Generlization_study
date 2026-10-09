# PPG Heart-Rate Estimation: Experiment Guide

This README shows how to follow the experiments, run the Python scripts and find the results. Steps **00–24** are listed in the order of the work.

## Get started

**Requirements:** Python 3.11 or later. Model training may need a GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If `.venv` already exists, just activate it. Run commands from the repository's main folder.

To check the installation:

```bash
python start_here/00_check_required_software.py --run
```

**Using the numbered scripts:** Without `--run`, a script only shows what it would do. Add `--run` to execute it. Training scripts also need `--allow-training`. New outputs go to `study_results/`; use a new `--results-dir` to avoid an existing output folder.

Example:

```bash
python start_here/05_prepare_reference_heart_rates.py --use-saved-inputs --run
```

`--use-saved-inputs` means the script reads the original saved files instead of generating all earlier inputs again. Some steps currently require this option.

## Data and saved models

Download the four datasets using:

```bash
bash scripts/download_datasets.sh all datasets
```

 The expected folders are:

```text
datasets/raw/ppg_dalia/data/PPG_FieldStudy/
datasets/raw/wesad/WESAD/
datasets/raw/ptt_ppg/
datasets/raw/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0/
```

A GitHub download may not include the recordings, result files or model weights. Follow each dataset's access terms.
Only load trusted checkpoint files.

## Experiments in order

Each file is under `start_here/`. **Training** means a model is fitted again. Other steps may read saved data, make calculations or create figures.

| Step | What it does | Trains a model? |
|---|---|---|
| `00_check_required_software.py` | Checks installed packages and runs tests | No |
| `01_download_the_recordings.py` | Downloads dataset files | No |
| `02_check_recordings_and_participants.py` | Checks recordings and counts participants | No |
| `03_compare_heartbeat_detection_methods.py` | Tests ECG heartbeat detection and reference-HR methods | No |
| `04_investigate_heart_rate_disagreements.py` | Examines disagreements in reference heart rates | No |
| `05_prepare_reference_heart_rates.py` | Creates reference HR labels | No |
| `06_prepare_ppg_signals.py` | Prepares PPG signals for analysis | No |
| `07_compare_the_four_datasets.py` | Compares datasets and fits a dataset-identification classifier | **Yes, classifier only** |
| `08_compare_signals_at_similar_heart_rates.py` | Collects saved comparisons of PPG signals at similar HR | No |
| `09_explore_activity_and_signal_quality.py` | Shows effects of activity and signal quality | No |
| `10_choose_windows.py` | Selects windows and creates earlier data splits | No |
| `11_estimate_heart_rate_using_signal_frequency.py` | Estimates HR using the frequency-based baseline | No |
| `12_test_timeppg_on_new_participants.py` | Trains 12 TimePPG models with three-fold participant splits | **Yes** |
| `13_test_timeppg_on_a_new_dataset.py` | Trains four TimePPG models, each tested on an excluded dataset | **Yes** |
| `14_find_where_prediction_errors_increase.py` | Analyses saved TimePPG errors | No |
| `15_compare_training_and_test_heart_rates.py` | Compares HR ranges in training and testing | No |
| `16_repeat_the_ptt_experiment.py` | Summarises three saved PTT runs; does not retrain | No |
| `17_compare_fingertip_and_finger_base_signals.py` | Compares saved predictions from two PTT recording locations | No |
| `18_check_the_other_channel_pairs.py` | Compares saved results for three PTT channel pairs | No |
| `19_check_the_pretrained_pulseppg_model.py` | Checks whether prepared inputs work with the saved Pulse-PPG model | No |
| `20_test_pulseppg_with_a_simple_predictor.py` | Trains a linear HR predictor on saved Pulse-PPG features | **Yes** |
| `21_test_pulseppg_with_a_neural_network_predictor.py` | Trains neural-network HR predictors on saved features | **Yes** |
| `22_train_the_last_part_of_pulseppg.py` | Trains the final part of Pulse-PPG | **Yes** |
| `23_compare_models_and_remaining_errors.py` | Compares saved Pulse-PPG predictions and creates four figures | No |
| `24_create_result_tables_and_figures.py` | Lists existing result figures; does not recreate them all | No |

## Notes and common problems

- **Missing `rq1_hr`:** Run from the repository root and install the project, or use `PYTHONPATH=src`.
- **Missing recordings or CSV files:** Check dataset extraction and the output of earlier steps.
- **Missing model weights:** Obtain the correct model and its settings. Do not substitute another checkpoint.
- **No GPU:** Many checks can run on CPU, but full training may be slow.
- **Running again:** The numbered scripts refuse to overwrite an existing step folder. Other original scripts may overwrite outputs, so check their destinations first.



For detailed scripts and output paths, see the existing `scripts/` folders.