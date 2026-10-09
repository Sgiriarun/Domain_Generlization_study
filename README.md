# Estimating heart rate from PPG across different recording conditions


This project estimates heart rate from contact PPG (an optical pulse signal) and
investigates why prediction errors change between recordings. We use BIDMC,
WESAD, PTT-PPG and PPG-DaLiA, with ECG-derived heart rate as the reference.

## Start here

**Open [the step-by-step guide](start_here/README.md).** It explains the software
setup, what each Python file does, the files it needs, and where results are saved.
## Overview of complete Experiments carried 


![Overview of completed experiments and findings](docs/PPG%20Heart-Rate%20Estimation%20Experimental%20Overview.png)

## Complete code Workflow

The following diagram illustrates how the research code, datasets, models, experiments, and results are connected.

![Overview of the existing research code and its connections](docs/diagram.png)


The diagram shows connections between parts of the code, not an exact command
order. Its technical labels are explained in ordinary English in the
[numbered guide](start_here/README.md#the-work-in-order).

## Where things are kept

| Folder | What it contains |
|---|---|
| `scripts/` | Files that run preparation, experiments and analysis |
| `src/rq1_hr/` | Shared calculations, signal readers and model code used by those files |
| `datasets/raw/` | Original recordings; obtain separately under the dataset terms |
| `reports/` | Existing research results, tables and figures; preserve these |
| `artifacts/` | Model files and supporting saved files, where present |
| `study_results/` | Destination for new runs where output redirection is supported |
| `tests/` | Automated implementation checks |
