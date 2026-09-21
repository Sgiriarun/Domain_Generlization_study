# Phase 12 — Pulse-PPG pretrained representation benchmark

## Purpose

Phase 12 tests whether self-supervised field pretraining reduces the
target-dependent PPG-HR failures found with supervised TimePPG. It is not a new
domain-generalisation algorithm. It is a modern pretrained baseline evaluated
with the same frozen, target-free protocol.

Pulse-PPG is therefore a **diagnostic bridge to RQ2**, not the final research
contribution. We use it to determine whether scale and generic pretraining alone
solve the failure, or whether a new source-only learning objective is needed.

## Primary question

> Does a frozen Pulse-PPG representation improve unseen-dataset HR estimation
> and reduce the HR-range and sensor-placement failures observed with TimePPG?

## Mechanism under investigation

Our current evidence points to **HR-conditional acquisition shift**:

> For the same ECG-derived HR, PPG morphology and spectral structure change with
> dataset, device and sensor placement. A model can therefore retain acquisition
> shortcuts and compress unfamiliar predictions toward the source-domain middle
> HR range.

Phase 12 tests whether Pulse-PPG already removes this mechanism. If it does not,
the proposed RQ2 direction is source-only, HR-conditioned acquisition-invariant
representation learning. See `RESEARCH_DIRECTION.md`.

## Evidence boundary

- Pulse-PPG was reportedly pretrained without HR labels on a separate 100-day
  field study involving 120 participants.
- The original Pulse-PPG paper evaluated downstream tasks on WESAD and
  PPG-DaLiA. These are evaluation exposure, not reported encoder-pretraining
  data, but the exact released checkpoint and configuration must be audited.
- BIDMC, WESAD, PTT-PPG and PPG-DaLiA are not reported as encoder-pretraining
  datasets. This remains a provenance claim, not proof of participant-level
  non-overlap.
- The primary Phase-12 result uses a frozen encoder and source-only HR labels.
- No held-out target signals or labels may be used during LODO development.

## Directory map

| Folder | Task | Required outcome |
|---|---|---|
| `00_provenance_audit/` | Code, weights, licence and overlap audit | Frozen checkpoint identity and admissible claims |
| `01_input_compatibility/` | Sampling, preprocessing and duration checks | Evidence that 8-second input is technically and physiologically usable |
| `02_frozen_linear_probe/` | Frozen encoder plus linear HR head | Primary within-dataset and LODO baseline |
| `03_head_and_tuning_ablation/` | Nonlinear head, partial and full fine-tuning | Separate representation benefit from adaptation benefit |
| `04_failure_analysis/` | Subject, HR range, bias, activity and quality | Direct comparison with Phase 10 |
| `05_ptt_site_transfer/` | All three synchronized PTT site pairs | Direct comparison with Phase 11 |
| `06_representation_audit/` | Domain information in embeddings | Test whether improved HR accuracy also means greater invariance |
| `07_frozen_outcome/` | Final synthesis and provenance hashes | Presentation- and thesis-ready frozen conclusion |

See `EXPERIMENT_PLAN.md` for questions, controls and stopping rules.
