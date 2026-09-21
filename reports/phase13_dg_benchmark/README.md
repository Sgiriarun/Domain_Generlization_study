# Phase 13 — representative target-free DG benchmark for PPG-HR

## Purpose

Phase 13 determines whether established domain-generalisation methods already
resolve the target-, HR-range- and sensor-dependent failures found in RQ1.

This phase prevents a premature method proposal. A new RQ2 method is justified
only if representative existing DG mechanisms leave a clear, reproducible
residual problem.

## Primary question

> Which existing source-only DG mechanisms improve PPG-HR LODO transfer, and do
> they correct the identified failure patterns rather than only pooled MAE?

## Fair-comparison rule

The first benchmark uses the TimePPG-Big PPG-only backbone, frozen manifests,
identical source subjects, source-only validation and the same four LODO targets.
The changed variable is the training strategy. Pulse-PPG is introduced only in a
secondary backbone interaction study for methods that survive screening.

## Work packages

| Folder | Purpose |
|---|---|
| `00_method_audit/` | Verify method definitions, licences and regression adaptations |
| `01_protocol/` | Freeze environments, model-selection rules, grids and metrics |
| `02_seed17_screen/` | Screen representative methods on all four LODO targets |
| `03_multiseed_confirmation/` | Repeat ERM and credible methods with seeds 17, 29 and 43 |
| `04_failure_comparison/` | Compare HR-bin bias, subjects, tails, activity and quality |
| `05_site_comparison/` | Test whether credible methods reduce PTT site sensitivity |
| `06_backbone_interaction/` | Test top methods with Pulse-PPG, if justified |
| `07_frozen_outcome/` | Freeze the supported conclusion and RQ2 decision |

The detailed selection logic is in `METHOD_SELECTION.md`, and the execution
order is in `EXPERIMENT_PLAN.md`.

