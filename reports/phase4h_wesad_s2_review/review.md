# WESAD S2 detector-disagreement review

S2 contains 267 of WESAD's 316 windows with detector disagreement above 10 bpm.
Because 8-second windows overlap every 2 seconds, ten non-overlapping
representative episodes were selected rather than treating every window as an
independent failure.

The plots show raw ECG, the same 5–25 Hz diagnostic filtering, and peaks from
all three detectors. They support visual classification of clipping, baseline
distortion, extra detections, and missed detections. With no supplied WESAD
annotations, the plots cannot establish which detector is correct. These
windows remain quality-flagged rather than automatically deleted or relabelled.

Visual review confirms that the selected examples contain strongly distorted,
multi-deflection ECG morphology. Different detectors select different subsets
of these deflections, so their HR estimates diverge. This supports retaining a
quality flag. It does not support declaring one detector correct without manual
R-peak annotations. Most large disagreements occur in label 0 transition or
unlabelled periods; a smaller number occur in named protocol conditions.

See `selected_disagreement_windows.csv`, `condition_disagreement_summary.csv`,
and `plots/`.
