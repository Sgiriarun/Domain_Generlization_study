# Frozen Phase 11 outcome: PTT pair 1/4 bidirectional site analysis

## Frozen scope

This conclusion applies to PTT-PPG pair `pleth_1`/`pleth_4`, TimePPG-Big
PPG-only, seed 17, the frozen preprocessing and the frozen three-fold subject
roles. It covers 15,982 synchronized windows from 22 subjects.

## Controlled design

For each checkpoint and window, proximal and distal absolute errors use the same
subject, activity, recording time and ECG-derived HR. These variables are
controlled by exact pairing rather than only by statistical adjustment. Both
training directions are included.

## Frozen result

- Proximal penalty when trained distal: **+3.109 bpm**
  (subject-bootstrap 95% interval +2.009 to
  +4.204).
- Proximal penalty when trained proximal: **+2.647 bpm**
  (+1.744 to
  +3.665).
- Bidirectional subject-macro proximal penalty: **+2.878 bpm**
  (+1.926 to +3.881).
- The bidirectional proximal penalty is positive for
  **20/22 subjects**.

## Frozen interpretation

> For PTT pair 1/4, proximal input is consistently less usable for TimePPG than
> synchronized distal input. The disadvantage remains across both training
> directions while subject, activity, time and HR label are held constant.

This supports a substantial placement/channel contribution. It does not isolate
an anatomical site cause from physical-channel, optical or hardware differences,
and it is not a universal claim about every PPG device, wavelength or model.
Pairs 2/5 and 3/6 are optional external-sensitivity extensions, not prerequisites
for this frozen pair-1/4 conclusion.
