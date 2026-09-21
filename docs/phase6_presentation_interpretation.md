# Phase 6 presentation interpretation

This document preserves the complete interpretation of the Phase 6 figures for
future slides, thesis writing, and discussion with the supervisor. Numerical
values must continue to be regenerated from the Phase 6 CSV files.

## Overall conclusion

Phase 6 gives useful research findings. The main result is that PPG domains are
heterogeneous even after common preprocessing and HR matching.

## 1. HR distributions

Source figure: `reports/phase6_distribution_shift/figures/hr_distributions.png`.

| Dataset | Median HR | Main observation |
|---|---:|---|
| WESAD | 75.52 bpm | Lower central HR |
| PPG-DaLiA | 85.16 bpm | Widest high-HR range |
| PTT-PPG | 85.48 bpm | More controlled HR range |
| BIDMC | 88.95 bpm | Relatively narrow clinical range |

Finding:

> The datasets do not contain the same HR distribution.

This is label/support shift. A model trained on one dataset may not have enough
examples for the HR range found in another dataset.

## 2. Overall dataset PCA

Source figure:
`reports/phase6_distribution_shift/figures/normalized_feature_pca.png`.

- Dataset clusters overlap.
- However, their distributions are not identical.
- BIDMC has a visibly different region.
- PPG-DaLiA and WESAD are relatively closer.

This PCA uses HR-matched and z-normalized features. Therefore, the separation is
not explained only by HR or signal amplitude.

Finding:

> Different datasets produce different PPG morphology and spectral
> characteristics for similar HR.

## 3. Device-based PCA

Source figure:
`reports/phase6_distribution_shift/figures/normalized_feature_pca_by_device.png`.

- Clinical pulse-oximeter PPG is more distinct.
- MAX30101 PPG has a different spread.
- PPG-DaLiA and WESAD are closer because both use Empatica E4.
- PPG-DaLiA and WESAD are still not identical despite using the same device.

Finding:

> The recording device creates strong domain differences, but device alone does
> not explain everything.

The remaining differences may come from activity, protocol, sensor attachment,
subjects, recording environment, and signal quality.

## 4. Recording-environment PCA

Source figure:
`reports/phase6_distribution_shift/figures/environment_domain_pca.png`.

- Daily-life, controlled-activity, laboratory, and ICU recordings have different
  distributions.
- The clinical ICU domain is particularly distinguishable.
- Laboratory and daily-life domains overlap more.

Finding:

> A model trained only on wearable or controlled data may face a large shift
> when applied to clinical ICU data.

Important limitation: each environment comes from a different dataset and
usually a different device. Therefore, environment, device, population, and
dataset are confounded. We cannot say that environment alone caused the
separation.

## 5. Sensor-position PCA

Source figure:
`reports/phase6_distribution_shift/figures/ptt_sensor_site_pca.png`.

- The same subject was measured simultaneously near the fingertip and near the
  finger base.
- Matching optical channels move to different PCA centroids when sensor position
  changes.
- This comparison uses the same dataset, device type, time, subject, activity,
  and HR.

Finding:

> Even a small change in sensor position on the same finger changes the PPG
> feature distribution.

This is one of the strongest controlled domain-shift findings because fewer
factors are changing. It directly supports the need for device- and
placement-robust HR models.

## 6. PTT-PPG activity PCA

Source figure:
`reports/phase6_distribution_shift/figures/ptt_activity_pca.png`.

- Sitting, walking, and running overlap substantially after HR balancing.
- Sitting appears more compact in some regions.
- Walking and running show broader or noisier distributions.
- There are also some clear outlying signal regions.

Finding:

> Activity affects PPG, but activity groups cannot be separated into simple
> clean clusters after controlling HR.

Activity affects PPG through two paths:

```text
Activity -> physiological HR change
Activity -> motion and sensor-contact change
```

We controlled HR approximately, so the remaining structure is more likely
related to motion, contact, and waveform quality. However, PCA alone cannot
prove this.

## 7. PPG-DaLiA daily-life PCA

Source figure:
`reports/phase6_distribution_shift/figures/dalia_activity_pca.png`.

- Most daily-life activities overlap strongly.
- Cycling and stairs contain some more extreme regions.
- Driving, lunch, working, and baseline are not completely separable.
- No single activity produces one unique PPG distribution.

Finding:

> Real daily-life PPG variability is complex and cannot be explained by
> activity name alone.

This indicates that subject variation, movement intensity, sensor attachment,
and individual behaviour may be more important than the activity label itself.

## 8. WESAD stress-condition PCA

Source figure:
`reports/phase6_distribution_shift/figures/wesad_stress_condition_pca.png`.

- Baseline, stress, amusement, and meditation overlap strongly.
- Stress has some extreme observations, but it does not form an independent
  cluster.
- After HR balancing, stress condition is not easily identified from these PPG
  features alone.

Finding:

> Stress-related PPG differences are weaker than device/environment differences
> in the current feature space.

This does not mean stress has no physiological effect. It means the effect is not
a simple global separation in the first two PCA components after controlling HR.

## 9. Signal-quality PCA

Source figure:
`reports/phase6_distribution_shift/figures/signal_quality_domain_pca.png`.

- Low, medium, and high spectral-concentration groups show visible structure.
- High-concentration windows form more regular regions.
- Low-concentration windows spread more widely.
- The groups still overlap.

Finding:

> Signal regularity is an important domain factor and can change the PPG feature
> distribution.

These groups were created using within-dataset spectral-concentration tertiles.
They are not supplied labels and are not confirmed motion-artifact classes.

Correct conclusion:

> Spectral quality affects PPG representation.

Unsupported conclusion to avoid:

> Every low-concentration window is corrupted by motion.

## Combined Phase 6 findings

### Finding 1: label/support shift

Different datasets contain different HR ranges:

```text
P(Y) differs
```

### Finding 2: raw covariate shift

Signal amplitude and scale strongly depend on the device:

```text
P(X) differs
```

A dataset classifier achieved 76.95% accuracy using raw-scale features, compared
with 25% chance.

### Finding 3: normalization does not remove domain shift

After HR matching and z-normalization, dataset classification was still 64.64%.

> Domain information remains inside normalized waveform shape and spectral
> structure.

### Finding 4: conditional waveform shift

For approximately the same HR, PPG characteristics differ between datasets:

```text
P(X | Y) differs
```

This is highly relevant because a model may associate a waveform pattern with a
particular HR in one domain, but encounter a different waveform pattern for the
same HR in another domain.

### Finding 5: device and environment shifts appear stronger

The PCA figures suggest that device/environment differences are stronger than
simple activity or stress-condition separation.

Research direction:

> Cross-device and cross-environment generalisation should be the primary RQ1
> analysis, while activity, stress, sensor position, and quality should be used
> to explain where model errors occur.

### Finding 6: sensor placement matters

The PTT experiment shows that sensor position changes PPG even within the same
finger.

This supports the idea that a model should learn device- and site-invariant
cardiac information rather than depending heavily on waveform appearance.

## What Phase 6 does not prove

PCA cannot prove:

- that one domain causes model failure;
- that one activity has worse HR accuracy;
- that `P(Y|X)` definitely changes;
- that low spectral concentration means motion; or
- that one domain is statistically better or worse.

PCA is an unsupervised two-dimensional projection. Separation is useful
evidence, but overlap or separation can be affected by selected features and
scaling.

## Research contribution from Phase 6

Recommended exact wording:

> After applying identical filtering, resampling, HR matching, and amplitude
> normalization, PPG representations remain domain-identifiable. Differences
> arise across datasets, devices, recording environments, sensor positions,
> activities, and signal-regularity conditions. Device/environment and
> sensor-position shifts appear stronger than stress or activity separation in
> the current feature space.

This supports RQ1 because it provides evidence that cross-dataset evaluation is
necessary.

## What Phase 7 must establish

```text
Train HR estimator in source domain
                |
                v
Test on unseen target domain
                |
                v
Measure generalisation loss
                |
                v
Connect errors to device, activity, site, and quality
```

If within-dataset performance is good but unseen-dataset performance becomes
worse, we can conclude that the measured domain shifts produce real
model-generalisation failure.

Final exact wording:

> Phase 6 shows that heterogeneous domains exist. Phase 7 must measure how much
> these heterogeneous domains affect PPG-based HR estimation.

