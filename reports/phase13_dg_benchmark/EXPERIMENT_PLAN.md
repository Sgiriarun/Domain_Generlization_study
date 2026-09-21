# Phase 13 execution plan

## 13A — freeze the protocol before target results

- Domains/environments are dataset identities.
- Use the Phase-7 canonical manifests and Phase-9 source subject roles.
- For each LODO target, train and validate only on the other three datasets.
- Hyperparameters are selected using source validation subjects only.
- Primary selection: mean source-validation MAE; tie-break using worst-source
  validation MAE and then model simplicity.
- No target signal, target label or target-derived statistic enters training,
  early stopping, normalization or selection.
- Use a matched training budget and record wall time and parameter count.

## 13B — implementation sanity checks

Before full experiments, each method must pass:

1. ERM-equivalence when its regularisation coefficient is zero;
2. finite loss and gradient tests on mini-batches from three domains;
3. deterministic repeatability under a fixed seed;
4. proof that every update receives the intended source-domain composition;
5. a leakage audit showing no target rows were loaded;
6. a small synthetic regression test where the intended penalty behaves in the
   expected direction.

## 13C — seed-17 screen

Run ERM plus the representative methods on all four LODO targets. This is a
development screen, not the final result.

A method advances only if it satisfies at least one of these without a major
failure elsewhere:

- improves MAE on at least two targets;
- reduces the worst positive generalisation gap;
- materially reduces a predeclared systematic HR-bin bias;
- reduces large-error tails or improves most subjects.

Do not advance a method based on one favourable target alone.

## 13D — multi-seed confirmation

Repeat ERM and advancing methods with seeds 17, 29 and 43. Report paired changes
against ERM by target and subject, not only mean ± standard deviation across four
heterogeneous targets.

## 13E — mechanism-level failure comparison

For confirmed methods, repeat the frozen Phase-10 analyses:

- HR-bin MAE and signed bias;
- prediction-range compression;
- per-subject change;
- activity × HR and quality strata;
- temporal persistence and large-error tails.

This determines *how* a method helped and whether it addressed our key problem.

## 13F — controlled PTT site comparison

Apply confirmed methods to the three synchronized PTT pairs where the training
protocol permits a leakage-safe comparison. Report whether proximal penalties
fall and whether overall HR performance is preserved.

## 13G — backbone interaction

Only the strongest two existing DG mechanisms are combined with Pulse-PPG. This
small factorial comparison asks whether benefits come from the pretrained
representation, the DG objective, or their interaction.

## 13H — RQ2 decision

Possible conclusions:

1. **Existing method solves the problem:** use it as the principal method or
   focus RQ2 on validation/efficiency rather than inventing a new loss.
2. **Existing methods improve averages but retain conditional bias/site shift:**
   develop HR-conditioned acquisition invariance.
3. **Alignment harms HR:** develop an objective that preserves HR structure
   while removing acquisition information.
4. **Worst-domain optimisation wins:** focus RQ2 on robust source weighting.
5. **No method is stable:** investigate environment definition, data support and
   reliability/uncertainty before proposing a more complex model.

