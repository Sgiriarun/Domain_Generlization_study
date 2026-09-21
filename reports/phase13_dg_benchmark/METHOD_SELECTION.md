# Selection of representative DG methods

## Why not run every method in the survey taxonomy?

The taxonomy contains many methods designed for image classification, discrete
classes, generative image models or settings that access target data. Running
all names would create an unfair and uninterpretable benchmark.

We select methods by mechanism, not popularity. A method must:

1. operate without target-domain samples;
2. support multiple labelled source domains;
3. admit a defensible continuous-regression formulation;
4. work with a one-dimensional PPG encoder;
5. have enough implementation detail for reproduction;
6. test a mechanism relevant to our observed failure.

The taxonomy source is Wang et al., *Generalizing to Unseen Domains: A Survey on
Domain Generalization*, IEEE TKDE, arXiv:2103.03097. DomainBed is an
implementation reference, not code that can be copied unchanged: its standard
networks and losses are primarily image-classification oriented.

## Core benchmark

| Method | DG family | Source-only regression adaptation | What it tests here |
|---|---|---|---|
| ERM | Reference | Pooled log-cosh HR loss | Whether complexity beats a strong baseline |
| Inter-domain Mixup | Data manipulation | Mix signals/features and continuous HR labels across source domains | Whether interpolated acquisition variation improves transfer |
| CORAL | Explicit alignment | Align source-domain embedding covariance; no target features | Whether second-order domain mismatch drives failure |
| MMD | Explicit alignment | Minimise pairwise source-domain kernel discrepancy | Whether distribution alignment improves transfer |
| DANN | Domain adversarial | Predict source dataset through gradient reversal while regressing HR | Whether removing source identity improves transfer |
| GroupDRO | Robust optimisation | Upweight the source dataset with the highest HR loss | Whether protecting the worst source improves unseen targets |
| VREx | Invariant risk | Penalise variance of HR risk across source datasets | Whether equal source risks improve transfer |
| IRM | Invariant risk | Regression-compatible IRMv1 penalty by source environment | Whether one invariant HR predictor exists across sources |
| MLDG | Meta-learning | Simulate source-domain train/test shifts during each update | Whether learning to survive held-out source domains transfers |

## Important interpretation cautions

- CORAL, MMD and DANN were often introduced for domain adaptation. They become
  source-only DG variants here by aligning **only the observed source domains**.
  The held-out target is never supplied.
- IRM can fail even in simple settings; it is included as a mechanism test, not
  assumed to be superior.
- Global alignment may remove HR information because our source datasets have
  different HR distributions. This is a key reason to inspect HR-bin bias and a
  possible motivation for conditional alignment.
- Classification-oriented default hyperparameters are not accepted blindly.
  Each regression adaptation and search range must be frozen in `01_protocol/`.

## Optional methods

Fish/Fishr, RSC, SagNet, SelfReg, CAD/CondCAD, EQRM and newer risk-distribution
methods may be added only after the core benchmark, if a specific unresolved
mechanism justifies them. Their absence from the first screen is not a claim
that they are unimportant.

## DA methods excluded from the primary benchmark

Any method using unlabelled PPG from the held-out target is domain adaptation,
not our target-free DG setting. Such methods may be discussed in the literature
or evaluated in a clearly separate upper-bound experiment, but cannot be mixed
with the primary comparison.

