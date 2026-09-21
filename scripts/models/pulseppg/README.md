# Pulse-PPG experiment scripts

Scripts in this directory will implement Phase 12 in the order defined by
`reports/phase12_pulseppg/EXPERIMENT_PLAN.md`.

Every training script must:

1. accept an explicit frozen manifest and split file;
2. record the encoder checkpoint checksum and code commit;
3. fit preprocessing statistics on source-training data only;
4. keep subjects disjoint across train, validation and test;
5. write predictions as well as aggregate metrics;
6. refuse to overwrite a frozen outcome silently.

