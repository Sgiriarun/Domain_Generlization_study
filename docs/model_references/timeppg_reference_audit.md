# TimePPG reference audit for RQ1

## Reference identity

- Paper: Alessio Burrello et al., *Embedding Temporal Convolutional Networks
  for Energy-Efficient PPG-Based Heart Rate Monitoring*.
- DOI: https://doi.org/10.1145/3487910
- Open preprint: https://arxiv.org/abs/2203.04396
- Official code: https://github.com/eml-eda/q-ppg
- Audited commit: `ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba`
- Code license: Apache-2.0.

## Released standard found in the code

The audit used the authors' files `architecture_search/config.py`,
`architecture_search/preprocessing/preprocessing_Dalia.py`,
`precision_search/model/TimePPG_float.py`, and
`precision_search/config_Dalia.json`.

| Component | Official implementation |
| --- | --- |
| Window | 8 seconds |
| Step | 2 seconds |
| Sampling rate | PPG reduced from 64 to 32 Hz |
| Standard input | 4 channels: 1 PPG + 3 wrist ACC axes |
| Input length | 256 samples per channel |
| Evaluation | 15-fold PPG-DaLiA leave-one-subject-out |
| Network family | TimePPG/TEMPONet temporal convolutional network |
| Feature blocks | Dilated Conv1d, ReLU6, BatchNorm1d |
| Reduction blocks | Conv1d, average pooling, ReLU6, BatchNorm1d |
| Head | Two dense regression blocks and one scalar output |
| Loss | log-cosh |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 128 |
| Maximum epochs | 200 in the released precision-search configuration |
| Early stopping | 20 epochs, monitoring validation loss |
| Published variants | TimePPG Big, Medium, and Small channel schedules |

The architecture-search code also explicitly lists a `ppg_only_1` input option.
However, the released fixed `TimePPG_float` class is constructed for four input
channels. Therefore a one-channel experiment necessarily changes its first
layer and is not an unchanged reproduction of the reported multimodal result.

## Frozen RQ1 adaptation

| Component | RQ1 decision | Reason |
| --- | --- | --- |
| Input | One PPG channel | Only signal common to all four datasets |
| Sampling rate | 64 Hz, 512 samples | Already frozen by supervisor protocol and Phase 7 |
| Architecture | Official TimePPG-Big block order, dilation, kernels, channel schedule, activations and regression head | Use released model rather than inventing a network |
| First layer | Change 4 input channels to 1 | Required for PPG-only comparison |
| Flattened head size | Derive from 512-sample input or use an explicitly verified equivalent | Required because the official model expects 256 samples |
| Loss | log-cosh | Match official training configuration |
| Optimizer | Adam, learning rate 0.001 | Match official configuration |
| Batch size | 128 | Match official configuration |
| Maximum epochs | 200 | Match official released PyTorch configuration |
| Early stopping | Patience 20 on source-validation loss | Match official configuration without target leakage |
| Evaluation | Frozen subject folds and four LODO tests | Central RQ1 requirement |

## Correct claim

Use this wording in the thesis and presentation:

> We implemented a PPG-only adaptation of the officially released TimePPG-Big
> temporal convolutional architecture. We retained its published block design,
> channel schedule, log-cosh loss, Adam optimiser, batch size and early-stopping
> standard. Input channels, sampling rate and evaluation protocol were changed
> to satisfy the common four-dataset RQ1 protocol. Consequently, the experiment
> is reference-faithful but not an exact reproduction of the paper's reported
> multimodal PPG-DaLiA result.

Never call our result the reproduced TimePPG paper score. Its value is a strong,
standardised supervised architecture evaluated fairly under a new cross-dataset
protocol.

