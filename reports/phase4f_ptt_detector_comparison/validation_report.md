# External validation of ECG detector candidates on PTT-PPG

## Protocol

WFDB XQRS, NeuroKit2 Emrich 2023, and SleepECG are applied unchanged to all 66 PTT-PPG ECG records after comparison on PPG-DaLiA. No PTT annotations are used for tuning. Detected peaks are compared with manually verified peaks using one-to-one ±100 ms matching.

Both annotation routes use the same 8-second windows, 2-second step, RR-midpoint assignment, 35–220 bpm interval range, minimum four valid RR intervals, and arithmetic mean HR.

## Overall and activity results

| detector | group | records | reference_peaks | detected_peaks | peak_precision_micro | peak_recall_micro | peak_f1_micro | peak_f1_macro | reference_windows | matched_windows | coverage | hr_mae_bpm | hr_rmse_bpm | hr_bias_bpm | hr_correlation | windows_over_5_bpm | windows_over_10_bpm | windows_over_20_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xqrs | all | 66 | 46405 | 46539 | 0.9946 | 0.9975 | 0.9960 | 0.9961 | 15982 | 15982 | 1.0000 | 0.3461 | 2.5150 | 0.2899 | 0.9855 | 274 | 198 | 54 |
| xqrs | run | 22 | 16565 | 16604 | 0.9942 | 0.9965 | 0.9953 | 0.9953 | 5339 | 5339 | 1.0000 | 0.4268 | 2.6852 | 0.3320 | 0.9811 | 115 | 79 | 28 |
| xqrs | sit | 22 | 13859 | 13882 | 0.9968 | 0.9984 | 0.9976 | 0.9975 | 5331 | 5331 | 1.0000 | 0.1735 | 2.2499 | 0.1373 | 0.9848 | 34 | 26 | 12 |
| xqrs | walk | 22 | 15981 | 16053 | 0.9932 | 0.9977 | 0.9954 | 0.9955 | 5312 | 5312 | 1.0000 | 0.4381 | 2.5890 | 0.4006 | 0.9801 | 125 | 93 | 14 |
| emrich2023 | all | 66 | 46405 | 46395 | 0.9990 | 0.9988 | 0.9989 | 0.9989 | 15982 | 15982 | 1.0000 | 0.0764 | 0.8473 | 0.0247 | 0.9983 | 53 | 27 | 7 |
| emrich2023 | run | 22 | 16565 | 16555 | 0.9991 | 0.9985 | 0.9988 | 0.9988 | 5339 | 5339 | 1.0000 | 0.0851 | 0.7167 | 0.0010 | 0.9986 | 15 | 6 | 0 |
| emrich2023 | sit | 22 | 13859 | 13858 | 0.9986 | 0.9986 | 0.9986 | 0.9986 | 5331 | 5331 | 1.0000 | 0.0884 | 1.1409 | 0.0432 | 0.9960 | 28 | 13 | 7 |
| emrich2023 | walk | 22 | 15981 | 15982 | 0.9992 | 0.9992 | 0.9992 | 0.9992 | 5312 | 5312 | 1.0000 | 0.0556 | 0.5806 | 0.0300 | 0.9990 | 10 | 8 | 0 |
| sleepecg | all | 66 | 46405 | 46545 | 0.9948 | 0.9978 | 0.9963 | 0.9964 | 15982 | 15982 | 1.0000 | 0.3548 | 2.6819 | 0.3115 | 0.9836 | 296 | 198 | 61 |
| sleepecg | run | 22 | 16565 | 16607 | 0.9941 | 0.9966 | 0.9954 | 0.9953 | 5339 | 5339 | 1.0000 | 0.4608 | 2.7571 | 0.3613 | 0.9801 | 135 | 89 | 31 |
| sleepecg | sit | 22 | 13859 | 13892 | 0.9968 | 0.9991 | 0.9979 | 0.9979 | 5331 | 5331 | 1.0000 | 0.1924 | 2.8552 | 0.1808 | 0.9758 | 27 | 27 | 18 |
| sleepecg | walk | 22 | 15981 | 16046 | 0.9938 | 0.9979 | 0.9958 | 0.9959 | 5312 | 5312 | 1.0000 | 0.4112 | 2.4120 | 0.3926 | 0.9828 | 134 | 82 | 12 |

## Records with highest HR MAE

| detector | record_id | activity | peak_precision | peak_recall | peak_f1 | hr_mae_bpm | hr_rmse_bpm | windows_over_10_bpm | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| emrich2023 | s13_sit | sit | 0.9703 | 0.9751 | 0.9727 | 1.3324 | 5.1303 | 13 | 1.0000 |
| emrich2023 | s12_run | run | 0.9834 | 0.9794 | 0.9814 | 1.0332 | 2.8286 | 4 | 1.0000 |
| emrich2023 | s12_walk | walk | 0.9887 | 0.9901 | 0.9894 | 0.6918 | 2.5789 | 8 | 1.0000 |
| emrich2023 | s2_run | run | 0.9991 | 0.9939 | 0.9965 | 0.4472 | 1.7705 | 2 | 1.0000 |
| emrich2023 | s5_sit | sit | 1.0000 | 0.9969 | 0.9984 | 0.1833 | 1.0736 | 0 | 1.0000 |
| sleepecg | s12_run | run | 0.9075 | 0.9437 | 0.9253 | 7.5128 | 12.1509 | 67 | 1.0000 |
| sleepecg | s12_walk | walk | 0.9314 | 0.9746 | 0.9525 | 4.4820 | 7.9635 | 46 | 1.0000 |
| sleepecg | s13_sit | sit | 0.9324 | 0.9834 | 0.9572 | 3.9932 | 13.4532 | 27 | 1.0000 |
| sleepecg | s1_walk | walk | 0.9546 | 0.9870 | 0.9705 | 2.8979 | 6.8995 | 20 | 1.0000 |
| sleepecg | s9_walk | walk | 0.9888 | 0.9968 | 0.9928 | 1.0096 | 3.6150 | 12 | 1.0000 |
| xqrs | s12_run | run | 0.9200 | 0.9478 | 0.9337 | 6.9451 | 11.9673 | 67 | 1.0000 |
| xqrs | s12_walk | walk | 0.9235 | 0.9690 | 0.9457 | 5.3946 | 9.6503 | 55 | 1.0000 |
| xqrs | s13_sit | sit | 0.9484 | 0.9751 | 0.9616 | 3.1634 | 10.4189 | 23 | 1.0000 |
| xqrs | s1_walk | walk | 0.9607 | 0.9883 | 0.9743 | 2.3490 | 6.0550 | 20 | 1.0000 |
| xqrs | s9_walk | walk | 0.9888 | 0.9968 | 0.9928 | 0.9740 | 3.4983 | 12 | 1.0000 |

## Interpretation

This is the external detector test because PTT-PPG differs from the development dataset in ECG hardware, sampling frequency, attachment, and activity protocol. Selection must be based on consistency across both datasets using peak accuracy, window HR error, large-error counts, coverage, runtime, and reproducibility.
