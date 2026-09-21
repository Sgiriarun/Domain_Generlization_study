# External validation of WFDB XQRS on PTT-PPG

## Protocol

The unchanged WFDB XQRS detector selected after PPG-DaLiA development validation is applied to all 66 PTT-PPG ECG records. No PTT annotations are used for tuning. Detected peaks are compared with manually verified peaks using one-to-one ±100 ms matching.

Both annotation routes use the same 8-second windows, 2-second step, RR-midpoint assignment, 35–220 bpm interval range, minimum four valid RR intervals, and arithmetic mean HR.

## Overall and activity results

| group | records | reference_peaks | detected_peaks | peak_precision_micro | peak_recall_micro | peak_f1_micro | peak_f1_macro | reference_windows | matched_windows | coverage | hr_mae_bpm | hr_rmse_bpm | hr_bias_bpm | hr_correlation | windows_over_5_bpm | windows_over_10_bpm | windows_over_20_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 66 | 46405 | 46539 | 0.9946 | 0.9975 | 0.9960 | 0.9961 | 15982 | 15982 | 1.0000 | 0.3461 | 2.5150 | 0.2899 | 0.9855 | 274 | 198 | 54 |
| run | 22 | 16565 | 16604 | 0.9942 | 0.9965 | 0.9953 | 0.9953 | 5339 | 5339 | 1.0000 | 0.4268 | 2.6852 | 0.3320 | 0.9811 | 115 | 79 | 28 |
| sit | 22 | 13859 | 13882 | 0.9968 | 0.9984 | 0.9976 | 0.9975 | 5331 | 5331 | 1.0000 | 0.1735 | 2.2499 | 0.1373 | 0.9848 | 34 | 26 | 12 |
| walk | 22 | 15981 | 16053 | 0.9932 | 0.9977 | 0.9954 | 0.9955 | 5312 | 5312 | 1.0000 | 0.4381 | 2.5890 | 0.4006 | 0.9801 | 125 | 93 | 14 |

## Records with highest HR MAE

| record_id | activity | peak_precision | peak_recall | peak_f1 | hr_mae_bpm | hr_rmse_bpm | windows_over_10_bpm | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s12_run | run | 0.9200 | 0.9478 | 0.9337 | 6.9451 | 11.9673 | 67 | 1.0000 |
| s12_walk | walk | 0.9235 | 0.9690 | 0.9457 | 5.3946 | 9.6503 | 55 | 1.0000 |
| s13_sit | sit | 0.9484 | 0.9751 | 0.9616 | 3.1634 | 10.4189 | 23 | 1.0000 |
| s1_walk | walk | 0.9607 | 0.9883 | 0.9743 | 2.3490 | 6.0550 | 20 | 1.0000 |
| s9_walk | walk | 0.9888 | 0.9968 | 0.9928 | 0.9740 | 3.4983 | 12 | 1.0000 |
| s20_run | run | 0.9894 | 0.9947 | 0.9920 | 0.7100 | 2.6191 | 7 | 1.0000 |
| s22_run | run | 0.9937 | 0.9987 | 0.9962 | 0.3053 | 1.1702 | 0 | 1.0000 |
| s3_run | run | 0.9942 | 1.0000 | 0.9971 | 0.2722 | 1.8910 | 1 | 1.0000 |
| s3_walk | walk | 0.9972 | 1.0000 | 0.9986 | 0.2607 | 1.9665 | 4 | 1.0000 |
| s7_run | run | 0.9971 | 0.9985 | 0.9978 | 0.2606 | 1.7736 | 4 | 1.0000 |

## Interpretation

This is the external detector test because PTT-PPG differs from the development dataset in ECG hardware, sampling frequency, attachment, and activity protocol. Results must be judged together using peak accuracy, window HR error, large-error counts, and retained coverage.
