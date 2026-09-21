# RQ1 dataset inventory

Verified 2026-09-02 against the official dataset hosts.

| Dataset | Version / subjects | PPG and ECG | Conditions | Official source | Access |
|---|---|---|---|---|---|
| PPG-DaLiA | UCI dataset 495; 15 subjects | Wrist BVP 64 Hz; chest ECG 700 Hz | Daily-life activities | https://archive.ics.uci.edu/dataset/495/ppg+dalia | Public, CC BY 4.0; 2.7 GB ZIP |
| WESAD | UCI dataset 465; 15 subjects | Wrist BVP 64 Hz; chest ECG 700 Hz | Baseline, stress, amusement and protocol segments | https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html | Public for scientific, non-commercial use with attribution; 2.5 GB ZIP |
| Pulse Transit Time PPG | PhysioNet 1.1.0; 22 subjects, 66 records | Six finger PPG channels 500 Hz; ECG 500 Hz with verified R peaks | Sit, walk, run | https://physionet.org/content/pulse-transit-time-ppg/1.1.0/ | Open access, ODbL 1.0; 2.9 GB uncompressed |
| BIDMC PPG and Respiration | PhysioNet 1.0.0; 53 eight-minute ICU records | PPG and ECG 125 Hz; monitor HR 1 Hz | Clinical recordings; no activity labels | https://physionet.org/content/bidmc/1.0.0/ | Open access, ODC Attribution 1.0; 207.7 MB uncompressed |

## Important compatibility notes

- PPG-DaLiA and WESAD use the same device families and sampling rates. They are
  still different protocols, but their device-domain shift is weaker than their
  inclusion as separate datasets may initially suggest.
- PTT-PPG version 1.1.0 supersedes 1.0.0 and corrects duplicated diastolic blood
  pressure metadata. Use 1.1.0.
- BIDMC has recordings rather than guaranteed one-record-per-person subject IDs.
  The `Fix.txt`/MAT metadata must be checked before calling the 53 records 53
  independent subjects.
- For a PPG-only benchmark, choose and document one PTT-PPG wavelength/site.
  A reasonable primary channel is distal green (`pleth_3`), with other channels
  reserved for sensitivity analysis.

