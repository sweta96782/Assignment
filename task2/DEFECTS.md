# Defect investigation

## Defect 1: Incorrect units in the peak-to-peak artifact threshold

**Location:** `sleep_pipeline.py`

**Original code:**
`REJECT_PTP = 500`

**Problem:**
MNE returns EEG signals in volts, but the threshold was specified as 500 microvolts. The code therefore compares voltage measurements against 500 volts instead of 500 microvolts.

**Evidence from the original dataset:**
- Sampling frequency: 100 Hz
- Median peak-to-peak amplitude: 0.00052458 V (524.58 µV)
- Maximum peak-to-peak amplitude: 0.00096884 V (968.84 µV)
- Epochs rejected by the original threshold: 0 out of 2650
- Epochs exceeding 500 µV: 1455 out of 2650

**Effect on the pipeline:**
The intended artifact-rejection procedure is effectively disabled because no epochs exceed the incorrect threshold of 500 V. Consequently, epochs with peak-to-peak amplitudes above the intended 500 µV threshold remain in the dataset.

**Correction:**
Express the threshold in volts: `REJECT_PTP = 500e-6`.

**Decision:**
Retain the intended 500 µV threshold and convert it to volts to match MNE's output units.
