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

## Defect 2: Unscored annotations incorrectly classified as Wake

**Location:** `sleep_pipeline.py`

**Original code:**
`y[start:stop] = STAGE_MAP.get(desc, 0)`

**Problem:**
The default value of 0 assigns unknown annotations to Wake. However, 0 is a valid Wake label. Unscored annotations and movement-time annotations should not be treated as Wake.

**Evidence from the original dataset:**
- Sleep stage ?: 2177.5 minutes
- Movement time: 17.5 minutes
- Total unknown annotation duration: 2195 minutes

**Effect on the pipeline:**
Unscored and movement-time epochs are incorrectly included as Wake-labelled examples. This introduces label noise and can distort the class distribution and model performance.

**Correction:**
Initialise the label array with -1, use -1 as the default for unknown annotations, and exclude training examples whose target epoch has an invalid label.

**Decision:**
Only the five recognised sleep stages are used as classification targets. Unscored and movement-time annotations are excluded from supervised target labels.

## Defect 3: Sequence target uses the last epoch instead of the centre epoch

**Location:** `SeqDataset.__getitem__` and training-label calculation in `main()`.

**Original code:**
`int(y[i + SEQ_LEN - 1])`

**Problem:**
The documentation states that the model predicts the centre epoch, and the model selects the centre representation using `z[:, L // 2]`. However, the dataset supplies the last epoch's label.

**Evidence from the original dataset:**
For recording 0, sequence starting at index 1011:
- Dataset label: 1 (N1)
- Centre label: 0 (Wake)
- Last label: 1 (N1)

The dataset label matches the last epoch, not the centre epoch.

**Effect on the pipeline:**
The model is trained against a label belonging to a different epoch from the representation selected by its classification head. This introduces incorrect training targets.

**Correction:**
Use `i + SEQ_LEN // 2` for the target in `SeqDataset.__getitem__`, the invalid-target filter, and the training-label count calculation.

**Decision:**
The centre epoch is the intended prediction target, consistent with the model documentation and output head.
