# EEG Deep Learning Assignment

This repository contains the code, experiments, audits and analysis completed for the EEG deep-learning assignment. The work is organised into two parts:

- **Task 1 – Motor-imagery EEG:** data-quality auditing, physiological analysis, classification experiments and model interpretation.
- **Task 2 – Sleep staging:** inspection and correction of a sleep-staging pipeline, subject-level evaluation, defect-impact experiments and error analysis.

The experiments were developed primarily in **Google Colab** for notebook-based analysis. Task 2 also includes a standalone Python pipeline and an automated impact-ledger script.

---

## Repository contents

```text
Assignment/
├── README.md
├── task1/
│   └── Task 1 notebooks, analysis scripts, experiment outputs and result tables
└── task2/
    ├── sleep_pipeline.py
    ├── DEFECTS.md
    ├── Decisions.md
    ├── impact_ledger.py
    ├── impact_ledger_output.txt
    └── fixed_output.txt
```

The Task 1 folder contains the motor-imagery analysis and experimental artefacts. Task 2 files listed above are the principal pipeline, audit documentation and reproducibility outputs.

## Task 1 – Motor-imagery EEG analysis

### Objective

Investigate whether the EEG recordings contain motor-imagery-related information, assess potential non-motor confounds, and compare deep-learning approaches for left- versus right-fist imagery classification.

### Data and audit

The analysis uses the PhysioNet EEG Motor Movement/Imagery dataset through MNE. The audited recordings focus on runs 4, 8 and 12:

- T1: left-fist motor imagery
- T2: right-fist motor imagery
- Rest events are not treated as imagery trials.

The audit checks include:

- Successful EDF loading and recording/run availability.
- Sampling frequency and recording consistency across participants.
- Annotation/event-code counts and event-to-class mapping.
- Per-subject and per-run imagery trial counts.
- Identification and documentation of anomalies before model training.

### Analysis and experiments

The Task 1 work includes:

1. **Physiological analysis (A2):** examination of mu- and beta-band event-related desynchronisation (ERD), including sensorimotor lateralisation at C3/C4 and subject-level statistical assessment.
2. **Confound analysis (A3):** restricted visual- and ocular-channel decoding experiments to assess whether non-motor information could contribute to classification performance.
3. **Baseline model comparison (B1):** comparison of EEGNet with a Transformer-based model using a subject-separated evaluation design.
4. **Robustness and scaling experiments (B2.1):** evaluation across training-subject counts and multiple random seeds.
5. **Tokenisation experiments (B2.2a):** comparison of channel tokens with and without learned channel identity and time-patch tokenisation.
6. **Electrode-position perturbation (B2.2b):** assessment of model predictions after controlled displacement of electrode coordinates and interpolation.
7. **Model attribution (B2.3):** analysis of channel/time importance and deletion-based checks of whether highly ranked tokens affect predictions.

The experiments are intended to test specific hypotheses, not merely maximise accuracy. Results that do not match the initial hypothesis are retained and discussed as findings.

## Task 2 – Sleep-staging pipeline

### Objective

Audit an EEG sleep-staging pipeline, identify implementation and evaluation defects, correct the pipeline, and quantify the effect of reintroducing individual defects.

### Data and task definition

The pipeline uses the Sleep-EDF Expanded Sleep Cassette data accessed through MNE. It works with 30-second epochs and predicts one of five sleep stages:

- W – Wake
- N1 – Stage 1
- N2 – Stage 2
- N3 – Stages 3 and 4 merged
- REM – Rapid eye movement sleep

The model receives a sequence of 11 consecutive epochs (5.5 minutes of context) and predicts the **centre epoch**.

### Model architecture

The implemented model consists of:

- A three-stage 1D CNN epoch encoder (16, 32 and 64 feature channels).
- Batch normalisation, ReLU activations and temporal pooling.
- Positional encoding applied to the sequence of epoch embeddings.
- A two-layer Transformer encoder with four attention heads, embedding dimension 64 and feed-forward dimension 128.
- A five-class linear classification head applied to the centre sequence representation.

### Audit and corrections

The investigation documents seven defects in `task2/DEFECTS.md` and the rationale for the selected corrections in `task2/Decisions.md`:

| ID | Defect investigated | Correction |
|---|---|---|
| D1 | Peak-to-peak artifact threshold used the wrong units | Express the 500 µV threshold in volts (`500e-6`). |
| D2 | Unscored and movement annotations were mapped to Wake | Preserve invalid labels as `-1` and exclude them as targets. |
| D3 | Sequence target used the final epoch rather than the centre epoch | Use the centre index consistently. |
| D4 | Transformer input dimensions did not match the batch-first tensor layout | Set `batch_first=True`. |
| D5 | Positional encoding was applied after self-attention | Add positional information before the Transformer. |
| D6 | Per-epoch normalisation suppressed between-epoch amplitude variation | Standardise per recording and channel across epochs/time. |
| D7 | Recording-level splitting allowed participant overlap | Split by subject so a participant's recordings stay in one partition. |

### Experimental evaluation

The corrected pipeline uses a subject-level train/test split. The saved run reports:

- Training subjects: 12
- Testing subjects: 3
- Training recordings: 24
- Testing recordings: 5
- Subject overlap: none

The saved output reports accuracy, macro-F1, Cohen's kappa, per-class F1 and a confusion matrix. The impact-ledger experiments compare fixed and deliberately reintroduced-defect variants using matched random seeds.

**Important evaluation limitation:** in the current version of `task2/sleep_pipeline.py`, the test-set macro-F1 is calculated after each training epoch and is used to select the best checkpoint. This exposes the test set to model selection. Therefore, the saved test metrics (including the reported 0.889 accuracy and 0.815 macro-F1) should be treated as **provisional, not as an unbiased final test estimate**. A validation split at the subject level should be used for checkpoint selection, followed by a single final evaluation on untouched test subjects.

### Impact-ledger experiment

Run `impact_ledger.py` from the `task2` directory to create temporary variants of the fixed pipeline, reintroduce one defect at a time, execute matched-seed comparisons and produce a ranked impact table.

The script is designed not to overwrite `sleep_pipeline.py`. Its saved output is in `impact_ledger_output.txt`. The comparisons are one-defect-at-a-time experiments; interactions between multiple defects were not established by these runs.

## Running Task 2

Run from the repository's Task 2 directory in an environment with the required Python packages installed:

```bash
cd task2
python sleep_pipeline.py
```

To run the defect impact experiments:

```bash
python impact_ledger.py
```

The pipeline obtains Sleep-EDF recordings through MNE's PhysioNet dataset interface. The first run may therefore require dataset download and sufficient disk space.

## Experimental environment and dependencies

- **Notebook environment:** Google Colab was used for notebook-based experiments and analysis.
- **Task 2 execution:** standalone Python script using PyTorch, MNE, NumPy and scikit-learn.
- **Task 1 tooling:** Python ecosystem including MNE, NumPy, Pandas, Matplotlib and deep-learning libraries used by the notebooks.

A pinned `requirements.txt` is not currently provided. Package versions and hardware can affect reproducibility; record the exact versions and accelerator details in the execution environment when reproducing the experiments. Do not assume that the same GPU is available in every Colab session.

## Reproducibility and interpretation notes

- Subject-level separation is important for estimating generalisation to previously unseen participants.
- Reported performance should always be interpreted alongside the split strategy, class distribution and per-class metrics.
- Accuracy alone can conceal weak performance on less frequent classes; macro-F1 and confusion matrices are also reported.
- Random-seed experiments provide an indication of variability, but a small number of seeds does not establish robust statistical certainty.
- The impact ledger measures isolated defect effects. It does not establish interactions among defects.
- Physiological and attribution analyses are exploratory evidence; they should not be interpreted as proof that a model relies exclusively on the intended neural mechanism.

## Key artefacts

| File | Purpose |
|---|---|
| `task2/sleep_pipeline.py` | Sleep-staging data pipeline, model, training and evaluation |
| `task2/DEFECTS.md` | Evidence and description of the seven investigated defects |
| `task2/Decisions.md` | Rationale for the corrective implementation decisions |
| `task2/impact_ledger.py` | Automated one-defect-at-a-time impact experiment |
| `task2/impact_ledger_output.txt` | Saved impact-ledger results |
| `task2/fixed_output.txt` | Saved output from the corrected subject-level split run |

---

## Scope

This repository documents an experimental assignment, not a clinically validated EEG system. The results are limited by dataset size, participant coverage, experimental design and the evaluation caveat described above. Further validation on larger, independent cohorts and a strictly held-out test set would be required before making claims about clinical or real-world generalisation.
