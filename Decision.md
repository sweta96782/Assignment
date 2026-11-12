# DECISIONS.md

This document records the technical decisions made while correcting the defects identified in `DEFECTS.md`. It focuses on the rationale behind each correction, the alternatives considered where relevant, and why the selected approach is appropriate for the sleep-staging pipeline.

## Decision 1: Peak-to-peak artifact rejection threshold

**Decision:** Retain the intended 500 µV artifact-rejection threshold and convert it into volts (`500e-6`) to match the units used by MNE.

**Justification:**

MNE represents EEG signals in volts, whereas the original threshold was specified numerically as 500, implicitly representing microvolts. Directly comparing these values resulted in an excessively high threshold of 500 V, effectively disabling artifact rejection.

The dataset inspection showed that the maximum peak-to-peak amplitude was approximately 0.00096884 V (968.84 µV). Consequently, none of the epochs exceeded the original threshold, whereas 1,455 of 2,650 epochs exceeded the intended 500 µV threshold.

The 500 µV threshold was therefore retained rather than changing the artifact-rejection criterion. Expressing it as `500e-6` ensures unit consistency and restores the intended artifact-rejection behaviour.

## Decision 2: Treatment of unscored and movement-time annotations

**Decision:** Use only the five recognised sleep stages (Wake, N1, N2, N3 and REM) as supervised classification targets. Assign invalid or unscored annotations a label of `-1` and exclude corresponding target epochs from training.

**Justification:**

The original implementation assigned unknown annotations to class `0`, which represents Wake. This incorrectly converted unscored and movement-time annotations into valid Wake examples.

Dataset inspection identified 2,177.5 minutes of unscored sleep annotations and 17.5 minutes of movement-time annotations. Treating these intervals as Wake would introduce substantial label noise and distort the class distribution.

Assigning `-1` provides an explicit distinction between valid sleep-stage labels and invalid targets. Excluding these targets prevents the model from learning incorrect associations between unscored periods and Wake.

This approach preserves the intended five-class classification problem without introducing an artificial sleep stage.

## Decision 3: Centre-epoch prediction target

**Decision:** Use the centre epoch of each input sequence as the classification target by indexing `i + SEQ_LEN // 2`.

**Justification:**

The model architecture selects the centre temporal representation using `z[:, L // 2]`. Therefore, the corresponding training label must also represent the centre epoch.

Using the final epoch's label creates a mismatch between the temporal representation used for classification and its ground-truth target. This can result in incorrect supervision, particularly when sleep stages change within a sequence.

The centre-epoch target was selected because it is consistent with the model's classification head and the documented prediction objective.

The same indexing convention must be applied consistently in the dataset implementation, invalid-target filtering and training-label count calculation to avoid inconsistencies between training and evaluation.

## Decision 4: Transformer batch dimension handling

**Decision:** Set `batch_first=True` in `nn.TransformerEncoderLayer`.

**Justification:**

The epoch encoder produces tensors with the shape `(B, L, d)`, representing batch size, sequence length and embedding dimension, respectively.

However, PyTorch's default Transformer configuration expects sequence-first input `(L, B, d)`. Using the default configuration without changing the input interpretation causes the Transformer to attend across batch elements rather than across temporal epochs.

This violates the intended temporal modelling objective and can make predictions dependent on unrelated samples within the same batch.

Setting `batch_first=True` aligns the Transformer input convention with the tensor produced by the epoch encoder. The decision was further supported by the batch-versus-individual prediction experiment, which showed a maximum prediction difference of 0.0474985 before correction.

After correction, predictions should remain approximately identical whether a sample is evaluated individually or as part of a batch, when the model is in evaluation mode.

## Decision 5: Positional encoding before self-attention

**Decision:** Apply positional encoding to epoch embeddings before passing them to the Transformer encoder.

**Justification:**

Sleep staging is a temporal classification problem in which the order of successive epochs carries important information. Sleep stages exhibit temporal dependencies and characteristic transitions that cannot be adequately represented by treating all epoch embeddings as position-independent inputs.

The original implementation added positional information after the Transformer encoder. Consequently, the self-attention mechanism received embeddings without information about their temporal positions.

Moving positional encoding before the Transformer ensures that temporal position information is available during self-attention, allowing the model to learn relationships between epochs in their correct sequence.

The sequence-reversal experiment provided supporting evidence: after correcting the positional encoding order, reversing the input sequence produced a maximum prediction difference of 0.00325775 in evaluation mode. This indicates that sequence position influences the model output.

## Decision 6: EEG signal standardisation strategy

**Decision:** Replace independent per-epoch standardisation with per-recording, per-channel standardisation across epochs and time samples.

**Justification:**

Independent standardisation of every EEG epoch forces each epoch towards zero mean and unit standard deviation. Although this reduces scale differences, it can also suppress relative amplitude variations that may contain useful physiological information for sleep-stage discrimination.

In particular, slow-wave activity is an important characteristic of N3 sleep. Removing amplitude differences independently across epochs may reduce the availability of this information to the model.

The inspected recording showed that after per-recording, per-channel standardisation, the standard deviations differed across sleep stages: approximately 1.036 for Wake, 0.632 for N1, 0.731 for N2, 1.335 for N3 and 0.667 for REM.

This demonstrated that the alternative approach retained stage-associated amplitude variation instead of forcing every epoch to have approximately unit standard deviation.

Per-recording, per-channel standardisation was therefore selected to preserve relative amplitude information within each recording and channel while still providing a consistent scaling strategy.

## Decision 7: Subject-level train/test separation

**Decision:** Replace recording-level random splitting with subject-level train/test splitting, ensuring that all recordings belonging to a participant are assigned to the same partition.

**Justification:**

Sleep recordings from the same participant can share subject-specific characteristics, including EEG amplitude patterns, physiological properties and recording-related characteristics.

The original recording-level split allowed recordings from the same participant to appear in both training and testing sets. Inspection of the reproduced split confirmed that all five test subjects (05, 07, 09, 10 and 11) also appeared in the training set.

This creates subject leakage because the model can learn participant-specific characteristics during training and subsequently encounter those characteristics during testing.

Subject-level splitting was selected to ensure that the model is evaluated on participants whose recordings were not available during training. All recordings from a selected test participant are therefore kept together in the test partition, while recordings from the remaining participants are assigned to training.

Verification confirmed zero subject overlap between the training and testing partitions.

This provides a more appropriate evaluation of generalisation to previously unseen participants than recording-level splitting.
