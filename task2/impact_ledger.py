#!/usr/bin/env python3
"""
Task 2b — Impact ledger for the sleep-staging pipeline.

Run from the task2 folder:
    python impact_ledger.py

The script reads the fixed sleep_pipeline.py without editing it, creates a
temporary copy for each one-defect experiment, runs matched seeds, and prints
the paired metric differences and ranked impact-ledger table.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

SEEDS_BY_DEFECT = {
    "D1": [42, 0, 21],
    "D2": [42],
    "D3": [42],
    "D4": [42, 0, 21],
    "D5": [42],
    "D6": [42],
    "D7": [42, 0, 21],
}
BASELINE_SEEDS = sorted({seed for seeds in SEEDS_BY_DEFECT.values() for seed in seeds})

DEFECT_INFO = {
    "D1": (
        "Incorrect peak-to-peak artifact threshold",
        "May raise or lower scores; expected to reduce reliability",
        "500 is used instead of 500e-6. Since EEG is represented in volts, this "
        "threshold is far too permissive and electrode-pop epochs are retained.",
    ),
    "D2": (
        "Unscored annotations mapped to Wake",
        "May inflate Wake performance and distort other classes",
        "Unmapped/unscored epochs are assigned label 0 (Wake) rather than -1, "
        "so they become valid Wake training examples.",
    ),
    "D3": (
        "Last-epoch target used instead of centre-epoch target",
        "Expected to reduce performance",
        "Each sequence is paired with its final epoch label rather than its centre "
        "label, misaligning the input context and prediction target.",
    ),
    "D4": (
        "Transformer batch_first setting reversed",
        "Expected to reduce performance",
        "Input is (batch, sequence, features), but batch_first=False makes the "
        "Transformer interpret the batch dimension as sequence.",
    ),
    "D5": (
        "Positional encoding applied after the Transformer",
        "Expected to reduce temporal modelling effectiveness",
        "Self-attention receives epoch embeddings without positional information; "
        "adding positional encoding afterward cannot inform the attention operation.",
    ),
    "D6": (
        "Per-epoch rather than channel-wise standardisation",
        "May change performance; may remove useful scale information",
        "Normalisation over axes (1, 2) standardises each epoch across channels "
        "and time, instead of using the fixed pipeline's axes (0, 2).",
    ),
    "D7": (
        "Recording-level train/test split (subject leakage)",
        "Expected to make reported performance overly optimistic",
        "Recordings are split independently, allowing the same participant to "
        "contribute recordings to training and testing.",
    ),
}


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(
            f"{label}: expected the source block exactly once, found {count}. "
            "The fixed pipeline may differ from the version used to prepare this script."
        )
    return source.replace(old, new, 1)


def make_defective_source(fixed_source: str, defect_id: str, seed: int) -> str:
    """Create a temporary source copy with exactly one defect and the requested seed."""
    source = fixed_source

    if defect_id == "D1":
        source = replace_once(
            source, "REJECT_PTP = 500e-6", "REJECT_PTP = 500", "D1 threshold"
        )

    elif defect_id == "D2":
        source = replace_once(
            source,
            "y = np.full(n_ep, -1, dtype=np.int64)",
            "y = np.zeros(n_ep, dtype=np.int64)",
            "D2 label initialisation",
        )
        source = replace_once(
            source,
            "y[start:stop] = STAGE_MAP.get(desc, -1)",
            "y[start:stop] = STAGE_MAP.get(desc, 0)",
            "D2 annotation mapping",
        )

    elif defect_id == "D3":
        old = "i + SEQ_LEN // 2"
        new = "i + SEQ_LEN - 1"
        count = source.count(old)
        if count != 3:
            raise ValueError(f"D3: expected 3 centre-label references, found {count}.")
        source = source.replace(old, new)

    elif defect_id == "D4":
        source = replace_once(
            source, "batch_first=True", "batch_first=False", "D4 Transformer setting"
        )

    elif defect_id == "D5":
        old = "        z = self.pos(z)\n        z = self.transformer(z)"
        new = "        z = self.transformer(z)\n        z = self.pos(z)"
        source = replace_once(source, old, new, "D5 positional-encoding order")

    elif defect_id == "D6":
        old = (
            "X = (X - X.mean(axis=(0, 2), keepdims=True)) / "
            "(X.std(axis=(0, 2), keepdims=True) + 1e-8)"
        )
        new = (
            "X = (X - X.mean(axis=(1, 2), keepdims=True)) / "
            "(X.std(axis=(1, 2), keepdims=True) + 1e-8)"
        )
        source = replace_once(source, old, new, "D6 normalisation")

    elif defect_id == "D7":
        old_split = "\n".join([
            "    # Correct subject-independent train/test split",
            "    subjects = list(SUBJECTS)",
            "    random.Random(SEED).shuffle(subjects)",
            "",
            "    n_test = max(1, int(0.2 * len(subjects)))",
            "    test_subjects = subjects[:n_test]",
            "    train_subjects = subjects[n_test:]",
            "",
            "    train_files = get_files(train_subjects)",
            "    test_files = get_files(test_subjects)",
            "",
            '    print("Training subjects:", sorted(train_subjects))',
            '    print("Testing subjects:", sorted(test_subjects))',
            '    print("Training recordings:", len(train_files))',
            '    print("Testing recordings:", len(test_files))',
            '    print("Subject overlap:", set(train_subjects) & set(test_subjects))',
        ]) + "\n"

        new_split = "\n".join([
            "    # D7: deliberately reproduce the recording-level split.",
            "    files = get_files(SUBJECTS)",
            "    random.Random(SEED).shuffle(files)",
            "    n_test = max(1, int(0.2 * len(files)))",
            "    test_files, train_files = files[:n_test], files[n_test:]",
            "",
            "    # Recover subject IDs from Sleep-EDF filenames to diagnose leakage.",
            "    def _subject_ids(recording_files):",
            "        ids = set()",
            "        for psg_file, _ in recording_files:",
            '            match = re.search(r"SC4(\\d{2})", str(psg_file))',
            "            if match:",
            "                ids.add(int(match.group(1)))",
            "        return ids",
            "",
            "    train_subjects = _subject_ids(train_files)",
            "    test_subjects = _subject_ids(test_files)",
            '    print("Training subjects:", sorted(train_subjects))',
            '    print("Testing subjects:", sorted(test_subjects))',
            '    print("Training recordings:", len(train_files))',
            '    print("Testing recordings:", len(test_files))',
            '    print("Subject overlap:", train_subjects & test_subjects)',
        ]) + "\n"

        source = replace_once(source, old_split, new_split, "D7 split")
        source = replace_once(
            source, "import random", "import random\nimport re", "D7 diagnostic import"
        )

    else:
        raise ValueError(f"Unknown defect ID: {defect_id}")

    source, count = re.subn(
        r"(?m)^SEED\s*=\s*42\s*$", f"SEED = {int(seed)}", source
    )
    if count != 1:
        raise ValueError(f"Expected one SEED = 42 declaration; found {count}.")
    return source


def make_baseline_source(fixed_source: str, seed: int) -> str:
    source, count = re.subn(
        r"(?m)^SEED\s*=\s*42\s*$", f"SEED = {int(seed)}", fixed_source
    )
    if count != 1:
        raise ValueError(f"Expected one SEED = 42 declaration; found {count}.")
    return source


def parse_metrics(output: str) -> dict:
    patterns = {
        "accuracy": r"(?m)^accuracy\s+([0-9]*\.?[0-9]+)\s*$",
        "macro_f1": r"(?m)^macro-F1\s+([0-9]*\.?[0-9]+)\s*$",
        "kappa": r"(?m)^kappa\s+([0-9]*\.?[0-9]+)\s*$",
    }
    result = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        if not matches:
            raise ValueError(f"Could not parse {key} from pipeline output.")
        result[key] = float(matches[-1])
    overlap = re.findall(r"(?m)^Subject overlap:\s*(.*)$", output)
    result["subject_overlap"] = overlap[-1].strip() if overlap else "not reported"
    return result


def run_one(
    source: str,
    label: str,
    seed: int,
    workdir: Path,
    logs_dir: Path,
) -> dict:
    script_path = workdir / f"_impact_{label.lower()}_seed{seed}.py"
    log_path = logs_dir / f"{label.lower()}_seed{seed}.log"
    script_path.write_text(source, encoding="utf-8")

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(workdir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-45:])
            raise RuntimeError(
                f"{label}, seed {seed} failed (exit {completed.returncode}).\n"
                f"Last output lines:\n{tail}\nFull log: {log_path}"
            )

        metrics = parse_metrics(completed.stdout)
        metrics.update({"defect": label, "seed": seed})
        print(
            f"Completed {label:8s} seed={seed:2d} | "
            f"Acc={metrics['accuracy']:.3f} "
            f"Macro-F1={metrics['macro_f1']:.3f} "
            f"Kappa={metrics['kappa']:.3f} "
            f"subject overlap={metrics['subject_overlap']}"
        )
        return metrics
    finally:
        script_path.unlink(missing_ok=True)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2b impact-ledger experiments.")
    parser.add_argument(
        "--base",
        default="sleep_pipeline.py",
        help="Fully fixed pipeline path (default: sleep_pipeline.py).",
    )
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {base_path}. Put impact_ledger.py beside sleep_pipeline.py "
            "or pass --base /path/to/sleep_pipeline.py."
        )

    fixed_source = base_path.read_text(encoding="utf-8")
    workdir = base_path.parent
    logs_dir = workdir / "impact_ledger_logs"
    logs_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print("TASK 2b — IMPACT LEDGER")
    print(f"Fixed pipeline: {base_path}")
    print(f"Baseline seeds: {BASELINE_SEEDS}")
    print("Three-seed defects: D1, D4 and D7")
    print("The original fixed pipeline will not be modified.")
    print("=" * 78)

    # Run one fixed baseline for every seed used by any defect.
    baseline = {}
    for seed in BASELINE_SEEDS:
        baseline[seed] = run_one(
            make_baseline_source(fixed_source, seed),
            "BASELINE", seed, workdir, logs_dir
        )

    # Run each defect alone. D1, D4 and D7 receive three seeds.
    defect_runs = {}
    for defect_id, seeds in SEEDS_BY_DEFECT.items():
        defect_runs[defect_id] = {}
        for seed in seeds:
            variant = make_defective_source(fixed_source, defect_id, seed)
            defect_runs[defect_id][seed] = run_one(
                variant, defect_id, seed, workdir, logs_dir
            )

    # Compute paired differences: defective run minus baseline at the same seed.
    paired = []
    summaries = []
    for defect_id, seed_results in defect_runs.items():
        metric_deltas = {"accuracy": [], "macro_f1": [], "kappa": []}
        for seed, result in seed_results.items():
            base = baseline[seed]
            delta = {
                metric: result[metric] - base[metric]
                for metric in metric_deltas
            }
            for metric in metric_deltas:
                metric_deltas[metric].append(delta[metric])
            paired.append((defect_id, seed, delta, result["subject_overlap"]))

        average_delta = {
            metric: mean(values) for metric, values in metric_deltas.items()
        }
        # Put all metrics on percentage-point scales before combining magnitudes.
        impact_index = mean([
            abs(average_delta["accuracy"]) * 100,
            abs(average_delta["macro_f1"]) * 100,
            abs(average_delta["kappa"]) * 100,
        ])
        summaries.append({
            "defect": defect_id,
            "delta": average_delta,
            "impact_index": impact_index,
        })

    summaries.sort(key=lambda item: item["impact_index"], reverse=True)

    print("\nPAIRED SEED-LEVEL DIFFERENCES")
    print("Δ = defective result − fixed result at the same seed.")
    detail_rows = []
    for defect_id, seed, delta, overlap in paired:
        detail_rows.append([
            defect_id,
            str(seed),
            f"{delta['accuracy'] * 100:+.2f}",
            f"{delta['macro_f1']:+.3f}",
            f"{delta['kappa']:+.3f}",
            overlap,
        ])
    print_table(
        ["Defect", "Seed", "Δ Accuracy (pp)", "Δ Macro-F1", "Δ Kappa", "Subject overlap"],
        detail_rows,
    )

    print("\nFINAL IMPACT LEDGER — RANKED BY MEAN ABSOLUTE METRIC CHANGE")
    print(
        "Impact index = mean absolute Δ across the three metrics after converting "
        "each to percentage-point units."
    )
    final_rows = []
    for rank, item in enumerate(summaries, start=1):
        defect_id = item["defect"]
        info = DEFECT_INFO[defect_id]
        delta = item["delta"]
        final_rows.append([
            str(rank),
            defect_id,
            f"{delta['accuracy'] * 100:+.2f}",
            f"{delta['macro_f1']:+.3f}",
            f"{delta['kappa']:+.3f}",
            f"{item['impact_index']:.2f}",
            info[1],
            info[2],
            "Not tested by one-defect-at-a-time runs",
        ])
    print_table(
        [
            "Rank", "Defect", "Δ Accuracy (pp)", "Δ Macro-F1", "Δ Kappa",
            "Impact index", "Predicted direction", "Mechanism", "Interaction",
        ],
        final_rows,
    )

    print("\nNOTES FOR INTERPRETATION")
    print(
        "- Positive Δ means the defective pipeline scored higher; it does not mean "
        "the defect is beneficial."
    )
    print(
        "- D7 changes the split unit. Its test set may include participants also "
        "present in training. Its score is therefore leakage-contaminated and is "
        "not a valid subject-independent result."
    )
    print(
        "- The isolated-defect experiments cannot establish whether an effect "
        "vanishes or flips after another defect is fixed. That requires additional "
        "paired runs with selected combinations of two defects. Do not claim an "
        "interaction unless those combination runs are performed."
    )
    print(f"- Full run logs: {logs_dir}")
    print("- The original fixed pipeline was not modified.")


if __name__ == "__main__":
    main()
