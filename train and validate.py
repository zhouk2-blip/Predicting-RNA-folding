import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
from DATASET import RNADataset as RNA
from model_conv_attn import RNAmodel
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from kabsch import kabsch_align_batch
from argparse import ArgumentParser
from training_config import (
    add_shared_training_args,
    build_config,
    cli_overrides_from_args,
    config_with_runtime_paths,
    print_run_summary,
    resolve_output_dir,
    write_used_config,
)
EPOCHS = 40
BATCH_SIZE = 32
IR = 3e-4
MAX_LEN = 256
PATIENCE = 7
DATA_DIR = "dataset_final"
OUTPUT_ROOT = "output"
MSA_DIR = os.path.join(DATA_DIR, "MSA")
VAL_LABELS_PATH = "validation_labels_new.normalized.csv"

RUN_DIR_RE = re.compile(r"^output(\d+)$")
STRUCTURE_FEATURE_SCALE_FLOORS = np.array(
    [0.10, 0.05, 0.05, 0.03, 0.03, 0.05, 0.05, 0.005, 0.02, 0.05, 0.05, 0.01],
    dtype=np.float32,
)

def threshold_tag(threshold):
    """Purpose: Convert an accuracy threshold into a filename-safe tag.

    Input:
        threshold: Numeric distance threshold used for accuracy.
    Output:
        String tag such as "1p0" or "0p75".
    """
    return str(float(threshold)).replace(".", "p")

def select_accuracy_threshold(thresholds, preferred=1.0):
    """Purpose: Pick the threshold used for the validation accuracy plot.

    Input:
        thresholds: Accuracy thresholds requested for reporting.
        preferred: Threshold to use when it is present.
    Output:
        The preferred threshold, or the first available threshold.
    """
    for threshold in thresholds:
        if abs(threshold - preferred) < 1e-9:
            return threshold
    return thresholds[0]

def next_output_dir(output_root=OUTPUT_ROOT):
    """Purpose: Create the next numbered run directory path.

    Input:
        output_root: Parent directory that contains output1, output2, ...
    Output:
        Path for the next run, such as output/output3.
    """
    os.makedirs(output_root, exist_ok=True)
    max_idx = 0
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        match = RUN_DIR_RE.match(name)
        if match and os.path.isdir(path):
            max_idx = max(max_idx, int(match.group(1)))
    return os.path.join(output_root, f"output{max_idx + 1}")

def resolve_label_path(data_dir, val_labels_path):
    """Purpose: Pick the validation label file requested by config or defaults.

    Input:
        data_dir: Dataset directory.
        val_labels_path: Preferred validation label path.
    Output:
        Existing label CSV path.
    """
    if val_labels_path and os.path.exists(val_labels_path):
        return val_labels_path
    if val_labels_path:
        data_relative_path = os.path.join(data_dir, val_labels_path)
        if os.path.exists(data_relative_path):
            return data_relative_path
    return os.path.join(data_dir, "validation_labels.csv")


def load_data(data_dir=DATA_DIR, val_labels_path=VAL_LABELS_PATH):
    """Purpose: Load sequence, label, and pair-feature CSV files.

    Input:
        data_dir: Directory containing training and validation CSV files.
        val_labels_path: Preferred validation label CSV path.
    Output:
        DataFrames for train/validation sequences, labels, and pair features.
    """
    print("Loading data...")
    train_seqs = pd.read_csv(os.path.join(data_dir, "train_sequences.csv"))
    val_seqs = pd.read_csv(os.path.join(data_dir, "validation_sequences.csv"))
    train_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    val_label_path = resolve_label_path(data_dir, val_labels_path)
    val_labels = pd.read_csv(val_label_path)
    train_pair_df = pd.read_csv(os.path.join(data_dir, "train_pair_features.csv"))
    val_pair_df = pd.read_csv(os.path.join(data_dir, "validation_pair_features.csv"))
    return train_seqs, val_seqs, train_labels, val_labels, train_pair_df, val_pair_df

def label_target_ids(label_df):
    """Purpose: Extract target IDs from residue-level label IDs.

    Input:
        label_df: DataFrame with an ID column like target_id_resid.
    Output:
        Series containing target_id for each label row.
    """
    return label_df["ID"].astype(str).str.rsplit("_", n=1).str[0]

def filter_training_data(
    train_seqs,
    train_labels,
    train_pair_df=None,
    max_len=MAX_LEN,
    min_valid_labels=None,
    min_label_coverage=None,
):
    """Purpose: Remove train targets that have too little usable coordinate supervision.

    Input:
        train_seqs: Training sequence DataFrame.
        train_labels: Training coordinate-label DataFrame.
        train_pair_df: Optional pair-feature DataFrame to filter with sequences.
        max_len: Maximum residues seen by the model.
        min_valid_labels: Minimum valid coordinate rows required per target.
        min_label_coverage: Minimum valid-label fraction over capped sequence length.
    Output:
        Tuple of filtered sequence, label, pair-feature DataFrames, and a report dict.
    """
    min_valid = 0 if min_valid_labels is None else int(min_valid_labels)
    min_coverage = 0.0 if min_label_coverage is None else float(min_label_coverage)
    enabled = min_valid > 0 or min_coverage > 0.0

    target_ids = train_seqs["target_id"].astype(str)
    capped_lengths = train_seqs["sequence"].astype(str).str.len().clip(upper=max_len)
    target_stats = pd.DataFrame({
        "target_id": target_ids,
        "capped_length": capped_lengths,
    })

    label_targets = label_target_ids(train_labels)
    coord_cols = ["x_1", "y_1", "z_1"]
    valid_coords = train_labels[coord_cols].notna().all(axis=1) & (train_labels[coord_cols] > -1e17).all(axis=1)
    length_by_target = target_stats.set_index("target_id")["capped_length"]
    label_lengths = label_targets.map(length_by_target).fillna(0)
    within_cap = (train_labels["resid"] >= 1) & (train_labels["resid"] <= label_lengths)
    valid_counts = (
        pd.DataFrame({"target_id": label_targets, "valid": valid_coords & within_cap})
        .groupby("target_id")["valid"]
        .sum()
    )

    target_stats["valid_labels"] = target_stats["target_id"].map(valid_counts).fillna(0).astype(int)
    target_stats["label_coverage"] = target_stats["valid_labels"] / target_stats["capped_length"].clip(lower=1)

    if enabled:
        keep_mask = (
            (target_stats["valid_labels"] >= min_valid)
            & (target_stats["label_coverage"] >= min_coverage)
        )
    else:
        keep_mask = pd.Series(True, index=target_stats.index)

    kept_ids = set(target_stats.loc[keep_mask, "target_id"])
    filtered_seqs = train_seqs[target_ids.isin(kept_ids)].copy()
    filtered_labels = train_labels[label_targets.isin(kept_ids)].copy()
    if train_pair_df is None:
        filtered_pairs = None
    else:
        filtered_pairs = train_pair_df[train_pair_df["target_id"].astype(str).isin(kept_ids)].copy()

    report = {
        "enabled": enabled,
        "min_valid_labels": min_valid_labels,
        "min_label_coverage": min_label_coverage,
        "original_targets": int(len(train_seqs)),
        "kept_targets": int(len(filtered_seqs)),
        "removed_targets": int(len(train_seqs) - len(filtered_seqs)),
        "original_valid_labels": int(target_stats["valid_labels"].sum()),
        "kept_valid_labels": int(target_stats.loc[keep_mask, "valid_labels"].sum()),
        "removed_target_sample": target_stats.loc[~keep_mask, "target_id"].head(10).tolist(),
    }
    return filtered_seqs, filtered_labels, filtered_pairs, report

def print_training_filter_report(report):
    """Purpose: Print a concise report for runtime train-label filtering.

    Input:
        report: Dictionary returned by filter_training_data().
    Output:
        None. Writes the filtering summary to stdout.
    """
    if not report["enabled"]:
        print("Train label filtering: disabled.")
        return
    print(
        "Train label filtering: "
        f"kept {report['kept_targets']}/{report['original_targets']} targets, "
        f"removed {report['removed_targets']}, "
        f"valid labels {report['kept_valid_labels']}/{report['original_valid_labels']}."
    )
    if report["removed_target_sample"]:
        print("Removed target sample:", ", ".join(report["removed_target_sample"]))

def normalize_path_list(paths):
    """Purpose: Normalize optional YAML/CLI path values into a path list.

    Input:
        paths: None, a single string path, or a sequence of paths.
    Output:
        List of non-empty string paths.
    """
    if paths is None:
        return []
    if isinstance(paths, str):
        return [paths] if paths.strip() else []
    return [str(path) for path in paths if str(path).strip()]


def normalized_sequences(seq_df):
    """Purpose: Normalize RNA sequence strings for exact-overlap checks.

    Input:
        seq_df: DataFrame with a sequence column.
    Output:
        Series of upper-case sequence strings.
    """
    return seq_df["sequence"].astype(str).str.upper()


def subset_training_rows(seq_df, label_df, pair_df, keep_ids):
    """Purpose: Keep sequence, label, and pair-feature rows for selected target IDs.

    Input:
        seq_df: Target-level sequence DataFrame.
        label_df: Residue-level label DataFrame.
        pair_df: Optional residue-level pair-feature DataFrame.
        keep_ids: Iterable of target IDs to retain.
    Output:
        Tuple of filtered sequence, label, and pair-feature DataFrames.
    """
    keep_set = {str(target_id) for target_id in keep_ids}
    seq_ids = seq_df["target_id"].astype(str)
    label_ids = label_target_ids(label_df)

    filtered_seq = seq_df[seq_ids.isin(keep_set)].copy()
    filtered_labels = label_df[label_ids.isin(keep_set)].copy()
    if pair_df is None:
        filtered_pairs = None
    else:
        pair_ids = pair_df["target_id"].astype(str)
        filtered_pairs = pair_df[pair_ids.isin(keep_set)].copy()

    return filtered_seq, filtered_labels, filtered_pairs


def limit_training_data(
    train_seqs,
    train_labels,
    train_pair_df=None,
    max_train_targets=None,
    max_len=MAX_LEN,
    selection="quality_diverse",
    seed=13,
    length_bin_size=50,
    val_seq_df=None,
    val_pair_df=None,
    contact_map_dir=None,
    structure_anchor_groups=None,
    targets_per_group=150,
    similarity_top_k_per_group=300,
):
    """Purpose: Cap the main training set and keep related label/pair rows aligned.

    Input:
        train_seqs: Training sequence DataFrame.
        train_labels: Training coordinate-label DataFrame.
        train_pair_df: Optional pair-feature DataFrame.
        max_train_targets: Maximum train targets to keep; None or <=0 keeps all targets.
        max_len: Model cap used by the diversity selector.
        selection: Selection strategy, either quality_diverse or first.
        seed: Deterministic selection seed for quality_diverse.
        length_bin_size: Width of capped-length bins for diversity.
        val_seq_df: Validation sequences used by validation_structure_random.
        val_pair_df: Validation pair features used by validation_structure_random.
        contact_map_dir: SPOT map directory used by validation_structure_random.
        structure_anchor_groups: Optional validation anchor-group mapping.
        targets_per_group: Per-validation-group sample cap for validation_structure_random.
        similarity_top_k_per_group: Top-K similar pool per group before random sampling.
    Output:
        Tuple of limited sequence, label, pair-feature DataFrames, and a report dict.
    """
    strategy = (selection or "quality_diverse").lower()
    if max_train_targets is None or int(max_train_targets) <= 0:
        report = {
            "enabled": False,
            "max_train_targets": max_train_targets,
            "selection": strategy,
            "original_targets": int(len(train_seqs)),
            "kept_targets": int(len(train_seqs)),
            "removed_targets": 0,
            "selected_target_sample": train_seqs["target_id"].astype(str).head(10).tolist(),
        }
        return train_seqs, train_labels, train_pair_df, report

    if strategy == "validation_structure_random":
        selected_ids, selection_report = select_validation_structure_random_target_ids(
            train_seqs,
            train_pair_df,
            val_seq_df,
            val_pair_df,
            contact_map_dir,
            max_targets=max_train_targets,
            targets_per_group=targets_per_group,
            seed=seed,
            max_len=max_len,
            anchor_groups=structure_anchor_groups,
            similarity_top_k_per_group=similarity_top_k_per_group,
        )
    else:
        selected_ids = select_extra_target_ids(
            train_seqs,
            max_extra_targets=max_train_targets,
            max_len=max_len,
            selection=strategy,
            seed=seed,
            length_bin_size=length_bin_size,
        )
        selection_report = {}

    limited_seqs, limited_labels, limited_pairs = subset_training_rows(
        train_seqs,
        train_labels,
        train_pair_df,
        selected_ids,
    )
    report = {
        "enabled": True,
        "max_train_targets": int(max_train_targets),
        "selection": strategy,
        "original_targets": int(len(train_seqs)),
        "kept_targets": int(len(limited_seqs)),
        "removed_targets": int(len(train_seqs) - len(limited_seqs)),
        "selected_target_sample": limited_seqs["target_id"].astype(str).head(10).tolist(),
    }
    report.update(selection_report)
    report["original_targets"] = int(len(train_seqs))
    report["kept_targets"] = int(len(limited_seqs))
    report["removed_targets"] = int(len(train_seqs) - len(limited_seqs))
    return limited_seqs, limited_labels, limited_pairs, report


def print_train_limit_report(report):
    """Purpose: Print a concise report for max-train-target limiting.

    Input:
        report: Dictionary returned by limit_training_data().
    Output:
        None. Writes the limiting summary to stdout.
    """
    if not report["enabled"]:
        print("Train target limiting: disabled.")
        return
    print(
        "Train target limiting: "
        f"kept {report['kept_targets']}/{report['original_targets']} targets "
        f"using {report['selection']}."
    )


def select_extra_target_ids(
    seq_df,
    max_extra_targets,
    max_len=MAX_LEN,
    selection="quality_diverse",
    seed=13,
    length_bin_size=50,
):
    """Purpose: Select a small, train-side-diverse subset of extra targets.

    Input:
        seq_df: Quality-filtered extra sequence DataFrame.
        max_extra_targets: Maximum extra targets to keep; None keeps all candidates.
        max_len: Model cap used for length binning.
        selection: Selection strategy, either quality_diverse or first.
        seed: Random seed for deterministic within-bin shuffling.
        length_bin_size: Width of capped-length bins for diversity.
    Output:
        Ordered list of selected target IDs.
    """
    if seq_df.empty:
        return []
    if max_extra_targets is None:
        return seq_df["target_id"].astype(str).tolist()

    max_targets = int(max_extra_targets)
    if max_targets <= 0:
        return []
    if len(seq_df) <= max_targets:
        return seq_df["target_id"].astype(str).tolist()

    strategy = (selection or "quality_diverse").lower()
    if strategy in {"first", "head"}:
        return seq_df["target_id"].astype(str).head(max_targets).tolist()
    if strategy != "quality_diverse":
        raise ValueError(
            f'Unsupported extra_train_selection="{selection}". '
            'Use "quality_diverse" or "first".'
        )

    bin_size = max(1, int(length_bin_size or 50))
    work = seq_df[["target_id", "sequence"]].copy()
    work["_target_id"] = work["target_id"].astype(str)
    work["_capped_length"] = work["sequence"].astype(str).str.len().clip(upper=max_len)
    work["_length_bin"] = (work["_capped_length"] // bin_size).astype(int)

    grouped_ids = {}
    for bin_id, group in work.groupby("_length_bin", sort=True):
        if len(group) > 1:
            group = group.sample(frac=1.0, random_state=int(seed) + int(bin_id))
        grouped_ids[int(bin_id)] = group["_target_id"].tolist()

    selected = []
    while len(selected) < max_targets and any(grouped_ids.values()):
        for bin_id in sorted(grouped_ids):
            if not grouped_ids[bin_id] or len(selected) >= max_targets:
                continue
            selected.append(grouped_ids[bin_id].pop(0))
    return selected


def resolve_contact_map_path(target_id, contact_map_dir):
    """Purpose: Find the first available SPOT contact map for a target.

    Input:
        target_id: RNA target identifier.
        contact_map_dir: One directory or ordered fallback directories.
    Output:
        Contact-map path string, or None when no map is available.
    """
    for map_dir in normalize_path_list(contact_map_dir):
        map_path = os.path.join(map_dir, f"{target_id}.npy")
        if os.path.exists(map_path):
            return map_path
    return None


def run_lengths_from_mask(values):
    """Purpose: Summarize contiguous paired-residue runs.

    Input:
        values: Boolean-like array where true means the residue is paired.
    Output:
        List of contiguous true-run lengths.
    """
    runs = []
    current = 0
    for value in values:
        if bool(value):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def pair_feature_arrays(pair_df, target_id, length):
    """Purpose: Convert per-residue pair-feature rows into fixed-length arrays.

    Input:
        pair_df: Pair-feature DataFrame, or None.
        target_id: RNA target identifier.
        length: Number of residues to represent.
    Output:
        Tuple of paired-indicator and partner-index arrays shaped (length,).
    """
    paired = np.zeros(length, dtype=np.float32)
    partners = np.zeros(length, dtype=np.float32)
    if pair_df is None or pair_df.empty or "target_id" not in pair_df.columns:
        return paired, partners

    rows = pair_df[pair_df["target_id"].astype(str) == str(target_id)]
    if rows.empty:
        return paired, partners

    for _, row in rows.iterrows():
        resid = int(row.get("resid", 0))
        if not 1 <= resid <= length:
            continue
        paired[resid - 1] = float(row.get("is_paired", 0) or 0)
        partners[resid - 1] = float(row.get("pair_partner", 0) or 0)
    return paired, partners


def contact_pair_structure_fingerprint(seq_df, pair_df, target_id, contact_map_dir, max_len=MAX_LEN):
    """Purpose: Build an input-side pair/contact structure fingerprint for one target.

    Input:
        seq_df: Sequence DataFrame with target_id and sequence columns.
        pair_df: Pair-feature DataFrame for the same target namespace.
        target_id: RNA target identifier.
        contact_map_dir: SPOT map directory or fallback directories.
        max_len: Maximum residues included in the fingerprint.
    Output:
        Float feature vector, or None when sequence or contact map is unavailable.
    """
    seq_rows = seq_df[seq_df["target_id"].astype(str) == str(target_id)]
    if seq_rows.empty:
        return None
    length = min(len(str(seq_rows.iloc[0]["sequence"])), int(max_len))
    if length <= 0:
        return None

    map_path = resolve_contact_map_path(target_id, contact_map_dir)
    if map_path is None:
        return None

    paired, partners = pair_feature_arrays(pair_df, target_id, length)
    runs = run_lengths_from_mask(paired > 0)
    residue_ids = np.arange(1, length + 1, dtype=np.float32)
    paired_partner_mask = partners > 0
    spans = np.abs(partners[paired_partner_mask] - residue_ids[paired_partner_mask]) / max(length, 1)

    loaded = np.load(map_path).astype(np.float32)
    contact = np.zeros((length, length), dtype=np.float32)
    h = min(length, loaded.shape[0])
    w = min(length, loaded.shape[1])
    contact[:h, :w] = loaded[:h, :w]

    if length > 1:
        upper = np.triu_indices(length, k=1)
        values = contact[upper]
        row_idx, col_idx = upper
        separations = np.abs(row_idx - col_idx).astype(np.float32) / max(length, 1)
        high_contacts = values > 0.5
        medium_contacts = values > 0.2
        high_sep_mean = float(separations[high_contacts].mean()) if high_contacts.any() else 0.0
        medium_sep_mean = float(separations[medium_contacts].mean()) if medium_contacts.any() else 0.0
        high_density = float(high_contacts.mean())
        medium_density = float(medium_contacts.mean())
        contact_mean = float(values.mean())
    else:
        high_sep_mean = medium_sep_mean = high_density = medium_density = contact_mean = 0.0

    max_run = max(runs) if runs else 0
    mean_run = float(np.mean(runs)) if runs else 0.0
    return np.array(
        [
            length / max(float(max_len), 1.0),
            float(paired.mean()) if length else 0.0,
            max_run / max(length, 1),
            mean_run / max(length, 1),
            len(runs) / max(length, 1),
            float(spans.mean()) if len(spans) else 0.0,
            float(spans.max()) if len(spans) else 0.0,
            high_density,
            medium_density,
            high_sep_mean,
            medium_sep_mean,
            contact_mean,
        ],
        dtype=np.float32,
    )


def normalize_structure_reinforcement_anchor_groups(anchor_groups):
    """Purpose: Normalize structure-reinforcement anchor groups from config.

    Input:
        anchor_groups: Mapping of group name to target ID list.
    Output:
        Ordered dictionary-like mapping with string group names and non-empty ID lists.
    """
    if not anchor_groups:
        return {}
    if not isinstance(anchor_groups, dict):
        raise ValueError("structure_reinforcement_anchor_groups must be a YAML mapping.")
    normalized = {}
    for group_name, target_ids in anchor_groups.items():
        if isinstance(target_ids, str):
            ids = [target_ids]
        else:
            ids = [str(target_id) for target_id in (target_ids or []) if str(target_id).strip()]
        if ids:
            normalized[str(group_name)] = ids
    return normalized


def structure_reinforcement_score_table(
    candidate_seq_df,
    candidate_pair_df,
    anchor_seq_df,
    anchor_pair_df,
    anchor_groups,
    contact_map_dir,
    max_len=MAX_LEN,
):
    """Purpose: Score extra targets by pair/contact similarity to structure anchors.

    Input:
        candidate_seq_df: Quality-filtered candidate sequence DataFrame.
        candidate_pair_df: Candidate pair-feature DataFrame.
        anchor_seq_df: Sequence DataFrame containing anchor targets.
        anchor_pair_df: Pair-feature DataFrame containing anchor targets.
        anchor_groups: Mapping of group name to anchor target IDs.
        contact_map_dir: SPOT map directory or fallback directories.
        max_len: Maximum residues included in fingerprints.
    Output:
        DataFrame with target_id, nearest group, and similarity score sorted ascending.
    """
    groups = normalize_structure_reinforcement_anchor_groups(anchor_groups)
    if candidate_seq_df.empty or not groups or not normalize_path_list(contact_map_dir):
        return pd.DataFrame(columns=["target_id", "group", "score"])

    group_vectors = {}
    all_anchor_vectors = []
    for group_name, anchor_ids in groups.items():
        vectors = []
        for anchor_id in anchor_ids:
            fingerprint = contact_pair_structure_fingerprint(
                anchor_seq_df,
                anchor_pair_df,
                anchor_id,
                contact_map_dir,
                max_len=max_len,
            )
            if fingerprint is not None:
                vectors.append(fingerprint)
                all_anchor_vectors.append(fingerprint)
        if vectors:
            group_vectors[group_name] = np.stack(vectors)

    if not group_vectors:
        return pd.DataFrame(columns=["target_id", "group", "score"])

    scale = np.std(np.stack(all_anchor_vectors), axis=0) + STRUCTURE_FEATURE_SCALE_FLOORS
    rows = []
    seen = set()
    for target_id in candidate_seq_df["target_id"].astype(str):
        if target_id in seen:
            continue
        seen.add(target_id)
        fingerprint = contact_pair_structure_fingerprint(
            candidate_seq_df,
            candidate_pair_df,
            target_id,
            contact_map_dir,
            max_len=max_len,
        )
        if fingerprint is None:
            continue

        best_group = None
        best_score = float("inf")
        for group_name, vectors in group_vectors.items():
            distances = np.linalg.norm((vectors - fingerprint) / scale, axis=1)
            group_score = float(distances.min())
            if group_score < best_score:
                best_score = group_score
                best_group = group_name
        rows.append({"target_id": target_id, "group": best_group, "score": best_score})

    if not rows:
        return pd.DataFrame(columns=["target_id", "group", "score"])
    return pd.DataFrame(rows).sort_values(["score", "target_id"]).reset_index(drop=True)


def select_structure_reinforcement_target_ids(
    candidate_seq_df,
    candidate_pair_df,
    anchor_seq_df,
    anchor_pair_df,
    anchor_groups,
    contact_map_dir,
    total_targets=0,
    targets_per_group=None,
    used_ids=None,
    max_len=MAX_LEN,
):
    """Purpose: Select extra targets that reinforce hard-target contact/pair structures.

    Input:
        candidate_seq_df: Quality-filtered extra sequence DataFrame.
        candidate_pair_df: Quality-filtered extra pair-feature DataFrame.
        anchor_seq_df: Validation/test sequence DataFrame containing anchors.
        anchor_pair_df: Pair-feature DataFrame containing anchors.
        anchor_groups: Mapping of group name to anchor target IDs.
        contact_map_dir: SPOT map directory or fallback directories.
        total_targets: Maximum structure-reinforcement targets to select.
        targets_per_group: Per-group quota before global fill.
        used_ids: Already selected target IDs to avoid duplicating.
        max_len: Maximum residues included in fingerprints.
    Output:
        Tuple of selected target ID list and report dictionary.
    """
    total = 0 if total_targets is None else int(total_targets)
    groups = normalize_structure_reinforcement_anchor_groups(anchor_groups)
    used = {str(target_id) for target_id in (used_ids or set())}
    empty_report = {
        "enabled": total > 0,
        "requested_targets": total,
        "selected_targets": 0,
        "scored_candidates": 0,
        "group_counts": {group_name: 0 for group_name in groups},
        "selected_target_sample": [],
    }
    if total <= 0 or not groups:
        return [], empty_report

    score_table = structure_reinforcement_score_table(
        candidate_seq_df,
        candidate_pair_df,
        anchor_seq_df,
        anchor_pair_df,
        groups,
        contact_map_dir,
        max_len=max_len,
    )
    if score_table.empty:
        empty_report["scored_candidates"] = 0
        return [], empty_report

    score_table = score_table[~score_table["target_id"].astype(str).isin(used)].copy()
    if score_table.empty:
        empty_report["scored_candidates"] = 0
        return [], empty_report

    if targets_per_group is None:
        per_group = int(np.ceil(total / max(len(groups), 1)))
    else:
        per_group = max(0, int(targets_per_group))

    selected = []
    selected_set = set()
    group_counts = {group_name: 0 for group_name in groups}
    for group_name in groups:
        group_rows = score_table[score_table["group"] == group_name]
        for target_id in group_rows["target_id"].astype(str):
            if len(selected) >= total or group_counts[group_name] >= per_group:
                break
            if target_id in selected_set:
                continue
            selected.append(target_id)
            selected_set.add(target_id)
            group_counts[group_name] += 1

    for _, row in score_table.iterrows():
        if len(selected) >= total:
            break
        target_id = str(row["target_id"])
        if target_id in selected_set:
            continue
        selected.append(target_id)
        selected_set.add(target_id)
        group_name = row["group"]
        group_counts[group_name] = group_counts.get(group_name, 0) + 1

    report = {
        "enabled": True,
        "requested_targets": total,
        "selected_targets": int(len(selected)),
        "scored_candidates": int(len(score_table)),
        "group_counts": group_counts,
        "selected_target_sample": selected[:10],
    }
    return selected, report


def default_validation_anchor_groups(val_seq_df, max_len=MAX_LEN):
    """Purpose: Build one structure-sampling anchor group per short validation target.

    Input:
        val_seq_df: Validation sequence DataFrame with target_id and sequence columns.
        max_len: Maximum original sequence length allowed for anchors.
    Output:
        Mapping of validation target ID to a single-ID anchor list.
    """
    if val_seq_df is None or val_seq_df.empty:
        return {}
    lengths = val_seq_df["sequence"].astype(str).str.len()
    short_rows = val_seq_df[lengths <= int(max_len)]
    return {
        str(target_id): [str(target_id)]
        for target_id in short_rows["target_id"].astype(str)
    }


def select_validation_structure_random_target_ids(
    candidate_seq_df,
    candidate_pair_df,
    val_seq_df,
    val_pair_df,
    contact_map_dir,
    max_targets=None,
    targets_per_group=150,
    seed=13,
    max_len=MAX_LEN,
    anchor_groups=None,
    similarity_top_k_per_group=300,
):
    """Purpose: Randomly sample train targets similar to short validation structures.

    Input:
        candidate_seq_df: Training sequence DataFrame after quality filtering.
        candidate_pair_df: Training pair-feature DataFrame.
        val_seq_df: Validation sequence DataFrame used as anchors.
        val_pair_df: Validation pair-feature DataFrame used as anchor structure input.
        contact_map_dir: SPOT contact-map directory or fallback directories.
        max_targets: Optional global cap after per-group sampling.
        targets_per_group: Maximum sampled candidates per validation structure group.
        seed: Deterministic random seed.
        max_len: Maximum original sequence length for candidates and anchors.
        anchor_groups: Optional explicit validation anchor groups.
        similarity_top_k_per_group: Top-K most similar candidates used as the random pool per group.
    Output:
        Tuple of selected target IDs and a report dictionary.
    """
    empty_report = {
        "enabled": True,
        "selection": "validation_structure_random",
        "candidate_targets": int(len(candidate_seq_df)),
        "length_filtered_targets": 0,
        "scored_targets": 0,
        "selected_targets": 0,
        "group_counts": {},
        "selected_target_sample": [],
    }
    if candidate_seq_df.empty:
        return [], empty_report

    groups = normalize_structure_reinforcement_anchor_groups(anchor_groups)
    if not groups:
        groups = default_validation_anchor_groups(val_seq_df, max_len=max_len)
    if not groups:
        return [], empty_report

    lengths = candidate_seq_df["sequence"].astype(str).str.len()
    short_candidates = candidate_seq_df[lengths <= int(max_len)].copy()
    empty_report["length_filtered_targets"] = int(len(short_candidates))
    if short_candidates.empty:
        return [], empty_report

    score_table = structure_reinforcement_score_table(
        short_candidates,
        candidate_pair_df,
        val_seq_df,
        val_pair_df,
        groups,
        contact_map_dir,
        max_len=max_len,
    )
    empty_report["scored_targets"] = int(len(score_table))
    if score_table.empty:
        return [], empty_report

    per_group = 0 if targets_per_group is None else int(targets_per_group)
    if per_group <= 0:
        per_group = len(score_table)
    pool_size = None if similarity_top_k_per_group is None else int(similarity_top_k_per_group)
    max_selected = None if max_targets is None or int(max_targets) <= 0 else int(max_targets)

    grouped_samples = {}
    group_counts = {group_name: 0 for group_name in groups}
    for offset, group_name in enumerate(groups):
        group_rows = score_table[score_table["group"] == group_name].sort_values(["score", "target_id"])
        if pool_size is not None and pool_size > 0:
            group_rows = group_rows.head(pool_size)
        sample_count = min(per_group, len(group_rows))
        if sample_count <= 0:
            grouped_samples[group_name] = []
            continue
        sampled = group_rows.sample(
            n=sample_count,
            random_state=int(seed) + offset,
            replace=False,
        ).sort_values(["score", "target_id"])
        grouped_samples[group_name] = sampled["target_id"].astype(str).tolist()

    selected = []
    selected_set = set()
    while any(grouped_samples.values()) and (max_selected is None or len(selected) < max_selected):
        for group_name in groups:
            if max_selected is not None and len(selected) >= max_selected:
                break
            group_list = grouped_samples.get(group_name, [])
            while group_list:
                target_id = group_list.pop(0)
                if target_id in selected_set:
                    continue
                selected.append(target_id)
                selected_set.add(target_id)
                group_counts[group_name] = group_counts.get(group_name, 0) + 1
                break

    report = dict(empty_report)
    report.update(
        {
            "selected_targets": int(len(selected)),
            "group_counts": group_counts,
            "selected_target_sample": selected[:10],
        }
    )
    return selected, report


def prepare_extra_training_data(
    extra_seq_df,
    extra_label_df,
    extra_pair_df,
    base_seq_df,
    val_seq_df,
    validation_pair_df=None,
    max_extra_targets=0,
    max_len=MAX_LEN,
    min_valid_labels=None,
    min_label_coverage=None,
    selection="quality_diverse",
    seed=13,
    length_bin_size=50,
    exclude_base_sequence_overlap=True,
    exclude_validation_sequence_overlap=True,
    contact_map_dir=None,
    structure_reinforcement_extra_targets=0,
    structure_reinforcement_anchor_groups=None,
    structure_reinforcement_targets_per_group=None,
):
    """Purpose: Build a validation-independent extra train subset.

    Input:
        extra_seq_df, extra_label_df, extra_pair_df: Raw extra training DataFrames.
        base_seq_df: Main training sequences used only to avoid duplicate training examples.
        val_seq_df: Validation sequences used only to exclude exact leakage overlaps.
        validation_pair_df: Validation pair features used only for input-side anchor fingerprints.
        max_extra_targets: Maximum number of extra targets to select.
        max_len: Model cap for quality and diversity checks.
        min_valid_labels: Minimum usable coordinate labels per target.
        min_label_coverage: Minimum valid-label coverage per capped target.
        selection: Extra-target selection strategy.
        seed: Deterministic selection seed.
        length_bin_size: Capped-length bin size for diversity selection.
        exclude_base_sequence_overlap: Whether to drop exact base ID/sequence duplicates.
        exclude_validation_sequence_overlap: Whether to drop exact validation ID/sequence overlaps.
        contact_map_dir: SPOT map directory or fallback directories for structure fingerprints.
        structure_reinforcement_extra_targets: Extra pair/contact-similar targets to add.
        structure_reinforcement_anchor_groups: Mapping of structure group to anchor target IDs.
        structure_reinforcement_targets_per_group: Per-group quota before global fill.
    Output:
        Tuple of selected extra sequence, label, pair DataFrames, and a report dict.
    """
    if extra_seq_df.empty:
        empty_pairs = None if extra_pair_df is None else extra_pair_df.copy()
        report = {
            "enabled": False,
            "loaded_targets": 0,
            "after_overlap_targets": 0,
            "quality_kept_targets": 0,
            "selected_targets": 0,
            "excluded_base_overlap": 0,
            "excluded_validation_overlap": 0,
            "quality_removed_targets": 0,
            "quality_diverse_selected_targets": 0,
            "structure_reinforcement_selected_targets": 0,
            "structure_reinforcement_group_counts": {},
            "selected_target_sample": [],
        }
        return extra_seq_df.copy(), extra_label_df.copy(), empty_pairs, report

    work_seq = extra_seq_df.copy()
    work_labels = extra_label_df.copy()
    work_pairs = None if extra_pair_df is None else extra_pair_df.copy()

    extra_ids = work_seq["target_id"].astype(str)
    extra_sequences = normalized_sequences(work_seq)
    base_overlap = pd.Series(False, index=work_seq.index)
    if exclude_base_sequence_overlap:
        base_ids = set(base_seq_df["target_id"].astype(str))
        base_sequences = set(normalized_sequences(base_seq_df))
        base_overlap = extra_ids.isin(base_ids) | extra_sequences.isin(base_sequences)
        keep_ids = work_seq.loc[~base_overlap, "target_id"].astype(str)
        work_seq, work_labels, work_pairs = subset_training_rows(work_seq, work_labels, work_pairs, keep_ids)

    validation_overlap = pd.Series(False, index=work_seq.index)
    if exclude_validation_sequence_overlap:
        val_ids = set(val_seq_df["target_id"].astype(str))
        val_sequences = set(normalized_sequences(val_seq_df))
        validation_overlap = (
            work_seq["target_id"].astype(str).isin(val_ids)
            | normalized_sequences(work_seq).isin(val_sequences)
        )
        keep_ids = work_seq.loc[~validation_overlap, "target_id"].astype(str)
        work_seq, work_labels, work_pairs = subset_training_rows(work_seq, work_labels, work_pairs, keep_ids)

    quality_seq, quality_labels, quality_pairs, quality_report = filter_training_data(
        work_seq,
        work_labels,
        work_pairs,
        max_len=max_len,
        min_valid_labels=min_valid_labels,
        min_label_coverage=min_label_coverage,
    )
    strategy = (selection or "quality_diverse").lower()
    base_selection = "quality_diverse" if strategy == "quality_plus_structure_reinforcement" else strategy
    selected_ids = select_extra_target_ids(
        quality_seq,
        max_extra_targets=max_extra_targets,
        max_len=max_len,
        selection=base_selection,
        seed=seed,
        length_bin_size=length_bin_size,
    )
    quality_diverse_selected_count = len(selected_ids)
    structure_report = {
        "enabled": False,
        "selected_targets": 0,
        "group_counts": {},
        "selected_target_sample": [],
    }
    if strategy == "quality_plus_structure_reinforcement":
        structure_ids, structure_report = select_structure_reinforcement_target_ids(
            quality_seq,
            quality_pairs,
            val_seq_df,
            validation_pair_df,
            structure_reinforcement_anchor_groups,
            contact_map_dir,
            total_targets=structure_reinforcement_extra_targets,
            targets_per_group=structure_reinforcement_targets_per_group,
            used_ids=set(selected_ids),
            max_len=max_len,
        )
        selected_ids = selected_ids + [
            target_id for target_id in structure_ids if target_id not in set(selected_ids)
        ]
    selected_seq, selected_labels, selected_pairs = subset_training_rows(
        quality_seq,
        quality_labels,
        quality_pairs,
        selected_ids,
    )
    report = {
        "enabled": True,
        "loaded_targets": int(len(extra_seq_df)),
        "after_overlap_targets": int(len(work_seq)),
        "quality_kept_targets": int(len(quality_seq)),
        "selected_targets": int(len(selected_seq)),
        "excluded_base_overlap": int(base_overlap.sum()),
        "excluded_validation_overlap": int(validation_overlap.sum()),
        "quality_removed_targets": int(quality_report["removed_targets"]),
        "selection": selection,
        "max_extra_targets": max_extra_targets,
        "quality_diverse_selected_targets": int(quality_diverse_selected_count),
        "structure_reinforcement_selected_targets": int(structure_report["selected_targets"]),
        "structure_reinforcement_group_counts": structure_report["group_counts"],
        "structure_reinforcement_target_sample": structure_report["selected_target_sample"],
        "selected_target_sample": selected_seq["target_id"].astype(str).head(10).tolist(),
    }
    return selected_seq, selected_labels, selected_pairs, report


def load_extra_train_sources(extra_train_data_dirs, pair_columns=None):
    """Purpose: Load one or more extra train-only dataset directories.

    Input:
        extra_train_data_dirs: Directory or directories with train CSV files.
        pair_columns: Pair-feature columns to use when an extra directory lacks pair features.
    Output:
        Tuple of concatenated sequence, label, and pair-feature DataFrames.
    """
    data_dirs = normalize_path_list(extra_train_data_dirs)
    seq_frames = []
    label_frames = []
    pair_frames = []

    for data_dir in data_dirs:
        seq_path = os.path.join(data_dir, "train_sequences.csv")
        label_path = os.path.join(data_dir, "train_labels.csv")
        pair_path = os.path.join(data_dir, "train_pair_features.csv")
        if not os.path.exists(seq_path):
            raise FileNotFoundError(f"Extra train sequences not found: {seq_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Extra train labels not found: {label_path}")

        seq_frames.append(pd.read_csv(seq_path))
        label_frames.append(pd.read_csv(label_path))
        if os.path.exists(pair_path):
            pair_frames.append(pd.read_csv(pair_path))
        elif pair_columns is not None:
            pair_frames.append(pd.DataFrame(columns=list(pair_columns)))

    if not seq_frames:
        empty_pairs = None if pair_columns is None else pd.DataFrame(columns=list(pair_columns))
        return pd.DataFrame(), pd.DataFrame(), empty_pairs

    seq_df = pd.concat(seq_frames, ignore_index=True)
    label_df = pd.concat(label_frames, ignore_index=True)
    if pair_frames:
        pair_df = pd.concat(pair_frames, ignore_index=True)
    else:
        pair_df = None
    return seq_df, label_df, pair_df


def append_training_data(train_seq_df, train_label_df, train_pair_df, extra_seq_df, extra_label_df, extra_pair_df):
    """Purpose: Append selected extra targets to the main training DataFrames.

    Input:
        train_seq_df, train_label_df, train_pair_df: Main training DataFrames.
        extra_seq_df, extra_label_df, extra_pair_df: Selected extra training DataFrames.
    Output:
        Tuple of merged sequence, label, and pair-feature DataFrames.
    """
    if extra_seq_df.empty:
        return train_seq_df, train_label_df, train_pair_df

    merged_seq = pd.concat([train_seq_df, extra_seq_df], ignore_index=True)
    merged_labels = pd.concat([train_label_df, extra_label_df], ignore_index=True)
    if train_pair_df is None:
        merged_pairs = extra_pair_df
    elif extra_pair_df is None:
        merged_pairs = train_pair_df
    else:
        merged_pairs = pd.concat([train_pair_df, extra_pair_df], ignore_index=True)
    return merged_seq, merged_labels, merged_pairs


def print_extra_training_report(report):
    """Purpose: Print a concise extra-train selection report.

    Input:
        report: Dictionary returned by prepare_extra_training_data().
    Output:
        None. Writes the summary to stdout.
    """
    if not report["enabled"]:
        print("Extra train data: disabled.")
        return
    print(
        "Extra train data: "
        f"loaded {report['loaded_targets']} targets, "
        f"excluded base overlaps {report['excluded_base_overlap']}, "
        f"excluded validation overlaps {report['excluded_validation_overlap']}, "
        f"quality kept {report['quality_kept_targets']}, "
        f"selected {report['selected_targets']}."
    )
    if report.get("structure_reinforcement_selected_targets", 0):
        print(
            "Structure reinforcement: "
            f"quality_diverse={report.get('quality_diverse_selected_targets', 0)}, "
            f"structure_reinforcement={report['structure_reinforcement_selected_targets']}, "
            f"group_counts={report.get('structure_reinforcement_group_counts', {})}."
        )
        if report.get("structure_reinforcement_target_sample"):
            print(
                "Structure reinforcement sample:",
                ", ".join(report["structure_reinforcement_target_sample"]),
            )
    if report["selected_target_sample"]:
        print("Extra target sample:", ", ".join(report["selected_target_sample"]))

def masked_rmse_loss(pred,target,mask,eps=1e-8):
    """Purpose: Compute RMSE while ignoring padded or invalid residues.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Target coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        eps: Small value to keep the square root stable.
    Output:
        Scalar masked RMSE tensor.
    """
    mask = mask.float().unsqueeze(-1)           # (B,L,1)
    diff2 = ((pred - target)**2) * mask
    valid = mask.sum().clamp(min=1.0)
    mse_loss = diff2.sum() / valid
    rmse_loss = torch.sqrt(mse_loss + eps)
    return rmse_loss

def kabsch_rmse_loss(pred, target, mask, raw_weight=0.0, aligned_weight=1):
    """Purpose: Combine raw and Kabsch-aligned coordinate RMSE.

    Input:
        pred: Raw predicted coordinates shaped (B, L, 3).
        target: Normalized ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        raw_weight: Weight for raw coordinate RMSE.
        aligned_weight: Weight for Kabsch-aligned RMSE.
    Output:
        Tuple of weighted loss, aligned prediction, raw RMSE, and aligned RMSE.
    """
    aligned_pred = kabsch_align_batch(pred, target, mask)

    raw_rmse = masked_rmse_loss(pred, target, mask)
    aligned_rmse = masked_rmse_loss(aligned_pred, target, mask)

    loss = raw_weight * raw_rmse + aligned_weight * aligned_rmse

    return loss, aligned_pred, raw_rmse, aligned_rmse

def spot_weighted_distance_loss(pred, target, mask, contact_map, min_seq_sep=4, eps=1e-8):
    """Purpose: Penalize pairwise-distance errors emphasized by SPOT contacts.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        contact_map: SPOT-RNA-2D contact probabilities shaped (B, L, L).
        min_seq_sep: Minimum sequence separation for weighted contact pairs.
        eps: Small value to keep the square root stable.
    Output:
        Scalar SPOT-weighted distance loss tensor.
    """
    if contact_map is None:
        return pred.new_tensor(0.0)

    B, L, _ = pred.shape
    contact_map = contact_map[:, :L, :L].to(device=pred.device, dtype=pred.dtype)

    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)

    valid_pair = mask.unsqueeze(1) * mask.unsqueeze(2)
    idx = torch.arange(L, device=pred.device)
    seq_sep_mask = (torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) >= min_seq_sep).to(pred.dtype)
    weights = contact_map * valid_pair * seq_sep_mask.unsqueeze(0)

    valid = weights.sum()
    if valid.item() <= 0:
        return pred.new_tensor(0.0)

    diff2 = (pred_dist - target_dist) ** 2
    return torch.sqrt((diff2 * weights).sum() / valid + eps)

def distance_matrix_loss(pred, target, mask, eps=1e-8):
    """Purpose: Preserve global structure through all pairwise distances.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        eps: Small value to keep the square root stable.
    Output:
        Scalar full distance-matrix RMSE tensor.
    """
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)

    pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).to(dtype=pred.dtype)
    valid = pair_mask.sum()
    if valid.item() <= 0:
        return pred.new_tensor(0.0)

    diff2 = (pred_dist - target_dist) ** 2
    return torch.sqrt((diff2 * pair_mask).sum() / valid + eps)

def adjacent_bond_range_loss(pred, mask, lower=0.18, upper=1.20, eps=1e-8):
    """Purpose: Discourage adjacent residues from becoming too close or far.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        lower: Minimum preferred adjacent-residue distance.
        upper: Maximum preferred adjacent-residue distance.
        eps: Small value to keep the square root stable.
    Output:
        Scalar adjacent bond range loss tensor.
    """
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    distances = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1)
    pair_mask = (mask[:, 1:] * mask[:, :-1]).to(dtype=pred.dtype)
    valid = pair_mask.sum()
    if valid.item() <= 0:
        return pred.new_tensor(0.0)

    too_short = torch.relu(lower - distances) ** 2
    too_long = torch.relu(distances - upper) ** 2
    return torch.sqrt(((too_short + too_long) * pair_mask).sum() / valid + eps)

def adjacent_target_distance_loss(pred, target, mask, eps=1e-8):
    """Purpose: Match predicted adjacent-residue distances to target distances.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        eps: Small value to keep the square root stable.
    Output:
        Scalar adjacent-distance RMSE tensor.
    """
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    pred_dist = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1)
    target_dist = torch.norm(target[:, 1:] - target[:, :-1], dim=-1)
    pair_mask = (mask[:, 1:] * mask[:, :-1]).to(dtype=pred.dtype)
    valid = pair_mask.sum()
    if valid.item() <= 0:
        return pred.new_tensor(0.0)

    diff2 = (pred_dist - target_dist) ** 2
    return torch.sqrt((diff2 * pair_mask).sum() / valid + eps)

def sequence_range_distance_loss(pred, target, mask, min_sep, max_sep, eps=1e-8):
    """Purpose: Preserve geometry for pairs within a sequence-separation range.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        min_sep: Minimum sequence separation included in the loss.
        max_sep: Maximum sequence separation included in the loss.
        eps: Small value to keep the square root stable.
    Output:
        Scalar distance RMSE tensor over the selected residue-pair range.
    """
    if pred.shape[1] < 2 or max_sep < min_sep:
        return pred.new_tensor(0.0)

    _, L, _ = pred.shape
    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)

    valid_pair = mask.unsqueeze(1) * mask.unsqueeze(2)
    idx = torch.arange(L, device=pred.device)
    seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    upper = torch.triu(torch.ones((L, L), dtype=torch.bool, device=pred.device), diagonal=1)
    selected_pair = (seq_sep >= min_sep) & (seq_sep <= max_sep) & upper
    weights = valid_pair.to(dtype=pred.dtype) * selected_pair.unsqueeze(0).to(dtype=pred.dtype)

    valid = weights.sum()
    if valid.item() <= 0:
        return pred.new_tensor(0.0)

    diff2 = (pred_dist - target_dist) ** 2
    return torch.sqrt((diff2 * weights).sum() / valid + eps)

def short_range_distance_loss(pred, target, mask, max_sep=4, eps=1e-8):
    """Purpose: Preserve local geometry through short-range pair distances.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        max_sep: Maximum sequence separation included in the local loss.
        eps: Small value to keep the square root stable.
    Output:
        Scalar short-range distance RMSE tensor.
    """
    return sequence_range_distance_loss(pred, target, mask, min_sep=1, max_sep=max_sep, eps=eps)

def medium_range_distance_loss(pred, target, mask, min_sep=5, max_sep=12, eps=1e-8):
    """Purpose: Preserve medium-range topology between local and global pairs.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        min_sep: Minimum sequence separation included in the loss.
        max_sep: Maximum sequence separation included in the loss.
        eps: Small value to keep the square root stable.
    Output:
        Scalar medium-range distance RMSE tensor.
    """
    return sequence_range_distance_loss(pred, target, mask, min_sep=min_sep, max_sep=max_sep, eps=eps)

def curvature_loss(pred, target, mask, eps=1e-8):
    """Purpose: Match local turning angles formed by consecutive residues.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        eps: Small value to keep cosine normalization stable.
    Output:
        Scalar RMSE between predicted and target cosine-angle values.
    """
    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)

    pred_v1 = pred[:, 1:-1] - pred[:, :-2]
    pred_v2 = pred[:, 2:] - pred[:, 1:-1]
    target_v1 = target[:, 1:-1] - target[:, :-2]
    target_v2 = target[:, 2:] - target[:, 1:-1]

    pred_norm = torch.norm(pred_v1, dim=-1) * torch.norm(pred_v2, dim=-1)
    target_norm = torch.norm(target_v1, dim=-1) * torch.norm(target_v2, dim=-1)
    pred_cos = (pred_v1 * pred_v2).sum(dim=-1) / pred_norm.clamp(min=eps)
    target_cos = (target_v1 * target_v2).sum(dim=-1) / target_norm.clamp(min=eps)

    triple_mask = (mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]).to(dtype=pred.dtype)
    valid = triple_mask.sum()
    if valid.item() <= 0:
        return pred.new_tensor(0.0)

    diff2 = (pred_cos - target_cos) ** 2
    return torch.sqrt((diff2 * triple_mask).sum() / valid + eps)

def radius_of_gyration_loss(pred, target, mask, eps=1e-8):
    """Purpose: Match the global coordinate spread of each predicted structure.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        eps: Small value to keep division and square roots stable.
    Output:
        Scalar RMSE between predicted and target radius of gyration values.
    """
    residue_mask = mask.to(dtype=pred.dtype).unsqueeze(-1)
    valid_counts = residue_mask.sum(dim=1).clamp(min=1.0)
    has_valid_residues = mask.sum(dim=1) > 0
    if not has_valid_residues.any():
        return pred.new_tensor(0.0)

    pred_center = (pred * residue_mask).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(1)
    target_center = (target * residue_mask).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(1)

    pred_sq_radius = (((pred - pred_center) ** 2).sum(dim=-1) * mask.to(dtype=pred.dtype)).sum(dim=1)
    target_sq_radius = (((target - target_center) ** 2).sum(dim=-1) * mask.to(dtype=pred.dtype)).sum(dim=1)
    pred_radius = torch.sqrt(pred_sq_radius / valid_counts.squeeze(-1) + eps)
    target_radius = torch.sqrt(target_sq_radius / valid_counts.squeeze(-1) + eps)

    diff2 = (pred_radius - target_radius) ** 2
    return torch.sqrt(diff2[has_valid_residues].mean() + eps)

def compute_accuracy(pred, target, mask, threshold=2):
    """Purpose: Compute residue-level distance-threshold accuracy.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        threshold: Distance threshold for a residue to count as correct.
    Output:
        Float accuracy over valid residues.
    """
    mask = mask.float()  # (B, L)

    # per residue distances
    dist = torch.norm((pred - target), dim=-1)  # (B, L)

    # apply mask
    correct = ((dist < threshold) * mask).sum().item()
    total = mask.sum().item()

    if total == 0:
        return 0.0

    return correct / total

def compute_accuracy_at_thresholds(pred, target, mask, thresholds):
    """Purpose: Compute accuracy for several distance thresholds.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        thresholds: Iterable of distance thresholds.
    Output:
        Dictionary mapping threshold to accuracy.
    """
    return {
        threshold: compute_accuracy(pred, target, mask, threshold=threshold)
        for threshold in thresholds
    }

def format_accuracy_metrics(prefix, metrics):
    """Purpose: Format threshold accuracy values for console logging.

    Input:
        prefix: Label such as "Train Raw" or "Val Aligned".
        metrics: Dictionary mapping threshold to accuracy.
    Output:
        Human-readable comma-separated metric string.
    """
    return ", ".join(
        f"{prefix} Acc@{threshold:g}: {value:.4f}"
        for threshold, value in metrics.items()
    )

def zero_accuracy_totals(thresholds):
    """Purpose: Initialize running sums for threshold accuracy metrics.

    Input:
        thresholds: Iterable of thresholds to track.
    Output:
        Dictionary mapping threshold to zero.
    """
    return {threshold: 0.0 for threshold in thresholds}

def add_accuracy_totals(totals, batch_metrics):
    """Purpose: Add one batch of accuracy metrics into running totals.

    Input:
        totals: Running metric totals updated in place.
        batch_metrics: Metrics from the current batch.
    Output:
        None. The totals dictionary is mutated.
    """
    for threshold, value in batch_metrics.items():
        totals[threshold] += value

def average_accuracy_totals(totals, count):
    """Purpose: Convert accumulated batch accuracy totals into averages.

    Input:
        totals: Running metric totals.
        count: Number of batches accumulated.
    Output:
        Dictionary mapping threshold to average accuracy.
    """
    return {threshold: value / count for threshold, value in totals.items()}

def compute_loss_bundle(
    outputs,
    labels,
    mask,
    contact_map,
    use_contact_map,
    raw_loss_weight,
    aligned_loss_weight,
    spot_loss_weight,
    bond_loss_weight,
    distmap_loss_weight,
    adj_loss_weight,
    short_range_loss_weight,
    medium_range_loss_weight,
    curvature_loss_weight,
    spread_loss_weight,
    bond_lower,
    bond_upper,
    short_range_max_sep,
    medium_range_min_sep,
    medium_range_max_sep,
):
    """Purpose: Compute all active loss terms from one model output batch.

    Input:
        outputs: Raw model predictions shaped (B, L, 3).
        labels: Normalized target coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        contact_map: SPOT contact map shaped (B, L, L).
        use_contact_map: Whether SPOT map should affect attention/loss.
        raw_loss_weight: Weight for raw RMSE inside the base loss.
        aligned_loss_weight: Weight for Kabsch-aligned RMSE inside the base loss.
        spot_loss_weight: Weight for SPOT-weighted distance loss.
        bond_loss_weight: Weight for adjacent bond range loss.
        distmap_loss_weight: Weight for full distance-matrix loss.
        adj_loss_weight: Weight for adjacent target-distance loss.
        short_range_loss_weight: Weight for short-range distance loss.
        medium_range_loss_weight: Weight for medium-range distance loss.
        curvature_loss_weight: Weight for local cosine-angle loss.
        spread_loss_weight: Weight for radius-of-gyration spread loss.
        bond_lower: Lower adjacent bond range.
        bond_upper: Upper adjacent bond range.
        short_range_max_sep: Maximum sequence separation for short-range loss.
        medium_range_min_sep: Minimum sequence separation for medium-range loss.
        medium_range_max_sep: Maximum sequence separation for medium-range loss.
    Output:
        Dictionary with total loss, individual loss terms, and aligned outputs.
    """
    base_loss, aligned_outputs, raw_rmse, aligned_rmse = kabsch_rmse_loss(
        outputs,
        labels,
        mask,
        raw_weight=raw_loss_weight,
        aligned_weight=aligned_loss_weight,
    )
    spot_loss = (
        spot_weighted_distance_loss(outputs, labels, mask, contact_map)
        if use_contact_map and spot_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    bond_loss = (
        adjacent_bond_range_loss(outputs, mask, lower=bond_lower, upper=bond_upper)
        if bond_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    distmap_loss = (
        distance_matrix_loss(outputs, labels, mask)
        if distmap_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    adj_loss = (
        adjacent_target_distance_loss(outputs, labels, mask)
        if adj_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    short_range_loss = (
        short_range_distance_loss(outputs, labels, mask, max_sep=short_range_max_sep)
        if short_range_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    medium_range_loss = (
        medium_range_distance_loss(
            outputs,
            labels,
            mask,
            min_sep=medium_range_min_sep,
            max_sep=medium_range_max_sep,
        )
        if medium_range_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    turn_loss = (
        curvature_loss(outputs, labels, mask)
        if curvature_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    spread_loss = (
        radius_of_gyration_loss(outputs, labels, mask)
        if spread_loss_weight != 0
        else outputs.new_tensor(0.0)
    )
    local_geometry_loss = (
        adj_loss_weight * adj_loss
        + short_range_loss_weight * short_range_loss
        + medium_range_loss_weight * medium_range_loss
        + curvature_loss_weight * turn_loss
    )
    total_loss = (
        base_loss
        + spot_loss_weight * spot_loss
        + bond_loss_weight * bond_loss
        + distmap_loss_weight * distmap_loss
        + local_geometry_loss
        + spread_loss_weight * spread_loss
    )
    return {
        "total": total_loss,
        "base": base_loss,
        "raw_rmse": raw_rmse,
        "aligned_rmse": aligned_rmse,
        "spot": spot_loss,
        "bond": bond_loss,
        "distmap": distmap_loss,
        "adj": adj_loss,
        "short_range": short_range_loss,
        "medium_range": medium_range_loss,
        "curvature": turn_loss,
        "spread": spread_loss,
        "local_geometry": local_geometry_loss,
        "aligned_outputs": aligned_outputs,
    }


def train_validate(
    train,
    label,
    val,
    val_label,
    msa_dir,
    train_pair_df,
    val_pair_df,
    save_path = None,
    output_dir = None,
    output_root = OUTPUT_ROOT,
    init_model_path = None,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    Ir = IR,
    max_len = MAX_LEN,
    patience = PATIENCE,
    weight_decay = 1e-4,
    contact_map_dir = None,
    spot_bias_scale = 1.0,
    use_graph = False,
    graph_layers = 0,
    graph_scale = 0.10,
    spot_edge_threshold = 0.50,
    spot_top_k = 8,
    local_edge_max_sep = 4,
    coord_refine_steps = 0,
    coord_refine_hidden = 128,
    coord_refine_dropout = 0.05,
    coord_refine_local_window = 4,
    coord_refine_delta_scale = 0.10,
    spot_loss_weight = 0.10,
    raw_loss_weight = 0.05,
    aligned_loss_weight = 1.0,
    bond_loss_weight = 0.30,
    distmap_loss_weight = 0.30,
    adj_loss_weight = 0.0,
    short_range_loss_weight = 0.0,
    medium_range_loss_weight = 0.0,
    curvature_loss_weight = 0.0,
    spread_loss_weight = 0.0,
    bond_lower = 0.30,
    bond_upper = 1.20,
    short_range_max_sep = 4,
    medium_range_min_sep = 5,
    medium_range_max_sep = 12,
    accuracy_thresholds = None,
    min_train_valid_labels = None,
    min_train_label_coverage = None,
    max_train_targets = None,
    train_selection = "quality_diverse",
    train_seed = 13,
    train_length_bin_size = 50,
    train_structure_anchor_groups = None,
    train_targets_per_group = 150,
    train_similarity_top_k_per_group = 300,
    used_config = None,
):
    """Purpose: Train the RNA 3D model and save run-specific outputs.

    Input:
        train, label, val, val_label: Training/validation DataFrames.
        msa_dir: Directory containing MSA FASTA files.
        train_pair_df, val_pair_df: ViennaRNA pair-feature DataFrames.
        save_path: Optional explicit best-checkpoint path.
        output_dir: Optional explicit run directory.
        output_root: Parent directory for numbered output runs.
        init_model_path: Optional checkpoint used to initialize model weights.
        epochs, batch_size, Ir, max_len, patience, weight_decay: Training hyperparameters.
        contact_map_dir: Optional SPOT-RNA-2D contact-map directory.
        graph settings: Optional residual graph message-passing hyperparameters.
        coordinate refinement settings: Optional iterative coordinate-head hyperparameters.
        loss weights and bond bounds: Hyperparameters for the hybrid loss.
        accuracy_thresholds: Distance thresholds for raw/aligned accuracy.
        min_train_valid_labels: Minimum usable coordinate labels per train target.
        min_train_label_coverage: Minimum usable-label fraction per train target.
        max_train_targets: Optional cap for the number of train targets consumed.
        train_selection: Strategy for selecting train targets when capped.
        train_seed: Deterministic seed for capped quality_diverse train selection.
        train_length_bin_size: Length-bin width for capped quality_diverse train selection.
        train_structure_anchor_groups: Optional validation anchor groups for structure sampling.
        train_targets_per_group: Per-anchor-group cap for structure sampling.
        train_similarity_top_k_per_group: Top-K similar candidates used as random sampling pool.
        used_config: Optional effective config to write into the output directory.
    Output:
        Tuple of (best validation loss, history dictionary).
    """
    if accuracy_thresholds is None:
        accuracy_thresholds = [0.5, 0.75, 1.0]
    main_threshold = select_accuracy_threshold(accuracy_thresholds, preferred=1.0)
    main_threshold_tag = threshold_tag(main_threshold)
    if output_dir is None:
        if save_path is None:
            output_dir = next_output_dir(output_root)
        else:
            output_dir = os.path.dirname(save_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(output_dir, "best_model.pth")
    last_model_path = os.path.join(output_dir, "last_model.pth")
    best_local_model_path = os.path.join(output_dir, "best_local_geometry.pth")
    if used_config is not None:
        write_used_config(used_config, output_dir)
    history  = {'epoch':[],'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    history_rows = []
    contact_map_sources = normalize_path_list(contact_map_dir)
    use_contact_maps = bool(contact_map_sources)
    dataset_contact_map_dir = contact_map_sources if use_contact_maps else None
    train, label, train_pair_df, filter_report = filter_training_data(
        train,
        label,
        train_pair_df,
        max_len=max_len,
        min_valid_labels=min_train_valid_labels,
        min_label_coverage=min_train_label_coverage,
    )
    print_training_filter_report(filter_report)
    train, label, train_pair_df, train_limit_report = limit_training_data(
        train,
        label,
        train_pair_df,
        max_train_targets=max_train_targets,
        max_len=max_len,
        selection=train_selection,
        seed=train_seed,
        length_bin_size=train_length_bin_size,
        val_seq_df=val,
        val_pair_df=val_pair_df,
        contact_map_dir=dataset_contact_map_dir,
        structure_anchor_groups=train_structure_anchor_groups,
        targets_per_group=train_targets_per_group,
        similarity_top_k_per_group=train_similarity_top_k_per_group,
    )
    print_train_limit_report(train_limit_report)
    train_dataset = RNA(
        train,
        label,
        msa_dir,
        pair_df=train_pair_df,
        max_len=max_len,
        contact_map_dir=dataset_contact_map_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle = True,
        collate_fn=RNA.collate_fn, 
        )
    val_dataset = RNA(
        val,
        val_label,
        msa_dir,
        pair_df=val_pair_df,
        max_len=max_len,
        contact_map_dir=dataset_contact_map_dir,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RNA.collate_fn,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_state = None
    input_channels = 8
    if init_model_path is not None:
        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"Initial checkpoint not found: {init_model_path}")
        checkpoint_state = torch.load(init_model_path, map_location=device)
        input_channels = checkpoint_state["conv_block.0.weight"].shape[1]
        checkpoint_max_len = checkpoint_state["pos_embed.weight"].shape[0]
        if checkpoint_max_len != max_len:
            raise ValueError(
                f"Checkpoint max_len={checkpoint_max_len}, but configured max_len={max_len}."
            )
    model = RNAmodel(
        input_channels=input_channels,
        max_len=max_len,
        spot_bias_scale=spot_bias_scale,
        use_graph=use_graph,
        graph_layers=graph_layers,
        graph_scale=graph_scale,
        spot_edge_threshold=spot_edge_threshold,
        spot_top_k=spot_top_k,
        local_edge_max_sep=local_edge_max_sep,
        coord_refine_steps=coord_refine_steps,
        coord_refine_hidden=coord_refine_hidden,
        coord_refine_dropout=coord_refine_dropout,
        coord_refine_local_window=coord_refine_local_window,
        coord_refine_delta_scale=coord_refine_delta_scale,
    ).to(device)
    if checkpoint_state is not None:
        if (use_graph and graph_layers > 0) or coord_refine_steps > 0:
            load_result = model.load_state_dict(checkpoint_state, strict=False)
            print("Loaded existing checkpoint with strict=False for architecture fine-tuning.")
            print(f"Missing checkpoint keys: {list(load_result.missing_keys)}")
            print(f"Unexpected checkpoint keys: {list(load_result.unexpected_keys)}")
        else:
            model.load_state_dict(checkpoint_state)
        print(f"Fine-tuning from checkpoint: {init_model_path}")
    else:
        print("Training from scratch with 8 input channels, including pair features.")
    if use_contact_maps:
        print(f"Using SPOT contact maps from: {contact_map_sources}")
        print(f"SPOT attention bias scale: {spot_bias_scale}, SPOT loss weight: {spot_loss_weight}")
    print(f"Run output directory: {output_dir}")
    print(f"Best checkpoint path: {save_path}")
    print(f"Last epoch checkpoint path: {last_model_path}")
    print(
        "Graph message passing: "
        f"use_graph={use_graph}, graph_layers={graph_layers}, graph_scale={graph_scale}, "
        f"spot_edge_threshold={spot_edge_threshold}, spot_top_k={spot_top_k}, "
        f"local_edge_max_sep={local_edge_max_sep}"
    )
    print(
        "Coordinate refinement: "
        f"steps={coord_refine_steps}, hidden={coord_refine_hidden}, "
        f"dropout={coord_refine_dropout}, local_window={coord_refine_local_window}, "
        f"delta_scale={coord_refine_delta_scale}"
    )
    print(f"Kabsch RMSE weights: raw={raw_loss_weight}, aligned={aligned_loss_weight}")
    print(f"Adjacent bond range loss: weight={bond_loss_weight}, lower={bond_lower}, upper={bond_upper}")
    print(f"Full distance matrix loss weight: {distmap_loss_weight}")
    print(f"Adjacent target distance loss weight: {adj_loss_weight}")
    print(f"Short-range distance loss: weight={short_range_loss_weight}, max_sep={short_range_max_sep}")
    print(
        f"Medium-range distance loss: weight={medium_range_loss_weight}, "
        f"min_sep={medium_range_min_sep}, max_sep={medium_range_max_sep}"
    )
    print(f"Curvature loss weight: {curvature_loss_weight}")
    print(f"Radius-of-gyration spread loss weight: {spread_loss_weight}")
    print("Accuracy thresholds:", ", ".join(f"{t:g}" for t in accuracy_thresholds))
    print(f"Validation accuracy plot threshold: {main_threshold:g}")
    optimizer = optim.Adam(model.parameters(), lr=Ir, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_local_geometry_loss = float('inf')
    use_local_geometry_checkpoint = (
        adj_loss_weight != 0
        or short_range_loss_weight != 0
        or medium_range_loss_weight != 0
        or curvature_loss_weight != 0
    )
    patience_counter = 0
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_base_loss = 0.0
        epoch_raw_rmse = 0.0
        epoch_aligned_rmse = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_spot_loss = 0.0
        epoch_bond_loss = 0.0
        epoch_distmap_loss = 0.0
        epoch_adj_loss = 0.0
        epoch_short_range_loss = 0.0
        epoch_medium_range_loss = 0.0
        epoch_curvature_loss = 0.0
        epoch_spread_loss = 0.0
        epoch_local_geometry_loss = 0.0
        train_raw_acc = zero_accuracy_totals(accuracy_thresholds)
        train_aligned_acc = zero_accuracy_totals(accuracy_thresholds)
        for feats,labels,contact_map,seq_mask,coord_mask,lengths,ids in progress:
            
            feats = feats.to(device)
            labels = labels.to(device)
            contact_map = contact_map.to(device)
            seq_mask = seq_mask.to(device)
            coord_mask = coord_mask.to(device)
            optimizer.zero_grad()
            model_contact_map = contact_map if use_contact_maps else None
            outputs = model(feats,seq_mask,contact_map=model_contact_map)
            losses = compute_loss_bundle(
                outputs,
                labels,
                coord_mask,
                contact_map,
                use_contact_map=use_contact_maps,
                raw_loss_weight=raw_loss_weight,
                aligned_loss_weight=aligned_loss_weight,
                spot_loss_weight=spot_loss_weight,
                bond_loss_weight=bond_loss_weight,
                distmap_loss_weight=distmap_loss_weight,
                adj_loss_weight=adj_loss_weight,
                short_range_loss_weight=short_range_loss_weight,
                medium_range_loss_weight=medium_range_loss_weight,
                curvature_loss_weight=curvature_loss_weight,
                spread_loss_weight=spread_loss_weight,
                bond_lower=bond_lower,
                bond_upper=bond_upper,
                short_range_max_sep=short_range_max_sep,
                medium_range_min_sep=medium_range_min_sep,
                medium_range_max_sep=medium_range_max_sep,
            )
            loss = losses["total"]
            aligned_outputs = losses["aligned_outputs"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_base_loss += losses["base"].item()
            epoch_raw_rmse += losses["raw_rmse"].item()
            epoch_aligned_rmse += losses["aligned_rmse"].item()
            epoch_spot_loss += losses["spot"].item()
            epoch_bond_loss += losses["bond"].item()
            epoch_distmap_loss += losses["distmap"].item()
            epoch_adj_loss += losses["adj"].item()
            epoch_short_range_loss += losses["short_range"].item()
            epoch_medium_range_loss += losses["medium_range"].item()
            epoch_curvature_loss += losses["curvature"].item()
            epoch_spread_loss += losses["spread"].item()
            epoch_local_geometry_loss += losses["local_geometry"].item()
            raw_batch_acc = compute_accuracy_at_thresholds(outputs, labels, coord_mask, accuracy_thresholds)
            aligned_batch_acc = compute_accuracy_at_thresholds(aligned_outputs, labels, coord_mask, accuracy_thresholds)
            add_accuracy_totals(train_raw_acc, raw_batch_acc)
            add_accuracy_totals(train_aligned_acc, aligned_batch_acc)
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_base_loss = epoch_base_loss / len(train_loader)
        avg_epoch_raw_rmse = epoch_raw_rmse / len(train_loader)
        avg_epoch_aligned_rmse = epoch_aligned_rmse / len(train_loader)
        avg_epoch_spot_loss = epoch_spot_loss / len(train_loader)
        avg_epoch_bond_loss = epoch_bond_loss / len(train_loader)
        avg_epoch_distmap_loss = epoch_distmap_loss / len(train_loader)
        avg_epoch_adj_loss = epoch_adj_loss / len(train_loader)
        avg_epoch_short_range_loss = epoch_short_range_loss / len(train_loader)
        avg_epoch_medium_range_loss = epoch_medium_range_loss / len(train_loader)
        avg_epoch_curvature_loss = epoch_curvature_loss / len(train_loader)
        avg_epoch_spread_loss = epoch_spread_loss / len(train_loader)
        avg_epoch_local_geometry_loss = epoch_local_geometry_loss / len(train_loader)
        avg_train_raw_acc = average_accuracy_totals(train_raw_acc, len(train_loader))
        avg_train_aligned_acc = average_accuracy_totals(train_aligned_acc, len(train_loader))
        avg_train_acc = avg_train_aligned_acc[main_threshold]
        print(
            f"Epoch {epoch+1} Train Loss: {avg_epoch_loss:.4f}, "
            f"Base Loss: {avg_epoch_base_loss:.4f}, Raw RMSE: {avg_epoch_raw_rmse:.4f}, "
            f"Aligned RMSE: {avg_epoch_aligned_rmse:.4f}, SPOT Loss: {avg_epoch_spot_loss:.4f}, "
            f"Bond Loss: {avg_epoch_bond_loss:.4f}, DistMap Loss: {avg_epoch_distmap_loss:.4f}, "
            f"Adj Loss: {avg_epoch_adj_loss:.4f}, ShortRange Loss: {avg_epoch_short_range_loss:.4f}, "
            f"MediumRange Loss: {avg_epoch_medium_range_loss:.4f}, "
            f"Curvature Loss: {avg_epoch_curvature_loss:.4f}, "
            f"Spread Loss: {avg_epoch_spread_loss:.4f}, "
            f"Local Geometry Loss: {avg_epoch_local_geometry_loss:.4f}"
        )
        print(format_accuracy_metrics("Train Raw", avg_train_raw_acc))
        print(format_accuracy_metrics("Train Aligned", avg_train_aligned_acc))
        
        model.eval()
        val_loss = 0.0
        val_base_loss = 0.0
        val_raw_rmse = 0.0
        val_aligned_rmse = 0.0
        val_spot_loss = 0.0
        val_bond_loss = 0.0
        val_distmap_loss = 0.0
        val_adj_loss = 0.0
        val_short_range_loss = 0.0
        val_medium_range_loss = 0.0
        val_curvature_loss = 0.0
        val_spread_loss = 0.0
        val_local_geometry_loss = 0.0
        val_raw_acc = zero_accuracy_totals(accuracy_thresholds)
        val_aligned_acc = zero_accuracy_totals(accuracy_thresholds)
        with torch.no_grad():
            for feats,labels,contact_map,seq_mask,coord_mask,lengths,ids in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                contact_map = contact_map.to(device)
                seq_mask = seq_mask.to(device)
                coord_mask = coord_mask.to(device)
                model_contact_map = contact_map if use_contact_maps else None
                outputs = model(feats,seq_mask,contact_map=model_contact_map)
                losses = compute_loss_bundle(
                    outputs,
                    labels,
                    coord_mask,
                    contact_map,
                    use_contact_map=use_contact_maps,
                    raw_loss_weight=raw_loss_weight,
                    aligned_loss_weight=aligned_loss_weight,
                    spot_loss_weight=spot_loss_weight,
                    bond_loss_weight=bond_loss_weight,
                    distmap_loss_weight=distmap_loss_weight,
                    adj_loss_weight=adj_loss_weight,
                    short_range_loss_weight=short_range_loss_weight,
                    medium_range_loss_weight=medium_range_loss_weight,
                    curvature_loss_weight=curvature_loss_weight,
                    spread_loss_weight=spread_loss_weight,
                    bond_lower=bond_lower,
                    bond_upper=bond_upper,
                    short_range_max_sep=short_range_max_sep,
                    medium_range_min_sep=medium_range_min_sep,
                    medium_range_max_sep=medium_range_max_sep,
                )
                loss = losses["total"]
                aligned_outputs = losses["aligned_outputs"]
                val_loss += loss.item()
                val_base_loss += losses["base"].item()
                val_raw_rmse += losses["raw_rmse"].item()
                val_aligned_rmse += losses["aligned_rmse"].item()
                val_spot_loss += losses["spot"].item()
                val_bond_loss += losses["bond"].item()
                val_distmap_loss += losses["distmap"].item()
                val_adj_loss += losses["adj"].item()
                val_short_range_loss += losses["short_range"].item()
                val_medium_range_loss += losses["medium_range"].item()
                val_curvature_loss += losses["curvature"].item()
                val_spread_loss += losses["spread"].item()
                val_local_geometry_loss += losses["local_geometry"].item()
                raw_batch_acc = compute_accuracy_at_thresholds(outputs, labels, coord_mask, accuracy_thresholds)
                aligned_batch_acc = compute_accuracy_at_thresholds(aligned_outputs, labels, coord_mask, accuracy_thresholds)
                add_accuracy_totals(val_raw_acc, raw_batch_acc)
                add_accuracy_totals(val_aligned_acc, aligned_batch_acc)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_base_loss = val_base_loss / len(val_loader)
        avg_val_raw_rmse = val_raw_rmse / len(val_loader)
        avg_val_aligned_rmse = val_aligned_rmse / len(val_loader)
        avg_val_spot_loss = val_spot_loss / len(val_loader)
        avg_val_bond_loss = val_bond_loss / len(val_loader)
        avg_val_distmap_loss = val_distmap_loss / len(val_loader)
        avg_val_adj_loss = val_adj_loss / len(val_loader)
        avg_val_short_range_loss = val_short_range_loss / len(val_loader)
        avg_val_medium_range_loss = val_medium_range_loss / len(val_loader)
        avg_val_curvature_loss = val_curvature_loss / len(val_loader)
        avg_val_spread_loss = val_spread_loss / len(val_loader)
        avg_val_local_geometry_loss = val_local_geometry_loss / len(val_loader)
        avg_val_raw_acc = average_accuracy_totals(val_raw_acc, len(val_loader))
        avg_val_aligned_acc = average_accuracy_totals(val_aligned_acc, len(val_loader))
        avg_val_acc = avg_val_aligned_acc[main_threshold]
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Base Loss: {avg_val_base_loss:.4f}, "
            f"Raw RMSE: {avg_val_raw_rmse:.4f}, Aligned RMSE: {avg_val_aligned_rmse:.4f}, "
            f"SPOT Loss: {avg_val_spot_loss:.4f}, Bond Loss: {avg_val_bond_loss:.4f}, "
            f"DistMap Loss: {avg_val_distmap_loss:.4f}, Adj Loss: {avg_val_adj_loss:.4f}, "
            f"ShortRange Loss: {avg_val_short_range_loss:.4f}, "
            f"MediumRange Loss: {avg_val_medium_range_loss:.4f}, "
            f"Curvature Loss: {avg_val_curvature_loss:.4f}, "
            f"Spread Loss: {avg_val_spread_loss:.4f}, "
            f"Local Geometry Loss: {avg_val_local_geometry_loss:.4f}"
        )
        print(format_accuracy_metrics("Val Raw", avg_val_raw_acc))
        print(format_accuracy_metrics("Val Aligned", avg_val_aligned_acc))
        if use_local_geometry_checkpoint and avg_val_local_geometry_loss < best_local_geometry_loss:
            best_local_geometry_loss = avg_val_local_geometry_loss
            torch.save(model.state_dict(), best_local_model_path)
            print(f"Best local-geometry model saved to: {best_local_model_path}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print("Model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs.')
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        history['epoch'].append(epoch+1)
        history['train_loss'].append(avg_epoch_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history_row = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "val_loss": avg_val_loss,
            "train_base_loss": avg_epoch_base_loss,
            "val_base_loss": avg_val_base_loss,
            "train_raw_rmse": avg_epoch_raw_rmse,
            "val_raw_rmse": avg_val_raw_rmse,
            "train_aligned_rmse": avg_epoch_aligned_rmse,
            "val_aligned_rmse": avg_val_aligned_rmse,
            "train_spot_loss": avg_epoch_spot_loss,
            "val_spot_loss": avg_val_spot_loss,
            "train_bond_loss": avg_epoch_bond_loss,
            "val_bond_loss": avg_val_bond_loss,
            "train_distmap_loss": avg_epoch_distmap_loss,
            "val_distmap_loss": avg_val_distmap_loss,
            "train_adj_loss": avg_epoch_adj_loss,
            "val_adj_loss": avg_val_adj_loss,
            "train_short_range_loss": avg_epoch_short_range_loss,
            "val_short_range_loss": avg_val_short_range_loss,
            "train_medium_range_loss": avg_epoch_medium_range_loss,
            "val_medium_range_loss": avg_val_medium_range_loss,
            "train_curvature_loss": avg_epoch_curvature_loss,
            "val_curvature_loss": avg_val_curvature_loss,
            "train_spread_loss": avg_epoch_spread_loss,
            "val_spread_loss": avg_val_spread_loss,
            "train_local_geometry_loss": avg_epoch_local_geometry_loss,
            "val_local_geometry_loss": avg_val_local_geometry_loss,
            f"train_aligned_acc@{main_threshold:g}": avg_train_acc,
            f"val_aligned_acc@{main_threshold:g}": avg_val_acc,
        }
        for threshold in accuracy_thresholds:
            tag = f"{threshold:g}"
            history_row[f"train_raw_acc@{tag}"] = avg_train_raw_acc[threshold]
            history_row[f"train_aligned_acc@{tag}"] = avg_train_aligned_acc[threshold]
            history_row[f"val_raw_acc@{tag}"] = avg_val_raw_acc[threshold]
            history_row[f"val_aligned_acc@{tag}"] = avg_val_aligned_acc[threshold]
        history_rows.append(history_row)
    print("Training complete.")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last epoch model saved to: {last_model_path}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    history_path = os.path.join(output_dir, "training_history.csv")
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_validation_loss.png"))
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(
        history['epoch'],
        history['val_acc'],
        label=f'Validation Aligned Acc@{main_threshold:g}',
        marker='o',
    )
    plt.xlabel('Epoch')
    plt.ylabel(f'Accuracy @ {main_threshold:g}')
    plt.title(f'Validation Accuracy History (threshold={main_threshold:g})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"validation_accuracy_acc_at_{main_threshold_tag}.png"))
    plt.show()
    return best_val_loss,history



def parse_args():
    """Purpose: Parse command-line options for a training run.

    Input:
        Command-line arguments.
    Output:
        argparse Namespace with training, output, and loss settings.
    """
    parser = ArgumentParser(description="Train the RNA 3D folding model from scratch.")
    add_shared_training_args(parser, include_mode=True)
    return parser.parse_args()


def run_training_config(config, dry_run=False, expected_mode="train"):
    """Purpose: Run training or fine-tuning from an already-merged config.

    Input:
        config: Fully merged training configuration.
        dry_run: Whether to print resolved settings without starting training.
        expected_mode: Required run.mode value for the calling entrypoint.
    Output:
        None. Starts training unless dry_run is enabled.
    """
    if expected_mode is not None and config["run"].get("mode") != expected_mode:
        raise ValueError(
            f'This entrypoint expects run.mode="{expected_mode}", '
            f'but got "{config["run"].get("mode")}".'
        )

    output_dir = resolve_output_dir(config)
    save_path = config["run"].get("save_path") or os.path.join(output_dir, "best_model.pth")
    used_config = config_with_runtime_paths(config, output_dir, save_path)
    print_run_summary(used_config, output_dir)
    if dry_run:
        print("Dry run complete. No training was started.")
        return

    data_config = used_config["data"]
    training_config = used_config["training"]
    model_config = used_config["model"]
    loss_config = used_config["losses"]
    metrics_config = used_config["metrics"]

    train_seqs, val_seqs, train_labels, val_labels, train_pair_df, val_pair_df = load_data(
        data_dir=data_config["data_dir"],
        val_labels_path=data_config["val_labels_path"],
    )
    extra_data_dirs = normalize_path_list(data_config.get("extra_train_data_dirs"))
    if extra_data_dirs:
        extra_seqs, extra_labels, extra_pair_df = load_extra_train_sources(
            extra_data_dirs,
            pair_columns=train_pair_df.columns if train_pair_df is not None else None,
        )
        extra_seqs, extra_labels, extra_pair_df, extra_report = prepare_extra_training_data(
            extra_seqs,
            extra_labels,
            extra_pair_df,
            train_seqs,
            val_seqs,
            validation_pair_df=val_pair_df,
            max_extra_targets=data_config.get("max_extra_train_targets"),
            max_len=training_config["max_len"],
            min_valid_labels=data_config.get("min_train_valid_labels"),
            min_label_coverage=data_config.get("min_train_label_coverage"),
            selection=data_config.get("extra_train_selection", "quality_diverse"),
            seed=data_config.get("extra_train_seed", 13),
            length_bin_size=data_config.get("extra_train_length_bin_size", 50),
            exclude_base_sequence_overlap=data_config.get("exclude_base_sequence_overlap", True),
            exclude_validation_sequence_overlap=data_config.get("exclude_validation_sequence_overlap", True),
            contact_map_dir=data_config.get("contact_map_dir"),
            structure_reinforcement_extra_targets=data_config.get("structure_reinforcement_extra_targets", 0),
            structure_reinforcement_anchor_groups=data_config.get("structure_reinforcement_anchor_groups"),
            structure_reinforcement_targets_per_group=data_config.get(
                "structure_reinforcement_targets_per_group"
            ),
        )
        print_extra_training_report(extra_report)
        train_seqs, train_labels, train_pair_df = append_training_data(
            train_seqs,
            train_labels,
            train_pair_df,
            extra_seqs,
            extra_labels,
            extra_pair_df,
        )
    else:
        print_extra_training_report({"enabled": False})

    train_validate(
        train_seqs,
        train_labels,
        val_seqs,
        val_labels,
        data_config["msa_dir"],
        train_pair_df,
        val_pair_df,
        save_path=save_path,
        output_dir=output_dir,
        output_root=used_config["run"]["output_root"],
        init_model_path=used_config["run"].get("init_model_path"),
        epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
        Ir=training_config["lr"],
        max_len=training_config["max_len"],
        patience=training_config["patience"],
        weight_decay=training_config["weight_decay"],
        contact_map_dir=data_config["contact_map_dir"],
        spot_bias_scale=model_config["spot_bias_scale"],
        use_graph=model_config["use_graph"],
        graph_layers=model_config["graph_layers"],
        graph_scale=model_config["graph_scale"],
        spot_edge_threshold=model_config["spot_edge_threshold"],
        spot_top_k=model_config["spot_top_k"],
        local_edge_max_sep=model_config["local_edge_max_sep"],
        coord_refine_steps=model_config["coord_refine_steps"],
        coord_refine_hidden=model_config["coord_refine_hidden"],
        coord_refine_dropout=model_config["coord_refine_dropout"],
        coord_refine_local_window=model_config["coord_refine_local_window"],
        coord_refine_delta_scale=model_config["coord_refine_delta_scale"],
        spot_loss_weight=loss_config["spot_loss_weight"],
        raw_loss_weight=loss_config["raw_loss_weight"],
        aligned_loss_weight=loss_config["aligned_loss_weight"],
        bond_loss_weight=loss_config["bond_loss_weight"],
        distmap_loss_weight=loss_config["distmap_loss_weight"],
        adj_loss_weight=loss_config["adj_loss_weight"],
        short_range_loss_weight=loss_config["short_range_loss_weight"],
        medium_range_loss_weight=loss_config["medium_range_loss_weight"],
        curvature_loss_weight=loss_config["curvature_loss_weight"],
        spread_loss_weight=loss_config["spread_loss_weight"],
        bond_lower=loss_config["bond_lower"],
        bond_upper=loss_config["bond_upper"],
        short_range_max_sep=loss_config["short_range_max_sep"],
        medium_range_min_sep=loss_config["medium_range_min_sep"],
        medium_range_max_sep=loss_config["medium_range_max_sep"],
        accuracy_thresholds=metrics_config["accuracy_thresholds"],
        min_train_valid_labels=data_config.get("min_train_valid_labels"),
        min_train_label_coverage=data_config.get("min_train_label_coverage"),
        max_train_targets=data_config.get("max_train_targets"),
        train_selection=data_config.get("train_selection", "quality_diverse"),
        train_seed=data_config.get("train_seed", 13),
        train_length_bin_size=data_config.get("train_length_bin_size", 50),
        train_structure_anchor_groups=data_config.get("train_structure_anchor_groups"),
        train_targets_per_group=data_config.get("train_targets_per_group", 150),
        train_similarity_top_k_per_group=data_config.get("train_similarity_top_k_per_group", 300),
        used_config=used_config,
    )


def main():
    args = parse_args()
    cli_overrides = cli_overrides_from_args(args)
    config = build_config(args.config, cli_overrides)
    run_training_config(config, dry_run=args.dry_run, expected_mode="train")

#python "train and validate.py" --config configs/train_default.yaml


if __name__ == "__main__":
    main()
