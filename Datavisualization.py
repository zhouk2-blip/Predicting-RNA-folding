import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
from DATASET import RNADataset as RNA
from model_conv_attn import RNAmodel
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from kabsch import kabsch_align_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "dataset"
OUTPUT_ROOT = "output"
MSA_DIR = os.path.join(DATA_DIR, "MSA")
VAL_LABELS_PATH = "validation_labels_new.normalized.csv"
RAW_VAL_LABELS_PATH = os.path.join(DATA_DIR, "validation_labels.csv")
RUN_DIR_RE = re.compile(r"^output(\d+)$")

def latest_output_dir(output_root=OUTPUT_ROOT):
    """Purpose: Find the highest-numbered run directory.

    Input:
        output_root: Parent directory containing output1, output2, ...
    Output:
        Path to the latest run directory, or None if no run exists.
    """
    if not os.path.exists(output_root):
        return None

    runs = []
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        match = RUN_DIR_RE.match(name)
        if match and os.path.isdir(path):
            runs.append((int(match.group(1)), path))

    if not runs:
        return None

    runs.sort()
    return runs[-1][1]

def find_model_path(run_dir):
    """Purpose: Select a checkpoint from a run directory.

    Input:
        run_dir: Directory containing one or more .pth checkpoints.
    Output:
        Preferred checkpoint path, falling back to the newest .pth file.
    """
    preferred = ["best_model.pth", "best_model2.pth", "best_model64.pth"]
    for name in preferred:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path

    candidates = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.endswith(".pth")
    ]
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoint found in {run_dir}")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def find_latest_model_path(output_root=OUTPUT_ROOT):
    """Purpose: Find the newest checkpoint across numbered run directories.

    Input:
        output_root: Parent directory containing output1, output2, ...
    Output:
        Checkpoint path from the newest run directory that has a .pth file.
    """
    if not os.path.exists(output_root):
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    runs = []
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        match = RUN_DIR_RE.match(name)
        if match and os.path.isdir(path):
            runs.append((int(match.group(1)), path))

    for _, run_dir in sorted(runs, reverse=True):
        try:
            return find_model_path(run_dir)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"No .pth checkpoint found in {output_root}/outputN directories")

def load_model(model_path):
    """Purpose: Load a checkpoint and reconstruct the matching RNA model.

    Input:
        model_path: Path to a saved PyTorch state_dict.
    Output:
        Tuple of (model, input channel count, checkpoint max length).
    """
    state = torch.load(model_path, map_location=DEVICE)
    input_channels = state["conv_block.0.weight"].shape[1]
    max_len = state["pos_embed.weight"].shape[0]
    model = RNAmodel(input_channels=input_channels, max_len=max_len).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, input_channels, max_len

def masked_rmse(pred, target, mask, eps=1e-8):
    """Purpose: Compute masked RMSE for diagnostic reporting.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        eps: Small value to keep the square root stable.
    Output:
        Scalar RMSE tensor.
    """
    mask = mask.float().unsqueeze(-1)
    return torch.sqrt((((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0) + eps)

def accuracy_at_threshold(pred, target, mask, threshold=1.0):
    """Purpose: Compute residue-level accuracy at one distance threshold.

    Input:
        pred: Predicted coordinates shaped (B, L, 3).
        target: Ground-truth coordinates shaped (B, L, 3).
        mask: Valid-residue mask shaped (B, L).
        threshold: Distance cutoff for a correct residue.
    Output:
        Float accuracy over valid residues.
    """
    distances = torch.norm(pred - target, dim=-1)
    mask = mask.float()
    total = mask.sum().clamp(min=1.0)
    return (((distances < threshold) * mask).sum() / total).item()

def coordinate_rows(ids, coords, lengths):
    """Purpose: Convert batched coordinates into a submission-style DataFrame rows list.

    Input:
        ids: Target IDs for the batch.
        coords: Coordinate array shaped (B, L, 3).
        lengths: True sequence lengths for each target.
    Output:
        List of dictionaries with ID, x_1, y_1, and z_1 fields.
    """
    rows = []
    for b, target_id in enumerate(ids):
        L = lengths[b]
        xyz = coords[b, :L, :]
        for i in range(L):
            rows.append({
                "ID": f"{target_id}_{i+1}",
                "x_1": xyz[i, 0],
                "y_1": xyz[i, 1],
                "z_1": xyz[i, 2],
            })
    return rows

def label_normalization_stats(label_df, target_id, L):
    """Purpose: Recover the mean and std used to normalize one validation target.

    Input:
        label_df: Raw coordinate label DataFrame in Angstrom units.
        target_id: RNA target identifier.
        L: Target sequence length after filtering/truncation.
    Output:
        Tuple of (mean, std) arrays shaped (1, 3), or (None, None) if unavailable.
    """
    rows = label_df[label_df["ID"].str.startswith(target_id + "_")]
    coords = np.full((L, 3), np.nan, dtype=np.float32)
    for _, row in rows.iterrows():
        resid = int(row["resid"])
        if 1 <= resid <= L:
            coords[resid - 1, 0] = row["x_1"]
            coords[resid - 1, 1] = row["y_1"]
            coords[resid - 1, 2] = row["z_1"]

    coords[coords <= -1e17] = np.nan
    valid = ~np.isnan(coords).any(axis=1)
    if not valid.any():
        return None, None

    mean = coords[valid].mean(axis=0, keepdims=True)
    std = coords[valid].std(axis=0, keepdims=True) + 1e-6
    return mean, std

def denormalize_coords(coords, mean, std):
    """Purpose: Convert normalized coordinates back to Angstrom units.

    Input:
        coords: Normalized coordinates shaped (L, 3).
        mean: Per-target coordinate mean shaped (1, 3).
        std: Per-target coordinate std shaped (1, 3).
    Output:
        Denormalized coordinates shaped (L, 3).
    """
    return coords * std + mean

def plot_structure(pred_xyz, true_xyz, true_valid, title, path):
    """Purpose: Save a 3D plot comparing predicted and true structures.

    Input:
        pred_xyz: Predicted coordinates shaped (L, 3).
        true_xyz: Ground-truth coordinates shaped (L, 3).
        true_valid: Boolean mask for valid true coordinates.
        title: Plot title.
        path: Output PNG path.
    Output:
        None. Writes a PNG file.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2],
               c="red", s=20, label="Predicted")
    ax.scatter(true_xyz[true_valid,0], true_xyz[true_valid,1], true_xyz[true_valid,2],
               c="blue", s=20, label="True")

    for i in range(len(pred_xyz) - 1):
        ax.plot([pred_xyz[i,0], pred_xyz[i+1,0]],
                [pred_xyz[i,1], pred_xyz[i+1,1]],
                [pred_xyz[i,2], pred_xyz[i+1,2]], "r-", alpha=0.3)
        if true_valid[i] and true_valid[i + 1]:
            ax.plot([true_xyz[i,0], true_xyz[i+1,0]],
                    [true_xyz[i,1], true_xyz[i+1,1]],
                    [true_xyz[i,2], true_xyz[i+1,2]], "b-", alpha=0.3)

    ax.set_title(title)
    ax.legend()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_prediction_only(pred_xyz, title, path):
    """Purpose: Save a 3D plot of the predicted structure only.

    Input:
        pred_xyz: Predicted coordinates shaped (L, 3).
        title: Plot title.
        path: Output PNG path.
    Output:
        None. Writes a PNG file.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111,projection="3d")
    ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2],
               c="red", s=20, label="Predicted")
    for i in range(len(pred_xyz) - 1):
        ax.plot([pred_xyz[i,0], pred_xyz[i+1,0]],
                [pred_xyz[i,1], pred_xyz[i+1,1]],
                [pred_xyz[i,2], pred_xyz[i+1,2]], "r-", alpha=0.3)
    ax.set_title(title)
    ax.legend()
    plt.savefig(path, dpi=150)
    plt.close()

def compactness_row(target_id, pred, aligned_pred, labels, mask, L):
    """Purpose: Compute one target's compactness and alignment diagnostics.

    Input:
        target_id: RNA target identifier.
        pred: Raw predicted coordinates shaped (L, 3).
        aligned_pred: Kabsch-aligned predicted coordinates shaped (L, 3).
        labels: Ground-truth coordinates shaped (L, 3).
        mask: Valid-residue mask shaped (L,).
        L: True sequence length.
    Output:
        Dictionary of compactness, RMSE, and accuracy diagnostics.
    """
    local_mask = mask[:L].bool()
    valid_pred = pred[:L][local_mask]
    valid_true = labels[:L][local_mask]
    if valid_pred.shape[0] > 1:
        pred_std_mean = valid_pred.std(dim=0).mean().item()
        true_std_mean = valid_true.std(dim=0).mean().item()
    else:
        pred_std_mean = 0.0
        true_std_mean = 0.0

    adjacent_mask = (mask[1:L] * mask[:L-1]).bool() if L > 1 else mask.new_zeros(0, dtype=torch.bool)
    if adjacent_mask.numel() > 0 and adjacent_mask.any():
        pred_adj = torch.norm(pred[1:L] - pred[:L-1], dim=-1)[adjacent_mask]
        true_adj = torch.norm(labels[1:L] - labels[:L-1], dim=-1)[adjacent_mask]
        pred_adj_median = pred_adj.median().item()
        true_adj_median = true_adj.median().item()
    else:
        pred_adj_median = np.nan
        true_adj_median = np.nan

    batch_mask = mask[:L].unsqueeze(0)
    return {
        "target_id": target_id,
        "length": L,
        "valid_residues": int(local_mask.sum().item()),
        "pred_std_mean": pred_std_mean,
        "true_std_mean": true_std_mean,
        "pred_adj_median": pred_adj_median,
        "true_adj_median": true_adj_median,
        "raw_rmse": masked_rmse(pred[:L].unsqueeze(0), labels[:L].unsqueeze(0), batch_mask).item(),
        "aligned_rmse": masked_rmse(aligned_pred[:L].unsqueeze(0), labels[:L].unsqueeze(0), batch_mask).item(),
        "raw_acc@1.0": accuracy_at_threshold(pred[:L].unsqueeze(0), labels[:L].unsqueeze(0), batch_mask, threshold=1.0),
        "aligned_acc@1.0": accuracy_at_threshold(aligned_pred[:L].unsqueeze(0), labels[:L].unsqueeze(0), batch_mask, threshold=1.0),
    }

def analyze_model_performance(model, val_loader, raw_label_df=None, out_dir=None, save_aligned=True):
    """Purpose: Generate validation predictions, plots, and diagnostic rows.

    Input:
        model: Trained RNAmodel in evaluation mode.
        val_loader: Validation DataLoader.
        raw_label_df: Optional raw Angstrom validation labels for denormalized plots.
        out_dir: Directory for 3D plot PNG files.
        save_aligned: Whether to save aligned predictions.
    Output:
        Tuple of raw prediction DataFrame, aligned prediction DataFrame, diagnostics DataFrame.
    """
    if out_dir is None:
        out_dir = "3d_plots"
    rows = []
    aligned_rows = []
    aligned_angstrom_rows = []
    diagnostic_rows = []
    os.makedirs(out_dir, exist_ok=True)
    
    model.eval()

    print("\nGenerating 3D comparison plots...")

    for batch_idx, (feats, labels, contact_map, mask, lengths, ids) in enumerate(
        tqdm(val_loader, desc="Analyzing")
    ):
        feats = feats.to(DEVICE)
        labels = labels.to(DEVICE)
        contact_map = contact_map.to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.no_grad():
            preds = model(feats, mask, contact_map=contact_map)              # (B, L, 3)
            aligned_preds = kabsch_align_batch(preds, labels, mask)

        diagnostic_rows.extend(
            compactness_row(ids[b], preds[b], aligned_preds[b], labels[b], mask[b], lengths[b])
            for b in range(len(ids))
        )

        preds_np = preds.cpu().numpy()
        aligned_np = aligned_preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        valid_mask = mask.cpu().numpy().astype(bool)
        rows.extend(coordinate_rows(ids, preds_np, lengths))
        if save_aligned:
            aligned_rows.extend(coordinate_rows(ids, aligned_np, lengths))

        B = preds_np.shape[0]

        for b in range(B):
            target_id = ids[b]                     # e.g., "R1107"
            L = lengths[b]                         # true length

            pred_xyz = preds_np[b, :L, :]
            aligned_xyz = aligned_np[b, :L, :]
            true_xyz = labels_np[b, :L, :]
            true_valid = valid_mask[b, :L]

            plot_structure(pred_xyz, true_xyz, true_valid, f"{target_id}: Raw Predicted vs True",
                           os.path.join(out_dir, f"{target_id}.png"))
            if save_aligned:
                plot_structure(aligned_xyz, true_xyz, true_valid, f"{target_id}: Aligned Predicted vs True",
                               os.path.join(out_dir, f"{target_id}_aligned.png"))
            plot_prediction_only(pred_xyz, f"{target_id}: Raw Predicted Structure",
                                 os.path.join(out_dir, f"{target_id}raw.png"))
            if raw_label_df is not None and save_aligned:
                mean, std = label_normalization_stats(raw_label_df, target_id, L)
                if mean is not None:
                    aligned_angstrom = denormalize_coords(aligned_xyz, mean, std)
                    true_angstrom = denormalize_coords(true_xyz, mean, std)
                    aligned_angstrom_rows.extend(
                        coordinate_rows([target_id], aligned_angstrom[np.newaxis, :, :], [L])
                    )
                    plot_structure(
                        aligned_angstrom,
                        true_angstrom,
                        true_valid,
                        f"{target_id}: Aligned Predicted vs True (Angstrom)",
                        os.path.join(out_dir, f"{target_id}_aligned_angstrom.png"),
                    )
    print("3D comparison plots saved in:", out_dir)
    print("3D predicted structure plots saved in:", out_dir)

    diagnostics = pd.DataFrame(diagnostic_rows)
    if not diagnostics.empty:
        print("\nCompactness diagnostics summary:")
        print(diagnostics[[
            "pred_std_mean",
            "true_std_mean",
            "pred_adj_median",
            "true_adj_median",
            "raw_rmse",
            "aligned_rmse",
            "raw_acc@1.0",
            "aligned_acc@1.0",
        ]].mean(numeric_only=True).to_string())

    return pd.DataFrame(rows), pd.DataFrame(aligned_rows), pd.DataFrame(aligned_angstrom_rows), diagnostics
def load_data():
    """Purpose: Load validation data needed for visualization.

    Input:
        None. Uses DATA_DIR and VAL_LABELS_PATH constants.
    Output:
        DataFrames for validation sequences, labels, raw labels, and pair features.
    """
    print("Loading data...")
    val_seqs = pd.read_csv(os.path.join(DATA_DIR, "validation_sequences.csv"))
    val_label_path = VAL_LABELS_PATH if os.path.exists(VAL_LABELS_PATH) else os.path.join(DATA_DIR, "validation_labels.csv")
    val_labels = pd.read_csv(val_label_path)
    raw_val_labels = pd.read_csv(RAW_VAL_LABELS_PATH) if os.path.exists(RAW_VAL_LABELS_PATH) else val_labels
    val_pair_df = pd.read_csv(os.path.join(DATA_DIR, "validation_pair_features.csv"))
    return val_seqs, val_labels, raw_val_labels, val_pair_df

def parse_args():
    """Purpose: Parse command-line options for visualization outputs.

    Input:
        Command-line arguments.
    Output:
        argparse Namespace with checkpoint and output path settings.
    """
    parser = ArgumentParser(description="Generate validation predictions and 3D plots.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--aligned-predictions-path", default=None)
    parser.add_argument("--aligned-angstrom-predictions-path", default=None)
    parser.add_argument("--diagnostics-path", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--contact-map-dir", default=os.path.join(DATA_DIR, "spot_maps"))
    parser.add_argument("--save-aligned", action="store_true", default=True)
    return parser.parse_args()


def main():

    args = parse_args()
    if args.model_path is None:
        args.model_path = find_latest_model_path(args.output_root)
        run_dir = os.path.dirname(args.model_path)
    else:
        run_dir = os.path.dirname(args.model_path) or "."

    output_dir = args.output_dir or run_dir
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = args.predictions_path or os.path.join(output_dir, "Predictions.csv")
    aligned_predictions_path = args.aligned_predictions_path or os.path.join(output_dir, "Predictions_aligned.csv")
    aligned_angstrom_predictions_path = (
        args.aligned_angstrom_predictions_path
        or os.path.join(output_dir, "Predictions_aligned_angstrom.csv")
    )
    diagnostics_path = args.diagnostics_path or os.path.join(output_dir, "compactness_diagnostics.csv")
    plots_dir = args.plots_dir or os.path.join(output_dir, "3d_plots")
    print(f"Using model checkpoint: {args.model_path}")
    print(f"Writing visualization outputs to: {output_dir}")
    val_seqs, val_labels, raw_val_labels, val_pair_df = load_data()
    model, input_channels, model_max_len = load_model(args.model_path)
    pair_df = val_pair_df if input_channels == 8 else None
    original_val_count = len(val_seqs)
    val_seqs = val_seqs[val_seqs["sequence"].str.len() <= model_max_len].copy()
    print(
        f"Filtered validation sequences by model max_len={model_max_len}: "
        f"kept {len(val_seqs)}/{original_val_count}, skipped {original_val_count - len(val_seqs)}."
    )
    val_loader = DataLoader(
        RNA(val_seqs, val_labels, MSA_DIR, pair_df=pair_df, max_len=model_max_len, contact_map_dir=args.contact_map_dir),
        batch_size=4,
        shuffle=False,
        collate_fn=RNA.collate_fn,
    )
    if args.contact_map_dir is not None:
        print(f"Using SPOT contact maps from: {args.contact_map_dir}")
    submission_df, aligned_df, aligned_angstrom_df, diagnostics_df = analyze_model_performance(
        model,
        val_loader,
        raw_label_df=raw_val_labels,
        out_dir=plots_dir,
        save_aligned=args.save_aligned,
    )
    submission_df.to_csv(predictions_path, index=False)
    if args.save_aligned:
        aligned_df.to_csv(aligned_predictions_path, index=False)
        aligned_angstrom_df.to_csv(aligned_angstrom_predictions_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)
if __name__ == "__main__":
    main()

#python Datavisualization.py --model-path output/output3/best_model.pth
#If --model-path is omitted, the latest output/outputN checkpoint is used.
#By default, predictions, diagnostics, and plots are saved next to the checkpoint.
