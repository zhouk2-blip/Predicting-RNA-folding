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
from kabsch import kabsch_align_batch
from argparse import ArgumentParser
EPOCHS = 40
BATCH_SIZE = 32
IR = 3e-4
MAX_LEN = 256
PATIENCE = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "dataset"
OUTPUT_ROOT = "output"
MSA_DIR = os.path.join(DATA_DIR, "MSA")
VAL_LABELS_PATH = "validation_labels_new.normalized.csv"

RUN_DIR_RE = re.compile(r"^output(\d+)$")

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

def load_data():
    """Purpose: Load sequence, label, and pair-feature CSV files.

    Input:
        None. Uses DATA_DIR and VAL_LABELS_PATH constants.
    Output:
        DataFrames for train/validation sequences, labels, and pair features.
    """
    print("Loading data...")
    train_seqs = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
    val_seqs = pd.read_csv(os.path.join(DATA_DIR, "validation_sequences.csv"))
    train_labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    val_label_path = VAL_LABELS_PATH if os.path.exists(VAL_LABELS_PATH) else os.path.join(DATA_DIR, "validation_labels.csv")
    val_labels = pd.read_csv(val_label_path)
    train_pair_df = pd.read_csv(os.path.join(DATA_DIR, "train_pair_features.csv"))
    val_pair_df = pd.read_csv(os.path.join(DATA_DIR, "validation_pair_features.csv"))
    return train_seqs, val_seqs, train_labels, val_labels, train_pair_df, val_pair_df

def load_model(model_path):
    """Purpose: Load a saved model checkpoint for inference.

    Input:
        model_path: Path to a PyTorch state_dict checkpoint.
    Output:
        RNAmodel with checkpoint weights loaded and eval mode enabled.
    """
    state = torch.load(model_path, map_location=DEVICE)
    input_channels = state["conv_block.0.weight"].shape[1]
    max_len = state["pos_embed.weight"].shape[0]
    model = RNAmodel(input_channels=input_channels, max_len=max_len).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model



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
        Tuple of (weighted loss tensor, aligned prediction tensor).
    """
    aligned_pred = kabsch_align_batch(pred, target, mask)

    raw_rmse = masked_rmse_loss(pred, target, mask)
    aligned_rmse = masked_rmse_loss(aligned_pred, target, mask)

    loss = raw_weight * raw_rmse + aligned_weight * aligned_rmse

    return loss, aligned_pred

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
    bond_lower,
    bond_upper,
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
        bond_lower: Lower adjacent bond range.
        bond_upper: Upper adjacent bond range.
    Output:
        Dictionary with total loss, individual loss terms, and aligned outputs.
    """
    base_loss, aligned_outputs = kabsch_rmse_loss(
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
    total_loss = (
        base_loss
        + spot_loss_weight * spot_loss
        + bond_loss_weight * bond_loss
        + distmap_loss_weight * distmap_loss
    )
    return {
        "total": total_loss,
        "base": base_loss,
        "spot": spot_loss,
        "bond": bond_loss,
        "distmap": distmap_loss,
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
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    Ir = IR,
    max_len = MAX_LEN,
    patience = PATIENCE,
    contact_map_dir = None,
    spot_bias_scale = 1.0,
    spot_loss_weight = 0.10,
    raw_loss_weight = 0.05,
    aligned_loss_weight = 1.0,
    bond_loss_weight = 0.30,
    distmap_loss_weight = 0.30,
    bond_lower = 0.30,
    bond_upper = 1.20,
    accuracy_thresholds = None,
):
    """Purpose: Train the RNA 3D model and save run-specific outputs.

    Input:
        train, label, val, val_label: Training/validation DataFrames.
        msa_dir: Directory containing MSA FASTA files.
        train_pair_df, val_pair_df: ViennaRNA pair-feature DataFrames.
        save_path: Optional explicit best-checkpoint path.
        output_dir: Optional explicit run directory.
        output_root: Parent directory for numbered output runs.
        epochs, batch_size, Ir, max_len, patience: Training hyperparameters.
        contact_map_dir: Optional SPOT-RNA-2D contact-map directory.
        loss weights and bond bounds: Hyperparameters for the hybrid loss.
        accuracy_thresholds: Distance thresholds for raw/aligned accuracy.
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
    history  = {'epoch':[],'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    history_rows = []
    train_dataset = RNA(train,label,msa_dir,pair_df=train_pair_df,max_len=max_len,contact_map_dir=contact_map_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle = True,
        collate_fn=RNA.collate_fn, 
        )
    val_dataset = RNA(val, val_label, msa_dir, pair_df=val_pair_df, max_len=max_len,contact_map_dir=contact_map_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RNA.collate_fn,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNAmodel(max_len=max_len, spot_bias_scale=spot_bias_scale).to(device)
    print("Training from scratch with 8 input channels, including pair features.")
    if contact_map_dir is not None:
        print(f"Using SPOT contact maps from: {contact_map_dir}")
        print(f"SPOT attention bias scale: {spot_bias_scale}, SPOT loss weight: {spot_loss_weight}")
    print(f"Run output directory: {output_dir}")
    print(f"Best checkpoint path: {save_path}")
    print(f"Last epoch checkpoint path: {last_model_path}")
    print(f"Kabsch RMSE weights: raw={raw_loss_weight}, aligned={aligned_loss_weight}")
    print(f"Adjacent bond range loss: weight={bond_loss_weight}, lower={bond_lower}, upper={bond_upper}")
    print(f"Full distance matrix loss weight: {distmap_loss_weight}")
    print("Accuracy thresholds:", ", ".join(f"{t:g}" for t in accuracy_thresholds))
    print(f"Validation accuracy plot threshold: {main_threshold:g}")
    optimizer = optim.Adam(model.parameters(), lr=Ir, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience_counter = 0
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_base_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_spot_loss = 0.0
        epoch_bond_loss = 0.0
        epoch_distmap_loss = 0.0
        train_raw_acc = zero_accuracy_totals(accuracy_thresholds)
        train_aligned_acc = zero_accuracy_totals(accuracy_thresholds)
        for feats,labels,contact_map,mask,lengths,ids in progress:
            
            feats = feats.to(device)
            labels = labels.to(device)
            contact_map = contact_map.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            model_contact_map = contact_map if contact_map_dir is not None else None
            outputs = model(feats,mask,contact_map=model_contact_map)
            losses = compute_loss_bundle(
                outputs,
                labels,
                mask,
                contact_map,
                use_contact_map=contact_map_dir is not None,
                raw_loss_weight=raw_loss_weight,
                aligned_loss_weight=aligned_loss_weight,
                spot_loss_weight=spot_loss_weight,
                bond_loss_weight=bond_loss_weight,
                distmap_loss_weight=distmap_loss_weight,
                bond_lower=bond_lower,
                bond_upper=bond_upper,
            )
            loss = losses["total"]
            aligned_outputs = losses["aligned_outputs"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_base_loss += losses["base"].item()
            epoch_spot_loss += losses["spot"].item()
            epoch_bond_loss += losses["bond"].item()
            epoch_distmap_loss += losses["distmap"].item()
            raw_batch_acc = compute_accuracy_at_thresholds(outputs, labels, mask, accuracy_thresholds)
            aligned_batch_acc = compute_accuracy_at_thresholds(aligned_outputs, labels, mask, accuracy_thresholds)
            add_accuracy_totals(train_raw_acc, raw_batch_acc)
            add_accuracy_totals(train_aligned_acc, aligned_batch_acc)
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_base_loss = epoch_base_loss / len(train_loader)
        avg_epoch_spot_loss = epoch_spot_loss / len(train_loader)
        avg_epoch_bond_loss = epoch_bond_loss / len(train_loader)
        avg_epoch_distmap_loss = epoch_distmap_loss / len(train_loader)
        avg_train_raw_acc = average_accuracy_totals(train_raw_acc, len(train_loader))
        avg_train_aligned_acc = average_accuracy_totals(train_aligned_acc, len(train_loader))
        avg_train_acc = avg_train_aligned_acc[main_threshold]
        print(
            f"Epoch {epoch+1} Train Loss: {avg_epoch_loss:.4f}, "
            f"Base Loss: {avg_epoch_base_loss:.4f}, SPOT Loss: {avg_epoch_spot_loss:.4f}, "
            f"Bond Loss: {avg_epoch_bond_loss:.4f}, DistMap Loss: {avg_epoch_distmap_loss:.4f}"
        )
        print(format_accuracy_metrics("Train Raw", avg_train_raw_acc))
        print(format_accuracy_metrics("Train Aligned", avg_train_aligned_acc))
        
        model.eval()
        val_loss = 0.0
        val_base_loss = 0.0
        val_spot_loss = 0.0
        val_bond_loss = 0.0
        val_distmap_loss = 0.0
        val_raw_acc = zero_accuracy_totals(accuracy_thresholds)
        val_aligned_acc = zero_accuracy_totals(accuracy_thresholds)
        with torch.no_grad():
            for feats,labels,contact_map,mask,lengths,ids in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                contact_map = contact_map.to(device)
                mask = mask.to(device)
                model_contact_map = contact_map if contact_map_dir is not None else None
                outputs = model(feats,mask,contact_map=model_contact_map)
                losses = compute_loss_bundle(
                    outputs,
                    labels,
                    mask,
                    contact_map,
                    use_contact_map=contact_map_dir is not None,
                    raw_loss_weight=raw_loss_weight,
                    aligned_loss_weight=aligned_loss_weight,
                    spot_loss_weight=spot_loss_weight,
                    bond_loss_weight=bond_loss_weight,
                    distmap_loss_weight=distmap_loss_weight,
                    bond_lower=bond_lower,
                    bond_upper=bond_upper,
                )
                loss = losses["total"]
                aligned_outputs = losses["aligned_outputs"]
                val_loss += loss.item()
                val_base_loss += losses["base"].item()
                val_spot_loss += losses["spot"].item()
                val_bond_loss += losses["bond"].item()
                val_distmap_loss += losses["distmap"].item()
                raw_batch_acc = compute_accuracy_at_thresholds(outputs, labels, mask, accuracy_thresholds)
                aligned_batch_acc = compute_accuracy_at_thresholds(aligned_outputs, labels, mask, accuracy_thresholds)
                add_accuracy_totals(val_raw_acc, raw_batch_acc)
                add_accuracy_totals(val_aligned_acc, aligned_batch_acc)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_base_loss = val_base_loss / len(val_loader)
        avg_val_spot_loss = val_spot_loss / len(val_loader)
        avg_val_bond_loss = val_bond_loss / len(val_loader)
        avg_val_distmap_loss = val_distmap_loss / len(val_loader)
        avg_val_raw_acc = average_accuracy_totals(val_raw_acc, len(val_loader))
        avg_val_aligned_acc = average_accuracy_totals(val_aligned_acc, len(val_loader))
        avg_val_acc = avg_val_aligned_acc[main_threshold]
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Base Loss: {avg_val_base_loss:.4f}, "
            f"SPOT Loss: {avg_val_spot_loss:.4f}, Bond Loss: {avg_val_bond_loss:.4f}, "
            f"DistMap Loss: {avg_val_distmap_loss:.4f}"
        )
        print(format_accuracy_metrics("Val Raw", avg_val_raw_acc))
        print(format_accuracy_metrics("Val Aligned", avg_val_aligned_acc))
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
            "train_spot_loss": avg_epoch_spot_loss,
            "val_spot_loss": avg_val_spot_loss,
            "train_bond_loss": avg_epoch_bond_loss,
            "val_bond_loss": avg_val_bond_loss,
            "train_distmap_loss": avg_epoch_distmap_loss,
            "val_distmap_loss": avg_val_distmap_loss,
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
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=IR)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--contact-map-dir", default=None)
    parser.add_argument("--spot-bias-scale", type=float, default=1.0)
    parser.add_argument("--spot-loss-weight", type=float, default=0.10)
    parser.add_argument("--raw-loss-weight", type=float, default=0.05)
    parser.add_argument("--aligned-loss-weight", type=float, default=1.0)
    parser.add_argument("--bond-loss-weight", type=float, default=0.30)
    parser.add_argument("--distmap-loss-weight", type=float, default=0.30)
    parser.add_argument("--bond-lower", type=float, default=0.30)
    parser.add_argument("--bond-upper", type=float, default=1.20)
    parser.add_argument("--accuracy-thresholds", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    return parser.parse_args()


def main():
    args = parse_args()
    train_seqs, val_seqs, train_labels, val_labels, train_pair_df, val_pair_df = load_data()
    msa_dir = MSA_DIR
    train_validate(
        train_seqs,
        train_labels,
        val_seqs,
        val_labels,
        msa_dir,
        train_pair_df,
        val_pair_df,
        save_path=args.save_path,
        output_dir=args.output_dir,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        Ir=args.lr,
        max_len=args.max_len,
        patience=args.patience,
        contact_map_dir=args.contact_map_dir,
        spot_bias_scale=args.spot_bias_scale,
        spot_loss_weight=args.spot_loss_weight,
        raw_loss_weight=args.raw_loss_weight,
        aligned_loss_weight=args.aligned_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        distmap_loss_weight=args.distmap_loss_weight,
        bond_lower=args.bond_lower,
        bond_upper=args.bond_upper,
        accuracy_thresholds=args.accuracy_thresholds,
    )

#python "train and validate.py" --save-path output/model_pair_feature_run1.pth
#python "train and validate.py" --epochs 40 --batch-size 32 --lr 0.0003 --max-len 256 --patience 7 --contact-map-dir dataset/spot_maps --raw-loss-weight 0.05 --aligned-loss-weight 1.0 --bond-loss-weight 0.30 --spot-loss-weight 0.10 --distmap-loss-weight 0.30 --accuracy-thresholds 0.5 0.75 1.0
#--epochs      training epochs, default 40
#--batch-size  batch size, default 32
#--lr          learning rate, default 0.0003
#--max-len     maximum length of each RNA sequence, default 256
#--patience    patience for early stopping, default 7
#--output-root directory containing numbered runs, default output
#--output-dir  optional explicit run directory; otherwise creates output/outputN
#--save-path   optional checkpoint path; otherwise saves best_model.pth in the run directory

if __name__ == "__main__":
    main()
