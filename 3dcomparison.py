from DATASET import RNADataset as RNA
from model_conv_attn import RNAmodel
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = RNAmodel().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def analyze_model_performance(model, val_loader, out_dir="./analysis"):
    rows = []
    os.makedirs(out_dir, exist_ok=True)
    
    model.eval()

    print("\nGenerating 3D comparison plots...")

    for batch_idx, (feats, labels, mask, lengths, ids) in enumerate(
        tqdm(val_loader, desc="Analyzing")
    ):
        feats = feats.to(DEVICE)
        labels = labels.to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.no_grad():
            preds = model(feats, mask)              # (B, L, 3)

        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        B = preds.shape[0]

        for b in range(B):
            target_id = ids[b]                     # e.g., "R1107"
            L = lengths[b]                         # true length

            pred_xyz = preds[b, :L, :]
            true_xyz = labels[b, :L, :]
            target_id = ids[b]                     # e.g., "R1107"
            L = lengths[b]                         # true length
            pred_xyz = preds[b, :L, :]             # (L, 3)
            for i in range(L):
                row = {
                    "ID": f"{target_id}_{i+1}",
                    "x_1": pred_xyz[i,0],
                    "y_1": pred_xyz[i,1],
                    "z_1": pred_xyz[i,2],
                }
                rows.append(row)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # scatter
            ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2],
                       c="red", s=20, label="Predicted")
            ax.scatter(true_xyz[:,0], true_xyz[:,1], true_xyz[:,2],
                       c="blue", s=20, label="True")

            # backbone lines   
            for i in range(L - 1):
                ax.plot([pred_xyz[i,0], pred_xyz[i+1,0]],
                        [pred_xyz[i,1], pred_xyz[i+1,1]],
                        [pred_xyz[i,2], pred_xyz[i+1,2]], "r-", alpha=0.3)

                ax.plot([true_xyz[i,0], true_xyz[i+1,0]],
                        [true_xyz[i,1], true_xyz[i+1,1]],
                        [true_xyz[i,2], true_xyz[i+1,2]], "b-", alpha=0.3)

            ax.set_title(f"{target_id}: Predicted vs True")
            ax.legend()
            
            plt.savefig(os.path.join(out_dir, f"{target_id}.png"), dpi=150)
            plt.close()
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111,projection="3d")
            ax2.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2],
                       c="red", s=20, label="Predicted")
            for i in range(L - 1):
                ax2.plot([pred_xyz[i,0], pred_xyz[i+1,0]],
                        [pred_xyz[i,1], pred_xyz[i+1,1]],
                        [pred_xyz[i,2], pred_xyz[i+1,2]], "r-", alpha=0.3)
            ax2.set_title(f"{target_id}: Predicted Structure")
            ax2.legend()
            plt.savefig(os.path.join(out_dir, f"{target_id}raw.png"), dpi=150)
            plt.close()
    print("3D comparison plots saved in:", out_dir)
    print("3D predicted structure plots saved in:", out_dir)

    return pd.DataFrame(rows)
def load_data():
    print("Loading data...")
    train_seqs = pd.read_csv('data/train_sequences.csv')
    val_seqs = pd.read_csv('data/validation_sequences.csv')
    test_seqs = pd.read_csv('data/test_sequences.csv')
    train_labels = pd.read_csv('data/train_labels.csv')
    val_labels = pd.read_csv('data/validation_labels_new.csv')
    return train_seqs, val_seqs, test_seqs, train_labels, val_labels

def main():
    train_seqs, val_seqs, test_seqs, train_labels, val_labels = load_data()
    msa_dir = "data2/MSA"
    val_loader = DataLoader(
        RNA(val_seqs, val_labels, "data2/MSA", max_len=400),
        batch_size=4,
        shuffle=False,
        collate_fn=RNA.collate_fn,
    )
    model = load_model("Trial/best_model1.pth")
    submission_df = analyze_model_performance(model, val_loader, out_dir="./validation_plots3")
    submission_df.to_csv("Predictions.csv", index=False)
if __name__ == "__main__":
    main()