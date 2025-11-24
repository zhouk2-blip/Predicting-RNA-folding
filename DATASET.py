import os
import numpy as np 
import pandas as pd 
import torch

from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO

from torch.nn import functional as F

class RNADataset(Dataset):
    def __init__(self, seq_df, label_df, msa_dir, max_len=256):
        self.seq_df = seq_df
        self.label_df = label_df
        self.msa_dir = msa_dir
        self.max_len = max_len

    def encode_sequence(self, seq):
        mapping = {'A':0,'C':1,'G':2,'U':3}
        L = len(seq)
        onehot = np.zeros((L,4), dtype=np.float32)
        for i, nt in enumerate(seq):
            if nt in mapping:
                onehot[i, mapping[nt]] = 1.0
        return onehot   # (L,4)

    def msa_features(self, target_id, L):
        msa_path = os.path.join(self.msa_dir, f"{target_id}.MSA.fasta")

        # no MSA → return zeros
        if not os.path.exists(msa_path):
            return np.zeros((L,2),dtype=np.float32)

        sequences = [str(rec.seq) for rec in SeqIO.parse(msa_path,"fasta")]
        counts = np.zeros((L,4), dtype=np.float32)

        for seq in sequences:
            for i, nt in enumerate(seq):
                if i >= L: break
                if nt == '-':
                    continue 
                if   nt=='A': counts[i,0]+=1
                elif nt=='C': counts[i,1]+=1
                elif nt=='G': counts[i,2]+=1
                elif nt=='U': counts[i,3]+=1

        freqs = counts / (counts.sum(axis=1, keepdims=True) + 1e-6)
        entropy = -np.sum(freqs * np.log(freqs + 1e-6), axis=1)
        conservation = 1 - entropy / np.log(4)
        depth = np.ones(L) * len(sequences)
        if depth.max() > 0:
            depth = depth / depth.max()

        return np.stack([conservation, depth], axis=1)
    

    def get_labels(self, target_id, L):
        if self.label_df is None:
            return np.full((L,3), np.nan, dtype=np.float32)
        row = self.label_df[self.label_df['ID'].str.startswith(target_id+'_')]
        coords = np.full((L,3), np.nan, dtype=np.float32)
        if row.empty:
            return coords
        for _, r in row.iterrows():
            resid = int(r['resid'])
            if 1 <= resid <= L:
                coords[resid-1,0] = r['x_1']
                coords[resid-1,1] = r['y_1']
                coords[resid-1,2] = r['z_1']
        valid = ~np.isnan(coords).any(axis=1)
        if valid.sum() > 0:
            mean = coords[valid].mean(axis=0, keepdims=True)
            std  = coords[valid].std(axis=0, keepdims=True) + 1e-6
            coords[valid] = (coords[valid] - mean) / std

        return coords
    def detect_outliers(self,coords, max_jump=20.0):
        # coords: numpy array (L,3)
        L = coords.shape[0]
        for i in range(1, L):
            if not np.any(np.isnan(coords[i])) and not np.any(np.isnan(coords[i-1])):
                d = np.linalg.norm(coords[i] - coords[i-1])
                if d > max_jump:
                    coords[i] = np.array([np.nan, np.nan, np.nan])
        return coords
    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        target_id = row['target_id']
        seq = row['sequence']
        L = min(len(seq), self.max_len)

        # 1. sequence onehot
        onehot = self.encode_sequence(seq)[:L]  # (L,4)

        # 2. MSA features
        msa = self.msa_features(target_id, L)    # (L,2)

        # 3. concat → (L,6)
        feats = np.concatenate([onehot, msa], axis=1)

        # 4. labels (coords)
        labels = self.get_labels(target_id, L)   # (L,3)
        labels = self.detect_outliers(labels)
        return (torch.tensor(feats,dtype=torch.float32), 
                torch.tensor(labels,dtype=torch.float32), 
                L,
                target_id )
    def __len__(self):
        return len(self.seq_df)
    
    def collate_fn(batch):
        feats, labels, lengths, ids = zip(*batch)

        max_L = max(lengths)

        feat_list = []
        label_list = []
        mask_list = []
        id_list = []
        
        for feat, label, L, target_id in batch:
            pad_f = F.pad(feat, (0,0,0,max_L-feat.shape[0]))
            pad_l = F.pad(label, (0,0,0,max_L-label.shape[0]))
            mask = torch.zeros(max_L)
            mask[:L] = 1

            feat_list.append(pad_f)
            label_list.append(pad_l)
            mask_list.append(mask)
            id_list.append(target_id)


        feats = torch.stack(feat_list)       # (B, L, 6)
        labels = torch.stack(label_list)     # (B, L, 3)
        mask = torch.stack(mask_list)        # (B, L)

        coord_mask = ~torch.isnan(labels).any(dim=-1)  # (B, L)
        final_mask = mask * coord_mask.float()  # (B, L)
        labels = torch.nan_to_num(labels, nan=0.0)
        # transpose for Conv1D:
        feats = feats.permute(0,2,1)         # (B, C, L)

        return feats, labels, final_mask, lengths, id_list

    
    
# feats:      (B, 6, L)   # input to model
# labels:     (B, L, 3)   # target coordinates
# final_mask: (B, L)      # 1 = valid, 0 = ignore
# lengths:    list[int]
# ids:        list[str]


