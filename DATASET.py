import os
import numpy as np 
import torch

from torch.utils.data import Dataset
from Bio import SeqIO

from torch.nn import functional as F

class RNADataset(Dataset):
    """Purpose: Build RNA residue features, normalized labels, masks, and contact maps.

    Input:
        seq_df: DataFrame with target_id and sequence columns.
        label_df: Coordinate labels, or None for unlabeled data.
        msa_dir: Directory containing target_id.MSA.fasta files.
        pair_df: Optional ViennaRNA pair-feature DataFrame.
        max_len: Maximum sequence length to keep.
        contact_map_dir: Optional directory, or ordered directories, of SPOT-RNA-2D .npy contact maps.
    Output:
        Dataset items consumed by training and visualization loaders.
    """

    def __init__(self, seq_df, label_df, msa_dir, pair_df=None, max_len=256, contact_map_dir=None):
        """Purpose: Store dataset sources and preprocessing limits.

        Input:
            seq_df: DataFrame containing RNA IDs and sequences.
            label_df: Optional coordinate-label DataFrame.
            msa_dir: Directory of MSA FASTA files.
            pair_df: Optional ViennaRNA pair-feature DataFrame.
            max_len: Maximum residues to keep per target.
            contact_map_dir: Optional SPOT contact-map directory or fallback directories.
        Output:
            Initialized dataset object.
        """
        self.seq_df = seq_df
        self.label_df = label_df
        self.msa_dir = msa_dir
        self.max_len = max_len
        self.pair_df = pair_df
        self.contact_map_dirs = self._normalize_path_list(contact_map_dir)
        self.label_groups = self._index_label_rows(label_df)
        self.pair_groups = self._index_pair_rows(pair_df)

    @staticmethod
    def _normalize_path_list(paths):
        """Purpose: Normalize an optional path or path sequence into a list.

        Input:
            paths: None, one filesystem path, or a sequence of filesystem paths.
        Output:
            List of non-empty path strings.
        """
        if paths is None:
            return []
        if isinstance(paths, (str, os.PathLike)):
            return [os.fspath(paths)] if str(paths).strip() else []
        return [os.fspath(path) for path in paths if str(path).strip()]

    @staticmethod
    def _index_label_rows(label_df):
        """Purpose: Group coordinate-label rows by target ID for fast lookup.

        Input:
            label_df: Label DataFrame with an ID column, or None.
        Output:
            Dictionary mapping target_id to label rows.
        """
        if label_df is None or label_df.empty:
            return {}
        target_ids = label_df["ID"].astype(str).str.rsplit("_", n=1).str[0]
        return {
            target_id: label_df.loc[target_ids == target_id]
            for target_id in target_ids.unique()
        }

    @staticmethod
    def _index_pair_rows(pair_df):
        """Purpose: Group per-residue pair features by target ID for fast lookup.

        Input:
            pair_df: Pair-feature DataFrame with target_id and resid columns, or None.
        Output:
            Dictionary mapping target_id to sorted pair-feature rows.
        """
        if pair_df is None or pair_df.empty:
            return {}
        return {
            target_id: group.sort_values("resid")
            for target_id, group in pair_df.groupby("target_id", sort=False)
        }

    def encode_sequence(self, seq):
        """Purpose: Convert an RNA sequence into nucleotide one-hot features.

        Input:
            seq: RNA sequence string.
        Output:
            Float array shaped (L, 4) for A/C/G/U one-hot channels.
        """
        mapping = {'A':0,'C':1,'G':2,'U':3}
        L = len(seq)
        onehot = np.zeros((L,4), dtype=np.float32)
        for i, nt in enumerate(seq):
            if nt in mapping:
                onehot[i, mapping[nt]] = 1.0
        return onehot   # (L,4)

    def msa_features(self, target_id, L):
        """Purpose: Compute conservation and MSA-depth features per residue.

        Input:
            target_id: RNA target identifier.
            L: Sequence length after max_len truncation.
        Output:
            Float array shaped (L, 2) with conservation and normalized depth.
        """
        msa_path = os.path.join(self.msa_dir, f"{target_id}.MSA.fasta")

        # No MSA file is available for this target.
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

    def pair_features(self, target_id, L):
        """Purpose: Load ViennaRNA pair indicators for one target.

        Input:
            target_id: RNA target identifier.
            L: Sequence length after max_len truncation.
        Output:
            Float array shaped (L, 2) with is_paired and pair_partner_norm.
        """
        if self.pair_df is None:
            return np.zeros((L, 0), dtype=np.float32)

        pair_feats = np.zeros((L, 2), dtype=np.float32)
        row = self.pair_groups.get(target_id)

        if row is None or row.empty:
            return pair_feats

        row = row.iloc[:L]

        pair_feats[:len(row), 0] = row["is_paired"].values.astype(np.float32)
        pair_feats[:len(row), 1] = row["pair_partner_norm"].values.astype(np.float32)
        return pair_feats

    def contact_map(self, target_id, L):
        """Purpose: Load a SPOT-RNA-2D contact map for one target.

        Input:
            target_id: RNA target identifier.
            L: Sequence length after max_len truncation.
        Output:
            Float array shaped (L, L) with contact probabilities.
        """
        contact = np.zeros((L, L), dtype=np.float32)
        map_path = None
        for contact_map_dir in self.contact_map_dirs:
            candidate_path = os.path.join(contact_map_dir, f"{target_id}.npy")
            if os.path.exists(candidate_path):
                map_path = candidate_path
                break
        if map_path is None:
            return contact

        loaded = np.load(map_path).astype(np.float32)
        if loaded.ndim != 2:
            raise ValueError(f"Contact map must be 2D for {target_id}: {loaded.shape}")

        h = min(L, loaded.shape[0])
        w = min(L, loaded.shape[1])
        contact[:h, :w] = loaded[:h, :w]
        return contact

    def get_labels(self, target_id, L):
        """Purpose: Build per-residue normalized coordinate labels.

        Input:
            target_id: RNA target identifier.
            L: Sequence length after max_len truncation.
        Output:
            Float array shaped (L, 3), with NaN rows for missing labels.
        """
        if self.label_df is None:
            return np.full((L,3), np.nan, dtype=np.float32)
        row = self.label_groups.get(target_id)
        coords = np.full((L,3), np.nan, dtype=np.float32)
        if row is None or row.empty:
            return coords
        for _, r in row.iterrows():
            resid = int(r['resid'])
            if 1 <= resid <= L:
                coords[resid-1,0] = r['x_1']
                coords[resid-1,1] = r['y_1']
                coords[resid-1,2] = r['z_1']
        coords[coords <= -1e17] = np.nan
        valid = ~np.isnan(coords).any(axis=1)
        if valid.sum() > 0:
            mean = coords[valid].mean(axis=0, keepdims=True)
            std  = coords[valid].std(axis=0, keepdims=True) + 1e-6
            coords[valid] = (coords[valid] - mean) / std

        return coords

    def __getitem__(self, idx):
        """Purpose: Return one target's model features, labels, contact map, and metadata.

        Input:
            idx: Dataset row index.
        Output:
            Tuple of (features, labels, contact_map, length, target_id).
        """
        row = self.seq_df.iloc[idx]
        target_id = row['target_id']
        seq = row['sequence']
        L = min(len(seq), self.max_len)

        # 1. sequence onehot
        onehot = self.encode_sequence(seq)[:L]  # (L,4)

        # 2. MSA features
        msa = self.msa_features(target_id, L)    # (L,2)

        # 3. pair features

        pair_feats = self.pair_features(target_id, L)

        # 4. concat -> (L, 8)
        feats = np.concatenate([onehot, msa, pair_feats], axis=1)  # (L, 8)


        # 5. labels (coords)
        labels = self.get_labels(target_id, L)   # (L,3)
        contact_map = self.contact_map(target_id, L)
        return (torch.tensor(feats,dtype=torch.float32), 
                torch.tensor(labels,dtype=torch.float32), 
                torch.tensor(contact_map,dtype=torch.float32),
                L,
                target_id )

    def __len__(self):
        """Purpose: Report the number of RNA targets in the dataset.

        Input:
            None.
        Output:
            Integer dataset length.
        """
        return len(self.seq_df)
    
    def collate_fn(batch):
        """Purpose: Pad variable-length RNA targets into one batch.

        Input:
            batch: Iterable of RNADataset items.
        Output:
            Tuple of padded features, labels, contact maps, masks, lengths, and IDs.
        """
        feats, labels, contact_maps, lengths, ids = zip(*batch)

        max_L = max(lengths)

        feat_list = []
        label_list = []
        contact_list = []
        seq_mask_list = []
        id_list = []
        
        for feat, label, contact_map, L, target_id in batch:
            pad_f = F.pad(feat, (0,0,0,max_L-feat.shape[0]))
            pad_l = F.pad(label, (0,0,0,max_L-label.shape[0]))
            pad_c = F.pad(contact_map, (0,max_L-contact_map.shape[1],0,max_L-contact_map.shape[0]))
            seq_mask = torch.zeros(max_L)
            seq_mask[:L] = 1

            feat_list.append(pad_f)
            label_list.append(pad_l)
            contact_list.append(pad_c)
            seq_mask_list.append(seq_mask)
            id_list.append(target_id)


        feats = torch.stack(feat_list)       # (B, L, 8)
        labels = torch.stack(label_list)     # (B, L, 3)
        contact_maps = torch.stack(contact_list) # (B, L, L)
        seq_mask = torch.stack(seq_mask_list)        # (B, L)

        coord_mask = (~torch.isnan(labels).any(dim=-1)).float() * seq_mask  # (B, L)
        labels = torch.nan_to_num(labels, nan=0.0)
        # transpose for Conv1D:
        feats = feats.permute(0,2,1)         # (B, C, L)

        return feats, labels, contact_maps, seq_mask, coord_mask, lengths, id_list

    
    
# feats:      (B, 8, L)   # input to model
# labels:     (B, L, 3)   # target coordinates
# contact_maps: (B, L, L) # SPOT-RNA-2D contact probabilities
# seq_mask:   (B, L)      # 1 = real residue, 0 = padding
# coord_mask: (B, L)      # 1 = valid coordinate label, 0 = ignore in loss/metric
# lengths:    list[int]
# ids:        list[str]


