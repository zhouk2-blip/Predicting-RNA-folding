import os
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
SEQ_LENGTH = 90
Batch_SIZE = 32
EPOCHS = 50
DEVICE = torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PREDICTIONS = 3



class RNADATAset(Dataset):
    def __init__(self, sequences, labels,msa_dir,max_len = SEQ_LENGTH,normalize= True ):#seqeunces df; labels df
        self.sequneces = sequences
        self.labels = labels
        self.msa_dir = msa_dir
        self.max_len = max_len
        self.feature_scaler = StandardScaler()
        self.normalize = normalize
        self.data = []
        self.preprocess_data()
    def preprocess_seqeunce(self,seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        one_hot = np.zeros((self.max_len, 4), dtype=np.float32)
        for i, base in enumerate(seq[:self.max_len]):
            if base in mapping:
                one_hot[i, mapping[base]] = 1.0
        return one_hot
    def get_msa_features(self, target_id):
        msa_path = os.path.join(self.msa_dir, f"{target_id}_msa.fasta")
        if not os.path.exists(msa_path):
            return np.zeros((self.max_len, 2), dtype=np.float32)
        try: 
            sequneces = [str(rec.seq) for rec in SeqIO.parse(msa_path, "fasta")]# load sequences from MSA file
            if sequneces :
                counts = np.zeros((self.max_len, 4), dtype=np.float32)
                for seq in sequneces:
                    for i, base in enumerate(seq[:self.max_len]):
                        if base == 'A':
                            counts[i, 0] += 1
                        elif base == 'C':
                            counts[i, 1] += 1
                        elif base == 'G':
                            counts[i, 2] += 1
                        elif base == 'U':
                            counts[i, 3] += 1
                counts += 1e-6  # avoid division by zero
                freqs = counts / counts.sum(axis=1, keepdims=True)
                entropy = -np.sum(freqs * np.log(freqs), axis=1)  # Shannon entropy
                conservation = 1.0 - (entropy / np.log(4))  # normalized conservation score
                sep_depth = np.ones(self.max_len,dtype = np.float32)*len(sequneces)/100 #normalize the sequence length and add it as a feature
                features = np.column_stack((conservation, sep_depth))
                if features.shape[0] < self.max_len:
                    pad_width = self.max_len - features.shape[0]
                    features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')#pad to max length
                elif features.shape[0] > self.max_len:
                    features = features[:self.max_len, :]#truncate to max length
                return features
            else:
                return np.zeros((self.max_len, 2), dtype=np.float32)
        except Exception as e:
            print(f"Error loading MSA for {target_id}: {e}")
            return np.zeros((self.max_len, 2), dtype=np.float32)
    def preprocess_labels(self,target_id):
        if self.labels.empty or 'ID' not in self.labels.columns:
            return np.zeros((NUM_PREDICTIONS,self.max_len, 3), dtype=np.float32)
        target_labels = self.labels[self.labels['ID'] == target_id]
        coords_list = []
        for _, row in target_labels.iterrows():# returns a Series, which iterate over each row in the DataFrame
            struct_coords = []
            for i in range(1,40):
                x = row[f'x_{i}'] if f"x_{i}" in row else np.nan
                y = row[f'y_{i}'] if f"y_{i}" in row else np.nan
                z = row[f'z_{i}'] if f"z_{i}" in row else np.nan
                if pd.notna(x) and pd.notna(y) and pd.notna(z):
                    struct_coords.append([x, y, z])
            if len(struct_coords) > self.max_len:
                struct_coords = struct_coords[:self.max_len]
            elif len(struct_coords) < self.max_len:
                pad_length = self.max_len - len(struct_coords)
                struct_coords.extend([[0.0, 0.0, 0.0]] * pad_length)
            coords_list.append(struct_coords)
        if len (coords_list) < NUM_PREDICTIONS:
            for _ in range(NUM_PREDICTIONS - len(coords_list)):
                coords_list.append([[0.0, 0.0, 0.0]] * self.max_len)
        else:
            coords_list = coords_list[:NUM_PREDICTIONS]
        coords_array = np.array(coords_list, dtype=np.float32)#transform to nprray to do normalization
        coords_array = (coords_array - np.mean(coords_array, axis=0)) / (np.std(coords_array, axis=0) + 1e-8) #(NUM_PREDICTIONS, seq_len, 3)
        return coords_array
    def preprocess_data(self):
        all_features = []
        
        print( "Creating dataset...")
        for _, row in tqdm(self.sequneces.iterrows(), total=len(self.sequneces)):
            target_id = row['target_id']
            sequence = row['sequence']
            seq_features = self.preprocess_seqeunce(sequence)
            msa_features = self.get_msa_features(target_id)
            if isinstance(msa_features, np.ndarray) and msa_features.ndim == 1:# fix shape issue
                msa_features = msa_features.reshape(-1, 1)
            if seq_features.shape[0] != msa_features.shape[0]:
                msa_features = msa_features[:seq_features.shape[0], :]
            features = np.concatenate((seq_features, msa_features), axis=1)
            if self.normalize:
                all_features.append(features.flatten()) #flatten to 1D array (num_features,)
        
        # Fit scaler once on all training features
        if self.normalize and all_features:
            all_features = np.array(all_features)  # shape (N, num_features)
            self.feature_scaler.fit(all_features)
        
        # Now transform each feature and store in dataset
        for idx, row in enumerate(tqdm(self.sequneces.iterrows(), total=len(self.sequneces), desc="Transforming")):
            _, row_data = row
            target_id = row_data['target_id']
            sequence = row_data['sequence']
            seq_features = self.preprocess_seqeunce(sequence)
            msa_features = self.get_msa_features(target_id)
            if isinstance(msa_features, np.ndarray) and msa_features.ndim == 1:
                msa_features = msa_features.reshape(-1, 1)
            if seq_features.shape[0] != msa_features.shape[0]:
                msa_features = msa_features[:seq_features.shape[0], :]
            features = np.concatenate((seq_features, msa_features), axis=1)
            
            if self.normalize:
                flat_features = features.reshape(1, -1) #(1, num_features)
                features = self.feature_scaler.transform(flat_features).reshape(features.shape)#(1,num_features) -> (seq_len,num_features)
            
            corrds = self.preprocess_labels(target_id)
            self.data.append((
                torch.tensor(features.T, dtype=torch.float32), # Transpose to (num_features, seq_len)
                torch.tensor(corrds, dtype=torch.float32)
                ))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
    def custrom_collate_fn(batch):
        #future plan: use mini batching with padding to handle variable length sequences
        features_list,labels_list = zip(*batch) # separates features and labels from the batch of samples
        features = torch.stack (features_list,dim = 0) #(batch_size,num_features,seq_len)
        
        # Track original sequence lengths before padding
        orig_seq_lens = torch.tensor([f.shape[1] for f in features_list], dtype=torch.long)
        
        fixed_seq_len = SEQ_LENGTH
        if features.shape[2]< fixed_seq_len:
            pad_size = fixed_seq_len - features.shape[2]
            features = F.pad(features, (0,pad_size), "constant",0)#tensor padding
        elif features.shape[2]> fixed_seq_len:
            features = features[:,:,:fixed_seq_len]
            orig_seq_lens = torch.min(orig_seq_lens, torch.tensor(fixed_seq_len))
        
        padded_labels = []
        for labels in labels_list:
            if labels.shape[1]< fixed_seq_len:
                pad_size = fixed_seq_len - labels.shape[1]
                padded_label = F.pad(labels, (0,pad_size), "constant",0)#tensor padding
            elif labels.shape[1]> fixed_seq_len:
                padded_label = labels[:,:fixed_seq_len,:]
            else:
                padded_label = labels
            padded_labels.append(padded_label)
        labels = torch.stack(padded_labels,dim=0) #(batch_size,NUM_PREDICTIONS,seq_len,3)
        return features, labels, orig_seq_lens

class RNA3DModel(nn.Module):
    def __init__(self, input_channels = 6, seq_length = SEQ_LENGTH, num_structures = NUM_PREDICTIONS):
        super().__init__()
        self.num_structures = num_structures
        self.seq_length = seq_length
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.output =nn.Sequential(
            nn.Conv1d(256, num_structures * 3, kernel_size=1),
            nn.Tanh()        
                    )
    def forward(self, x):# x: (batch_size, num_features, seq_len)
        x = self.conv_block(x)  # (batch_size, 256, seq_len)
        x = self.output(x) #(batch_size,num_structures*3, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_features*3)
        x = x.view(x.size(0), x.size(1), self.num_structures, 3)  # (batch_size, seq_len, num_structures, 3)
        return x
    
def tm_score_loss(pred, target, seq_lengths=None):# target: (batch_size,NUM_PREDICTIONS,seq_len,3)
    target = target.permute(0,2,1,3)#(batch_size,seq_len,NUM_PREDICTIONS,3)   
    pred_struct = pred[:,:,0,:]
    
    # Create mask to ignore padded positions
    if seq_lengths is not None:
        mask = torch.arange(pred.shape[1], device=DEVICE).unsqueeze(0) < seq_lengths.unsqueeze(1)  # (batch, seq_len)
    else:
        mask = torch.ones(pred.shape[0], pred.shape[1], device=DEVICE, dtype=torch.bool)
    
    all_tm_scores = []
    for i in range(target.shape[2]):
        target_struct = target[:,:,i,:]
        sqaured_dists = torch.sum((pred_struct - target_struct)**2, dim=-1)+1e-8
        dists = torch.sqrt(sqaured_dists)
        
        # Use actual sequence length (average of batch, or use first sample)
        if seq_lengths is not None:
            seq_len = int(seq_lengths[0].item())  # Use first sample's actual length
        else:
            seq_len = target.shape[1]
        
        if seq_len >= 30:
            d0 = 0.6 * (seq_len - 0.5) ** (1/2) - 2.5
        else:
            d0 = 0.5  # default for shorter sequences
        
        tm_score_components = 1 / (1 + (dists / d0) ** 2)
        
        # Apply mask: only compute TM-score on non-padded positions
        tm_score_components = tm_score_components * mask.float()
        
        # Compute mean only over non-padded positions
        if seq_lengths is not None:
            tm_scores = tm_score_components.sum(dim=1) / seq_lengths.float()
        else:
            tm_scores = tm_score_components.mean(dim=1)
        
        all_tm_scores.append(tm_scores)
    
    all_tm_scores = torch.stack(all_tm_scores, dim=-1)  #(batch_size,NUM_PREDICTIONS)
    best_tm_scores = all_tm_scores.max(dim=1)[0] #(batch_size,)
    l2_reg = torch.mean(torch.norm(pred_struct, dim=2)) * 0.001
    return -best_tm_scores.mean() + l2_reg
def compute_accuracy(pred, target, seq_lengths=None, threshold=0.5):
    pred_struct = pred[:,:,0,:]
    target = target.permute(0,2,1,3)
    best_acc = torch.zeros(pred.size(0)).to(DEVICE)
    
    # Create mask to ignore padded positions
    if seq_lengths is not None:
        mask = torch.arange(pred.shape[1], device=DEVICE).unsqueeze(0) < seq_lengths.unsqueeze(1)  # (batch, seq_len)
    else:
        mask = torch.ones(pred.shape[0], pred.shape[1], device=DEVICE, dtype=torch.bool)
    
    all_tm_scores = []
    for i in range(target.shape[2]):
        target_struct = target[:,:,i,:]
        distances = torch.sqrt(torch.sum((pred_struct - target_struct)**2, dim=-1)+1e-8)
        correct_per_pos = (distances < threshold).float()
        
        # Apply mask: set padded positions to 0 (don't count towards accuracy)
        correct_per_pos = correct_per_pos * mask.float()
        
        # Compute mean only over non-padded positions
        if seq_lengths is not None:
            correct = correct_per_pos.sum(dim=1) / seq_lengths.float()
        else:
            correct = correct_per_pos.mean(dim=1)
        
        best_acc = torch.max(torch.stack([best_acc, correct]),dim = 0)[0]
    return best_acc.mean().item()
       



def load_data():
    print("Loading data...")
    train_seqs = pd.read_csv('data/train_sequences.csv')
    val_seqs = pd.read_csv('data/validation_sequences.csv')
    test_seqs = pd.read_csv('data/test_sequences.csv')
    train_labels = pd.read_csv('data/train_labels.csv')
    val_labels = pd.read_csv('data/validation_labels.csv')
    return train_seqs, val_seqs, test_seqs, train_labels, val_labels


def main():
    train_seqs, val_seqs, test_seqs, train_labels, val_labels = load_data()
    msa_dir = "data2/MSA"
    train_dataset = RNADATAset(train_seqs, train_labels, msa_dir)
    val_dataset = RNADATAset(val_seqs, val_labels, msa_dir)
    test_dataset = RNADATAset(test_seqs, pd.DataFrame(), msa_dir)
    #create dataloaders 
    train_loader = DataLoader(train_dataset, batch_size= Batch_SIZE, shuffle=True, collate_fn= RNADATAset.custrom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size= Batch_SIZE, shuffle=False, collate_fn= RNADATAset.custrom_collate_fn)
    
    test_loader = DataLoader(test_dataset, batch_size= Batch_SIZE, shuffle=False, collate_fn= RNADATAset.custrom_collate_fn)
    
    model = RNA3DModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
   
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            features, labels, seq_lengths = batch
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            seq_lengths = seq_lengths.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(features)
            loss = tm_score_loss(predictions, labels, seq_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                features, labels, seq_lengths = batch
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                seq_lengths = seq_lengths.to(DEVICE)
                predictions = model(features)
                loss = tm_score_loss(predictions, labels, seq_lengths)
                val_loss += loss.item() * features.size(0)
                acc = compute_accuracy(predictions, labels, seq_lengths)
                val_acc += acc * features.size(0)
        
        val_loss /= len(val_dataset)
        val_acc /= len(val_dataset)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        scheduler.step(val_loss)

        # Question: how to deal with the padding issue and let the model know the true sequence length?

if __name__ == "__main__":
    main()

