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
EPOCHS = 40
BATCH_SIZE = 32
IR = 3e-4
MAX_LEN = 128
THRESHOLD = 0.3
PATIENCE = 7
DISTANCE = 1.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_data():
    print("Loading data...")
    train_seqs = pd.read_csv('data/train_sequences.csv')
    val_seqs = pd.read_csv('data/validation_sequences.csv')
    test_seqs = pd.read_csv('data/test_sequences.csv')
    train_labels = pd.read_csv('data/train_labels.csv')
    val_labels = pd.read_csv('data/validation_labels_new.csv')
    return train_seqs, val_seqs, test_seqs, train_labels, val_labels

def load_model(model_path):
    model = RNAmodel().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# the loss function combining TM-score loss and masked MSE loss
# decide not to use that for now
# def hybrid_loss(pred,target,mask,alpha = 0.8,beta = 0.2):
#     # --- TM loss ---   
#     tm = compute_tm_score(pred, target, mask)  # (B,)
#     tm_loss = (1 - tm).mean()                  # lower = better

#     # --- Masked MSE loss ---
#     mask = mask.float().unsqueeze(-1)           # (B,L,1)
#     diff2 = ((pred - target)**2) * mask
#     valid = mask.sum().clamp(min=1.0)
#     mse_loss = diff2.sum() / valid

#     # --- Hybrid ---
#     loss = alpha * tm_loss + beta * mse_loss
#     return loss
# the loss function using only masked MSE loss

def masked_mse_loss(pred,target,mask):
    mask = mask.float().unsqueeze(-1)           # (B,L,1)
    diff2 = ((pred - target)**2) * mask
    valid = mask.sum().clamp(min=1.0)
    mse_loss = diff2.sum() / valid
    return mse_loss

# def compute_tm_score(pred,target,mask):
#     B,L,_ = pred.shape
#     base_mask = mask.float() # (B,L)
#     mask = base_mask.unsqueeze(-1)  # (B,L,1)
#     dist = torch.norm((pred-target)*mask,dim=-1) # (B,L)
#     valid = mask.sum(dim=1).clamp(min=1.0)  # (B,)
#     d0 = 0.6*torch.sqrt(valid - 0.5)-2.5
#     d0 = torch.clamp(d0,min = 0.1)
#     d0 = d0.unsqueeze(-1)  # (B,1)
#     tm_components = 1 / (1 + (dist / d0)**2)  # (B,L)
#     tm_scores = (tm_components * base_mask).sum(dim=1) / valid  # (B,)
#     return tm_scores

# def compute_tm_accuracy(pred,target,mask,threshold =THRESHOLD):
#     tm_scores = compute_tm_score(pred,target,mask)
#     correct = (tm_scores > threshold).float().mean().item()
#     return correct

def compute_accuracy(pred, target, mask, threshold=2):
    """
    pred: (B, L, 3)
    target: (B, L, 3)
    mask: (B, L)
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


def train_validate(train,label,val,val_label,msa_dir,save_path = 'Trial/best_model2.pth', epochs = EPOCHS,batch_size = BATCH_SIZE,Ir = IR,max_len = MAX_LEN,patience = PATIENCE):
    history  = {'epoch':[],'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    train_dataset = RNA(train,label,msa_dir,max_len=max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle = True,
        collate_fn=RNA.collate_fn, 
        )
    val_dataset = RNA(val, val_label, msa_dir, max_len=max_len)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RNA.collate_fn,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNAmodel().to(device)
    
    #model = load_model('Trial/best_model1.pth').to(device)
    optimizer = optim.Adam(model.parameters(), lr=Ir)

    best_val_loss = float('inf')
    patience_counter = 0
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_tm_acc = 0.0
        train_acc = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for feats,labels,mask,lengths,ids in progress:
            
            feats = feats.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            outputs = model(feats,mask)
            loss = masked_mse_loss(outputs, labels, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            acc = compute_accuracy(outputs, labels, mask, threshold=DISTANCE)
            train_acc += acc
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        print(f"Epoch {epoch+1}train Loss: {avg_epoch_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
        
        model.eval()
        val_loss = 0.0
    
        val_acc = 0.0
        with torch.no_grad():
            for feats,labels,mask,lengths,ids in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                outputs = model(feats,mask)
                loss = masked_mse_loss(outputs, labels, mask)
                val_loss += loss.item()
                
                acc = compute_accuracy(outputs, labels, mask, threshold=DISTANCE)
                val_acc += acc

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    print("Training complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_history.png")
    plt.show()
    plt.figure(figsize=(10,6))
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy', marker='o')     
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy History')
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_accuracy_history.png")
    plt.show()
    return best_val_loss,history



def main():
    train_seqs, val_seqs, test_seqs, train_labels, val_labels = load_data()
    msa_dir = "data2/MSA"
    train_validate(train_seqs, train_labels, val_seqs, val_labels, msa_dir)
    

if __name__ == "__main__":
    main()