import os 
import numpy as np
import pandas as pd
from Bio import SeqIO
train_sequence = pd.read_csv('data/train_sequences.csv')
train_sequence2 = pd.read_csv('data/train_sequences.v2.csv')
train_labels = pd.read_csv('data/train_labels.csv')
train_labels2 = pd.read_csv('data/train_labels.v2.csv')
print(train_sequence.head())
def one_hot_encode_sequence(sequence):
    mapping = {'A':0, 'C':1, 'G':2, 'U':3}
    one_hot = np.zeros((len(sequence), 4), dtype=int)
    for i, nucleotide in enumerate(sequence[:len(sequence)]):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1
    return one_hot

def main():
    
    sequences1 = train_sequence['sequence']
    sequences2 = train_sequence2['sequence']
    encoded_sequences1 = [one_hot_encode_sequence(seq).tolist() for seq in sequences1]
    encoded_sequences2 = [one_hot_encode_sequence(seq) for seq in sequences2]
    encoded_df1 = pd.DataFrame({'sequence':sequences1, 'one_hot': encoded_sequences1})
    encoded_df2 = pd.DataFrame({'sequence':sequences2, 'one_hot': encoded_sequences2})
    encoded_df1.to_csv('data2/encoded_sequences.csv', index=False)
    encoded_df2.to_csv('data2/encoded_sequences2.csv', index=False)
if __name__ == "__main__":
    main()
    