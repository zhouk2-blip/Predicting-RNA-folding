import pandas as pd
import numpy as np
import RNA
from tqdm import tqdm


def dotbracket_to_pairs(dotbracket):
    """
    Convert dot-bracket structure into partner index array.

    Returns:
        partner: np.array of shape (L,)
                 partner[i] = paired position index, 0-based
                 partner[i] = -1 if unpaired
    """
    L = len(dotbracket)
    partner = np.full(L, -1, dtype=np.int64)
    stack = []

    for i, ch in enumerate(dotbracket):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if len(stack) == 0:
                raise ValueError(f"Unmatched ')' at position {i}")
            j = stack.pop()
            partner[i] = j
            partner[j] = i

    if len(stack) > 0:
        raise ValueError("Unmatched '(' in dot-bracket structure")

    return partner


def make_pair_features(target_id, sequence):
    """
    Use ViennaRNA to predict secondary structure,
    then convert it into per-residue pair features.
    """
    sequence = sequence.upper().replace("T", "U")

    structure, mfe = RNA.fold(sequence)
    partner = dotbracket_to_pairs(structure)

    L = len(sequence)
    rows = []

    for i in range(L):
        is_paired = 1 if partner[i] != -1 else 0

        # 1-based partner index for readability.
        # If unpaired, use 0.
        pair_partner = int(partner[i] + 1) if partner[i] != -1 else 0

        # normalized partner index for neural network input.
        # If unpaired, use 0.0.
        pair_partner_norm = pair_partner / L if pair_partner != 0 else 0.0

        rows.append({
            "target_id": target_id,
            "resid": i + 1,
            "base": sequence[i],
            "dotbracket_symbol": structure[i],
            "is_paired": is_paired,
            "pair_partner": pair_partner,
            "pair_partner_norm": pair_partner_norm,
            "mfe": mfe,
        })

    structure_row = {
        "target_id": target_id,
        "sequence": sequence,
        "dotbracket": structure,
        "mfe": mfe,
        "length": L,
    }

    return structure_row, rows


def main():
    input_csv = "project\\dataset\\validation_sequences.csv"

    seq_df = pd.read_csv(input_csv)

    all_structure_rows = []
    all_feature_rows = []

    for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="Running ViennaRNA"):
        target_id = row["target_id"]
        sequence = row["sequence"]

        structure_row, feature_rows = make_pair_features(target_id, sequence)

        all_structure_rows.append(structure_row)
        all_feature_rows.extend(feature_rows)

    structure_df = pd.DataFrame(all_structure_rows)
    feature_df = pd.DataFrame(all_feature_rows)

    structure_df.to_csv("train_secondary_structures.csv", index=False)
    feature_df.to_csv("train_pair_features.csv", index=False)

    print("Saved train_secondary_structures.csv")
    print("Saved train_pair_features.csv")


if __name__ == "__main__":
    main()