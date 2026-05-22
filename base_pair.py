import pandas as pd
import numpy as np
import RNA
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path


DATA_DIR = "dataset"
DEFAULT_INPUT_CSV = Path(DATA_DIR) / "validation_sequences.csv"
DEFAULT_STRUCTURE_OUTPUT = Path(DATA_DIR) / "validation_secondary_structures.csv"
DEFAULT_PAIR_OUTPUT = Path(DATA_DIR) / "validation_pair_features.csv"


def dotbracket_to_pairs(dotbracket):
    """Purpose: Convert dot-bracket notation into partner indices.

    Input:
        dotbracket: ViennaRNA dot-bracket secondary-structure string.
    Output:
        NumPy array shaped (L,), where partner[i] is the paired 0-based index
        or -1 when residue i is unpaired.
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
    """Purpose: Predict secondary structure and build per-residue pair features.

    Input:
        target_id: RNA target identifier.
        sequence: RNA sequence string.
    Output:
        Tuple of (structure metadata row, per-residue feature rows).
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


def parse_args():
    """Purpose: Parse command-line options for ViennaRNA pair-feature generation.

    Input:
        Command-line arguments.
    Output:
        argparse Namespace with input and output CSV paths.
    """
    parser = ArgumentParser(description="Generate ViennaRNA secondary-structure pair features.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--structure-output", default=str(DEFAULT_STRUCTURE_OUTPUT))
    parser.add_argument("--pair-output", default=str(DEFAULT_PAIR_OUTPUT))
    return parser.parse_args()


def main():
    """Purpose: Generate ViennaRNA secondary-structure feature CSV files.

    Input:
        Command-line arguments parsed by parse_args().
    Output:
        Writes secondary-structure metadata and per-residue pair-feature CSVs.
    """
    args = parse_args()
    input_csv = Path(args.input_csv)
    structure_output = Path(args.structure_output)
    pair_output = Path(args.pair_output)

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

    structure_output.parent.mkdir(parents=True, exist_ok=True)
    pair_output.parent.mkdir(parents=True, exist_ok=True)
    structure_df.to_csv(structure_output, index=False)
    feature_df.to_csv(pair_output, index=False)

    print(f"Saved {structure_output}")
    print(f"Saved {pair_output}")


if __name__ == "__main__":
    main()
