import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = "dataset"
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "spot_maps")
DEFAULT_INPUT_FEATS_DIR = os.path.join(DATA_DIR, "spot_input_features")
DEFAULT_SPOT_OUTPUTS_DIR = os.path.join(DATA_DIR, "spot_outputs")


def read_sequences(csv_paths, max_len=None):
    """Purpose: Read unique RNA target sequences from one or more CSV files.

    Input:
        csv_paths: Iterable of CSV paths with target_id and sequence columns.
        max_len: Optional maximum sequence length to keep.
    Output:
        List of (target_id, sequence) tuples with duplicate target IDs removed.
    """
    rows = []
    seen = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "target_id" not in df.columns or "sequence" not in df.columns:
            raise ValueError(f"{csv_path} must contain target_id and sequence columns")
        for _, row in df.iterrows():
            target_id = str(row["target_id"])
            if target_id in seen:
                continue
            seq = str(row["sequence"]).upper().replace("T", "U")
            if max_len is not None:
                seq = seq[:max_len]
            rows.append((target_id, seq))
            seen.add(target_id)
    return rows


def write_spot_inputs(sequences, input_feats_dir, list_path, use_empty_rnafold_prob=False):
    """Purpose: Write sequence files and ID list expected by SPOT-RNA-2D.

    Input:
        sequences: List of (target_id, sequence) tuples.
        input_feats_dir: Directory for SPOT-RNA-2D input feature files.
        list_path: Path to write the target ID list.
        use_empty_rnafold_prob: Whether to write RNAfold placeholder files.
    Output:
        None. Writes files to disk.
    """
    input_feats_dir = Path(input_feats_dir)
    input_feats_dir.mkdir(parents=True, exist_ok=True)
    Path(list_path).parent.mkdir(parents=True, exist_ok=True)

    with open(list_path, "w", encoding="utf-8") as list_file:
        for target_id, seq in sequences:
            seq_path = input_feats_dir / target_id
            with open(seq_path, "w", encoding="utf-8") as seq_file:
                seq_file.write(f">{target_id}\n{seq}\n")
            if use_empty_rnafold_prob:
                dp_path = input_feats_dir / f"{target_id}_dp.ps"
                ss_path = input_feats_dir / f"{target_id}_ss.ps"
                dp_path.write_text("% Empty RNAfold probability placeholder\n1 1 0 lbox\n", encoding="utf-8")
                ss_path.write_text("% Empty RNAfold structure placeholder\n", encoding="utf-8")
            list_file.write(f"{target_id}\n")


def run_spot_single(args, list_path):
    """Purpose: Execute the external SPOT-RNA-2D repository for prepared inputs.

    Input:
        args: Parsed command-line arguments with SPOT paths and runtime options.
        list_path: Path to the SPOT target ID list.
    Output:
        None. Raises if the external command fails.
    """
    spot_repo = Path(args.spot_repo).resolve()
    run_py = spot_repo / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"Could not find SPOT-RNA-2D run.py at {run_py}")

    input_feats_dir = Path(args.input_feats_dir).resolve()
    spot_outputs_dir = Path(args.spot_outputs_dir).resolve()
    list_path = Path(list_path).resolve()

    run_args = [
        str(run_py),
        "--list_rna_ids",
        str(list_path),
        "--input_feats",
        str(input_feats_dir),
        "--single_seq",
        "1",
        "--outputs",
        str(spot_outputs_dir),
        "--gpu",
        str(args.gpu),
        "--cpu",
        str(args.cpu),
    ]
    if args.conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python"] + run_args
    else:
        cmd = [args.python] + run_args
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["CONDA_REPORT_ERRORS"] = "false"
    subprocess.run(cmd, cwd=spot_repo, check=True, env=env)


def load_numeric_text_matrix(path):
    """Purpose: Load a numeric 2D matrix from a text-like file.

    Input:
        path: Path to a whitespace- or comma-delimited matrix file.
    Output:
        NumPy matrix, or None if the file cannot be parsed as a 2D matrix.
    """
    try:
        matrix = np.loadtxt(path, dtype=np.float32)
    except Exception:
        try:
            matrix = np.loadtxt(path, dtype=np.float32, delimiter=",")
        except Exception:
            return None

    if matrix.ndim == 2:
        return matrix
    return None


def edge_list_to_matrix(data, length):
    """Purpose: Convert an edge-list contact format into a square matrix.

    Input:
        data: Numeric array with at least i, j, value columns.
        length: Expected RNA sequence length.
    Output:
        Symmetric matrix shaped (length, length), or None if data is invalid.
    """
    if data.ndim != 2 or data.shape[1] < 3:
        return None

    ij = data[:, :2]
    values = data[:, 2]
    if not np.all(np.isfinite(ij)) or not np.all(np.isfinite(values)):
        return None

    rounded = np.rint(ij)
    if not np.allclose(ij, rounded):
        return None

    idx = rounded.astype(np.int64)
    if idx.min() >= 1 and idx.max() <= length:
        idx -= 1
    elif idx.min() >= 0 and idx.max() < length:
        pass
    else:
        return None

    matrix = np.zeros((length, length), dtype=np.float32)
    matrix[idx[:, 0], idx[:, 1]] = values.astype(np.float32)
    matrix[idx[:, 1], idx[:, 0]] = values.astype(np.float32)
    return matrix


def load_candidate_matrix(path, length):
    """Purpose: Load one possible SPOT output file as a contact matrix.

    Input:
        path: Candidate output path.
        length: Expected RNA sequence length.
    Output:
        Contact matrix shaped (length, length), or None if incompatible.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            matrix = np.load(path).astype(np.float32)
        except Exception:
            return None
    else:
        matrix = load_numeric_text_matrix(path)
        if matrix is None:
            return None

    if matrix.ndim != 2:
        return None

    if matrix.shape[0] == length and matrix.shape[1] == length:
        return matrix.astype(np.float32)

    edge_matrix = edge_list_to_matrix(matrix, length)
    if edge_matrix is not None:
        return edge_matrix

    return None


def find_spot_output_matrix(target_id, length, spot_outputs_dir):
    """Purpose: Locate and parse the SPOT output matrix for one target.

    Input:
        target_id: RNA target identifier.
        length: Expected RNA sequence length.
        spot_outputs_dir: Directory containing SPOT-RNA-2D outputs.
    Output:
        Tuple of (matrix, source_path), or (None, None) when missing.
    """
    spot_outputs_dir = Path(spot_outputs_dir)
    if not spot_outputs_dir.exists():
        return None, None

    suffixes = {".npy", ".txt", ".csv", ".tsv", ".out", ".prob", ".prob_single", ".prob_profile", ".map"}
    candidates = [
        path
        for path in spot_outputs_dir.rglob("*")
        if path.is_file()
        and target_id.lower() in str(path).lower()
        and path.suffix.lower() in suffixes
    ]
    candidates.sort(key=lambda p: (len(p.name), str(p)))

    for path in candidates:
        matrix = load_candidate_matrix(path, length)
        if matrix is not None:
            return matrix, path

    return None, None


def convert_outputs_to_npy(sequences, spot_outputs_dir, out_dir, overwrite=False):
    """Purpose: Convert SPOT output matrices into cleaned .npy contact maps.

    Input:
        sequences: List of (target_id, sequence) tuples.
        spot_outputs_dir: Directory containing raw SPOT-RNA-2D outputs.
        out_dir: Directory for cleaned .npy contact maps.
        overwrite: Whether to overwrite existing .npy files.
    Output:
        Tuple of (converted_count, missing_target_ids).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    missing = []
    for target_id, seq in sequences:
        out_path = out_dir / f"{target_id}.npy"
        if out_path.exists() and not overwrite:
            converted += 1
            continue

        matrix, source = find_spot_output_matrix(target_id, len(seq), spot_outputs_dir)
        if matrix is None:
            missing.append(target_id)
            continue

        matrix = clean_contact_map(matrix)
        np.save(out_path, matrix)
        converted += 1
        print(f"Saved {out_path} from {source}")

    return converted, missing


def clean_contact_map(matrix):
    """Purpose: Normalize a contact map into a safe probability-like matrix.

    Input:
        matrix: Raw contact matrix.
    Output:
        Float32 matrix clipped to [0, 1], symmetrized, and zero-diagonal.
    """
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, 1.0)
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float32)


def parse_args():
    """Purpose: Parse command-line options for SPOT contact-map generation.

    Input:
        Command-line arguments.
    Output:
        argparse Namespace with input, output, and runtime settings.
    """
    parser = argparse.ArgumentParser(
        description="Generate training-ready .npy SPOT-RNA-2D contact maps."
    )
    parser.add_argument(
        "--sequence-csv",
        action="append",
        dest="sequence_csvs",
        default=None,
        help="CSV with target_id and sequence columns. Can be provided multiple times.",
    )
    parser.add_argument("--spot-repo", default=None, help="Path to cloned SPOT-RNA-2D repo.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-feats-dir", default=DEFAULT_INPUT_FEATS_DIR)
    parser.add_argument("--spot-outputs-dir", default=DEFAULT_SPOT_OUTPUTS_DIR)
    parser.add_argument("--list-path", default=os.path.join(DATA_DIR, "spot_rna_ids.txt"))
    parser.add_argument("--python", default=sys.executable, help="Python executable for SPOT-RNA-2D.")
    parser.add_argument("--conda-env", default=None, help="Conda environment used to run SPOT-RNA-2D.")
    parser.add_argument("--gpu", default="-1")
    parser.add_argument("--cpu", default="16")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument(
        "--use-empty-rnafold-prob",
        action="store_true",
        help="Write empty *_dp.ps placeholders so SPOT-RNA-2D skips external RNAfold.",
    )
    return parser.parse_args()


def main():
    """Purpose: Prepare SPOT inputs, optionally run SPOT, and convert outputs.

    Input:
        Command-line arguments parsed by parse_args().
    Output:
        None. Writes cleaned .npy contact maps or exits on missing outputs.
    """
    args = parse_args()
    csvs = args.sequence_csvs or [
        os.path.join(DATA_DIR, "train_sequences.csv"),
        os.path.join(DATA_DIR, "validation_sequences.csv"),
        os.path.join(DATA_DIR, "test_sequences.csv"),
    ]
    sequences = read_sequences(csvs, max_len=args.max_len)
    print(f"Loaded {len(sequences)} unique sequences.")

    if not args.convert_only:
        write_spot_inputs(
            sequences,
            args.input_feats_dir,
            args.list_path,
            use_empty_rnafold_prob=args.use_empty_rnafold_prob,
        )
        print(f"Wrote SPOT inputs to {args.input_feats_dir}")
        print(f"Wrote RNA id list to {args.list_path}")

    if args.prepare_only:
        return

    Path(args.spot_outputs_dir).mkdir(parents=True, exist_ok=True)
    if not args.convert_only:
        if args.spot_repo is None:
            raise ValueError("--spot-repo is required unless --convert-only is used")
        run_spot_single(args, args.list_path)

    converted, missing = convert_outputs_to_npy(
        sequences,
        args.spot_outputs_dir,
        args.out_dir,
        overwrite=args.overwrite,
    )
    print(f"Converted/found {converted}/{len(sequences)} maps in {args.out_dir}")
    if missing:
        print("Missing maps for:")
        for target_id in missing:
            print(target_id)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
