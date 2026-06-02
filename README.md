# RNA 3D Folding Training Guide

This project trains an RNA 3D structure prediction model for the Stanford RNA
3D Folding task. The model combines sequence features, MSA features, ViennaRNA
pair features, and SPOT-RNA-2D contact maps, then predicts one 3D coordinate
for each RNA residue.

## Main Files

- `train and validate.py`: main training entry point.
- `training_config.py`: default config values, CLI overrides, and output path handling.
- `model_conv_attn.py`: model architecture.
- `DATASET.py`: dataset loading, feature construction, padding, and masking.
- `Datavisualization.py`: validation prediction, aligned prediction, diagnostics, and 3D plots.
- `configs/`: YAML experiment configs.
- `dataset_final/`: prepared training and validation data used by the current configs.

## Dependencies

Install the main Python dependencies before training:

```bash
pip install torch pandas numpy biopython matplotlib pyyaml tqdm
```

Optional dependencies are only needed when regenerating feature files:

- ViennaRNA Python bindings are needed by `base_pair.py`.
- SPOT-RNA-2D is needed only if you regenerate contact maps. The external tool is needed; you need to use https://github.com/jaswindersingh2/SPOT-RNA-2D

## Data Layout

The training configs expect `dataset_final/` to contain:

```text
dataset_final/
  train_sequences.csv
  train_labels.csv
  train_pair_features.csv
  validation_sequences.csv
  validation_labels.csv
  validation_labels_new.normalized.csv
  validation_pair_features.csv
  test_sequences.csv
  MSA/
    <target_id>.MSA.fasta
  spot_maps/
    <target_id>.npy
```

`DATASET.py` builds each residue feature vector from:

- nucleotide one-hot encoding,
- MSA conservation and depth features,
- ViennaRNA pair features,
- SPOT-RNA-2D contact maps.

Labels and padded residues are masked so missing coordinates do not contribute
to the training loss or validation metrics.

## Recommended Training Command

The reported `output3` style model uses graph message passing, SPOT contact
guidance, and direct coordinate prediction without coordinate refinement. To
train that model, run:

```bash
python "train and validate.py" --config configs/train_small_extra_structure_reinforcement.yaml
```

Because the script filename contains spaces, keep the quotes around
`"train and validate.py"`.

Before launching a full run, you can check the resolved config without training:

```bash
python "train and validate.py" --config configs/train_small_extra_structure_reinforcement.yaml --dry-run
```

## Other Useful Configs

- `configs/train_default.yaml`: baseline training config.
- `configs/train_small_clean.yaml`: smaller clean graph training run.
- `configs/train_small_extra_structure_reinforcement.yaml`: reported run.
- `configs/finetune_v3.yaml`: finetuning from a previous checkpoint.

You can override config values from the command line. For example:

```bash
python "train and validate.py" --config configs/train_small_extra_structure_reinforcement.yaml --epochs 20 --batch-size 32 --output-dir output_test
```

## What Happens During Training

1. The script loads train and validation CSV files from `data.data_dir`.
2. `RNADataset` constructs residue-level features, labels, masks, and contact maps.
3. The model predicts normalized residue coordinates with shape `(B, L, 3)`.
4. The loss combines Kabsch-aligned RMSE with geometry-aware distance/contact terms.
5. Validation uses Kabsch-aligned predictions so rotation and translation do not dominate the score.
6. Early stopping saves the best checkpoint according to validation loss.

## Training Outputs

Each run writes outputs under `run.output_root` and `run.output_dir`. With
`output_dir: auto`, the code creates the next available directory under
`output/`.

Typical output files are:

```text
output/<run_name>/
  used_config.yaml
  best_model.pth
  last_model.pth
  best_local_geometry.pth
  training_history.csv
  training_validation_loss.png
  validation_accuracy_acc_at_1p0.png
```

`used_config.yaml` is important because it records the final resolved config,
including any command-line overrides.

## Generate Predictions and 3D Plots

After training, run the visualization script on the best checkpoint:

```bash
python Datavisualization.py --model-path output/<run_name>/best_model.pth
```

This writes validation prediction files and plots into the same run directory
unless you pass a custom `--output-dir`.

Typical visualization outputs are:

```text
output/<run_name>/
  Predictions.csv
  Predictions_aligned.csv
  Predictions_aligned_angstrom.csv
  compactness_diagnostics.csv
  3d_plots/
```

## Rebuilding or Verifying the Dataset

The current repo already contains `dataset_final/`. If you need to verify the
prepared data layout, run:

```bash
python prepare_training_dataset.py --output-dir dataset_final --msa-dir dataset_final/MSA --contact-map-dir dataset_final/spot_maps --verify-only
```

Regenerating MSA files or SPOT maps may require external tools and is slower
than normal model training.

## Tests

Run the available unit tests with:

```bash
python -m unittest discover tests
```

The tests cover dataset masking and filtering behavior used by training and
visualization.
