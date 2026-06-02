import copy
import os
import re

import yaml


DEFAULT_CONFIG = {
    "run": {
        "mode": "train",
        "init_model_path": None,
        "output_root": "output",
        "output_dir": "auto",
        "save_path": None,
    },
    "data": {
        "data_dir": "dataset_final",
        "msa_dir": os.path.join("dataset_final", "MSA"),
        "val_labels_path": "validation_labels_new.normalized.csv",
        "contact_map_dir": os.path.join("dataset_final", "spot_maps"),
        "min_train_valid_labels": None,
        "min_train_label_coverage": None,
        "max_train_targets": None,
        "train_selection": "quality_diverse",
        "train_seed": 13,
        "train_length_bin_size": 50,
        "train_structure_anchor_groups": {},
        "train_targets_per_group": 150,
        "train_similarity_top_k_per_group": 300,
        "extra_train_data_dirs": [],
        "max_extra_train_targets": 0,
        "extra_train_selection": "quality_diverse",
        "extra_train_seed": 13,
        "extra_train_length_bin_size": 50,
        "structure_reinforcement_extra_targets": 0,
        "structure_reinforcement_anchor_groups": {},
        "structure_reinforcement_targets_per_group": None,
        "exclude_base_sequence_overlap": True,
        "exclude_validation_sequence_overlap": True,
    },
    "training": {
        "epochs": 40,
        "batch_size": 32,
        "lr": 3e-4,
        "patience": 7,
        "max_len": 256,
        "weight_decay": 1e-4,
    },
    "model": {
        "spot_bias_scale": 1.0,
        "use_graph": False,
        "graph_layers": 0,
        "graph_scale": 0.10,
        "spot_edge_threshold": 0.50,
        "spot_top_k": 8,
        "local_edge_max_sep": 4,
        "coord_refine_steps": 0,
        "coord_refine_hidden": 128,
        "coord_refine_dropout": 0.05,
        "coord_refine_local_window": 4,
        "coord_refine_delta_scale": 0.10,
    },
    "losses": {
        "raw_loss_weight": 0.05,
        "aligned_loss_weight": 1.0,
        "bond_loss_weight": 0.30,
        "spot_loss_weight": 0.10,
        "distmap_loss_weight": 0.30,
        "adj_loss_weight": 0.0,
        "short_range_loss_weight": 0.0,
        "medium_range_loss_weight": 0.0,
        "curvature_loss_weight": 0.0,
        "spread_loss_weight": 0.0,
        "bond_lower": 0.30,
        "bond_upper": 1.20,
        "short_range_max_sep": 4,
        "medium_range_min_sep": 5,
        "medium_range_max_sep": 12,
    },
    "metrics": {
        "accuracy_thresholds": [0.5, 0.75, 1.0],
    },
}

CLI_CONFIG_PATHS = {
    "mode": ("run", "mode"),
    "init_model_path": ("run", "init_model_path"),
    "output_root": ("run", "output_root"),
    "output_dir": ("run", "output_dir"),
    "save_path": ("run", "save_path"),
    "data_dir": ("data", "data_dir"),
    "msa_dir": ("data", "msa_dir"),
    "val_labels_path": ("data", "val_labels_path"),
    "contact_map_dir": ("data", "contact_map_dir"),
    "min_train_valid_labels": ("data", "min_train_valid_labels"),
    "min_train_label_coverage": ("data", "min_train_label_coverage"),
    "max_train_targets": ("data", "max_train_targets"),
    "train_selection": ("data", "train_selection"),
    "train_seed": ("data", "train_seed"),
    "train_length_bin_size": ("data", "train_length_bin_size"),
    "train_targets_per_group": ("data", "train_targets_per_group"),
    "train_similarity_top_k_per_group": ("data", "train_similarity_top_k_per_group"),
    "extra_train_data_dirs": ("data", "extra_train_data_dirs"),
    "max_extra_train_targets": ("data", "max_extra_train_targets"),
    "extra_train_selection": ("data", "extra_train_selection"),
    "extra_train_seed": ("data", "extra_train_seed"),
    "extra_train_length_bin_size": ("data", "extra_train_length_bin_size"),
    "structure_reinforcement_extra_targets": ("data", "structure_reinforcement_extra_targets"),
    "structure_reinforcement_targets_per_group": ("data", "structure_reinforcement_targets_per_group"),
    "epochs": ("training", "epochs"),
    "batch_size": ("training", "batch_size"),
    "lr": ("training", "lr"),
    "patience": ("training", "patience"),
    "max_len": ("training", "max_len"),
    "weight_decay": ("training", "weight_decay"),
    "spot_bias_scale": ("model", "spot_bias_scale"),
    "use_graph": ("model", "use_graph"),
    "graph_layers": ("model", "graph_layers"),
    "graph_scale": ("model", "graph_scale"),
    "spot_edge_threshold": ("model", "spot_edge_threshold"),
    "spot_top_k": ("model", "spot_top_k"),
    "local_edge_max_sep": ("model", "local_edge_max_sep"),
    "coord_refine_steps": ("model", "coord_refine_steps"),
    "coord_refine_hidden": ("model", "coord_refine_hidden"),
    "coord_refine_dropout": ("model", "coord_refine_dropout"),
    "coord_refine_local_window": ("model", "coord_refine_local_window"),
    "coord_refine_delta_scale": ("model", "coord_refine_delta_scale"),
    "raw_loss_weight": ("losses", "raw_loss_weight"),
    "aligned_loss_weight": ("losses", "aligned_loss_weight"),
    "bond_loss_weight": ("losses", "bond_loss_weight"),
    "spot_loss_weight": ("losses", "spot_loss_weight"),
    "distmap_loss_weight": ("losses", "distmap_loss_weight"),
    "adj_loss_weight": ("losses", "adj_loss_weight"),
    "short_range_loss_weight": ("losses", "short_range_loss_weight"),
    "medium_range_loss_weight": ("losses", "medium_range_loss_weight"),
    "curvature_loss_weight": ("losses", "curvature_loss_weight"),
    "spread_loss_weight": ("losses", "spread_loss_weight"),
    "bond_lower": ("losses", "bond_lower"),
    "bond_upper": ("losses", "bond_upper"),
    "short_range_max_sep": ("losses", "short_range_max_sep"),
    "medium_range_min_sep": ("losses", "medium_range_min_sep"),
    "medium_range_max_sep": ("losses", "medium_range_max_sep"),
    "accuracy_thresholds": ("metrics", "accuracy_thresholds"),
}

OUTPUT_DIR_RE = re.compile(r"^output(\d+)$")
FINETUNE_DIR_RE = re.compile(r"^Finetune_output(\d+)(?:_v(\d+))?(?:_run(\d+))?$")
GRAPH_FINETUNE_DIR_RE = re.compile(r"^Graph_Finetune_output(\d+)(?:_v(\d+))?(?:_run(\d+))?$")
SAFE_RUN_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def deep_update(base, updates):
    """Purpose: Recursively merge one dictionary into another.

    Input:
        base: Dictionary to update in place.
        updates: Dictionary containing replacement values.
    Output:
        Updated base dictionary.
    """
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(config_path):
    """Purpose: Read a YAML configuration file.

    Input:
        config_path: Path to a YAML file, or None.
    Output:
        Parsed configuration dictionary.
    """
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def set_nested(config, path, value):
    """Purpose: Set a nested config value by path.

    Input:
        config: Configuration dictionary to mutate.
        path: Tuple of nested keys.
        value: Value to assign.
    Output:
        None. The config dictionary is updated in place.
    """
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def cli_overrides_from_args(args):
    """Purpose: Convert explicitly supplied argparse values into config updates.

    Input:
        args: argparse Namespace with None for options not supplied by the user.
    Output:
        Nested dictionary containing only explicit CLI overrides.
    """
    overrides = {}
    for arg_name, path in CLI_CONFIG_PATHS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            set_nested(overrides, path, value)
    return overrides


def build_config(config_path=None, cli_overrides=None):
    """Purpose: Merge defaults, YAML values, and explicit CLI overrides.

    Input:
        config_path: Optional YAML config path.
        cli_overrides: Optional nested dictionary of CLI overrides.
    Output:
        Fully merged configuration dictionary.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    deep_update(config, load_yaml_config(config_path))
    deep_update(config, cli_overrides or {})
    return config


def is_auto_output_dir(output_dir):
    """Purpose: Decide whether a run directory should be auto-generated.

    Input:
        output_dir: Output directory value from config or CLI.
    Output:
        True when output_dir is empty or set to "auto".
    """
    return output_dir is None or str(output_dir).lower() == "auto"


def next_output_dir(output_root):
    """Purpose: Find the next numbered scratch-training output directory.

    Input:
        output_root: Parent output directory.
    Output:
        Path like output/output3.
    """
    os.makedirs(output_root, exist_ok=True)
    max_idx = 0
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        match = OUTPUT_DIR_RE.match(name)
        if match and os.path.isdir(path):
            max_idx = max(max_idx, int(match.group(1)))
    return os.path.join(output_root, f"output{max_idx + 1}")


def checkpoint_parent_name(init_model_path):
    """Purpose: Create a readable run-name suffix from a checkpoint parent directory.

    Input:
        init_model_path: Path to a checkpoint file.
    Output:
        Filesystem-safe parent directory name, or "checkpoint" when unavailable.
    """
    parent_name = os.path.basename(os.path.dirname(os.path.normpath(init_model_path)))
    if not parent_name:
        return "checkpoint"
    return SAFE_RUN_NAME_RE.sub("_", parent_name)


def finetune_source_name(init_model_path):
    """Purpose: Derive a clear fine-tune output name from a checkpoint path.

    Input:
        init_model_path: Path to a checkpoint under an output directory.
    Output:
        Directory label such as "Finetune_output2_v3".
    """
    for part in reversed(os.path.normpath(init_model_path).split(os.sep)):
        if OUTPUT_DIR_RE.match(part):
            return f"Finetune_{part}"
        match = FINETUNE_DIR_RE.match(part)
        if match:
            output_idx, version, _ = match.groups()
            next_version = int(version) + 1 if version is not None else 3
            return f"Finetune_output{output_idx}_v{next_version}"
    return f"Finetune_{checkpoint_parent_name(init_model_path)}"


def unique_finetune_dir(output_root, init_model_path):
    """Purpose: Create a non-conflicting fine-tune output directory name.

    Input:
        output_root: Parent output directory.
        init_model_path: Checkpoint path used to derive the fine-tune label.
    Output:
        Path like output/Finetune_output2_v3 or output/Finetune_output2_v3_run2.
    """
    os.makedirs(output_root, exist_ok=True)
    source_name = finetune_source_name(init_model_path)
    base = os.path.join(output_root, source_name)
    if not os.path.exists(base):
        return base

    run_idx = 2
    while True:
        candidate = f"{base}_run{run_idx}"
        if not os.path.exists(candidate):
            return candidate
        run_idx += 1


def graph_finetune_source_name(init_model_path):
    """Purpose: Derive a clear graph fine-tune output name from a checkpoint path.

    Input:
        init_model_path: Path to a checkpoint under an output or fine-tune directory.
    Output:
        Directory label such as "Graph_Finetune_output2".
    """
    for part in reversed(os.path.normpath(init_model_path).split(os.sep)):
        output_match = OUTPUT_DIR_RE.match(part)
        if output_match:
            return f"Graph_Finetune_output{output_match.group(1)}"
        finetune_match = FINETUNE_DIR_RE.match(part)
        if finetune_match:
            output_idx, version, _ = finetune_match.groups()
            if version is not None:
                return f"Graph_Finetune_output{output_idx}_v{version}"
            return f"Graph_Finetune_output{output_idx}"
        graph_match = GRAPH_FINETUNE_DIR_RE.match(part)
        if graph_match:
            output_idx, version, _ = graph_match.groups()
            if version is not None:
                return f"Graph_Finetune_output{output_idx}_v{version}"
            return f"Graph_Finetune_output{output_idx}"
    return f"Graph_Finetune_{checkpoint_parent_name(init_model_path)}"


def unique_graph_finetune_dir(output_root, init_model_path):
    """Purpose: Create a non-conflicting graph fine-tune output directory name.

    Input:
        output_root: Parent output directory.
        init_model_path: Checkpoint path used to derive the graph fine-tune label.
    Output:
        Path like output/Graph_Finetune_output2 or output/Graph_Finetune_output2_run2.
    """
    os.makedirs(output_root, exist_ok=True)
    source_name = graph_finetune_source_name(init_model_path)
    base = os.path.join(output_root, source_name)
    if not os.path.exists(base):
        return base

    run_idx = 2
    while True:
        candidate = f"{base}_run{run_idx}"
        if not os.path.exists(candidate):
            return candidate
        run_idx += 1


def graph_enabled(config):
    """Purpose: Check whether the current configuration enables graph layers.

    Input:
        config: Effective configuration dictionary.
    Output:
        True when model.use_graph is true and graph_layers is positive.
    """
    model = config.get("model", {})
    return bool(model.get("use_graph")) and int(model.get("graph_layers", 0)) > 0


def resolve_output_dir(config):
    """Purpose: Resolve the output directory for train or fine-tune mode.

    Input:
        config: Merged configuration dictionary.
    Output:
        Output directory path.
    """
    run_cfg = config["run"]
    output_root = run_cfg.get("output_root") or DEFAULT_CONFIG["run"]["output_root"]
    output_dir = run_cfg.get("output_dir")
    save_path = run_cfg.get("save_path")
    mode = run_cfg.get("mode", "train")

    if not is_auto_output_dir(output_dir):
        return output_dir
    if save_path:
        return os.path.dirname(save_path) or "."
    if mode == "finetune":
        init_model_path = run_cfg.get("init_model_path")
        if not init_model_path:
            raise ValueError("run.init_model_path is required for fine-tuning.")
        if graph_enabled(config):
            return unique_graph_finetune_dir(output_root, init_model_path)
        return unique_finetune_dir(output_root, init_model_path)
    return next_output_dir(output_root)


def config_with_runtime_paths(config, output_dir, save_path):
    """Purpose: Record resolved runtime paths in a copy of the config.

    Input:
        config: Merged configuration dictionary.
        output_dir: Resolved output directory.
        save_path: Resolved best checkpoint path.
    Output:
        Config copy with run.output_dir and run.save_path updated.
    """
    resolved = copy.deepcopy(config)
    resolved.setdefault("run", {})["output_dir"] = output_dir
    resolved["run"]["save_path"] = save_path
    return resolved


def write_used_config(config, output_dir):
    """Purpose: Save the effective run configuration next to outputs.

    Input:
        config: Effective configuration dictionary.
        output_dir: Run output directory.
    Output:
        Path to the written YAML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "used_config.yaml")
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    return config_path


def print_run_summary(config, output_dir):
    """Purpose: Print the resolved run settings for dry-runs and real runs.

    Input:
        config: Effective configuration dictionary.
        output_dir: Resolved output directory.
    Output:
        None. Writes a readable summary to stdout.
    """
    run = config["run"]
    training = config["training"]
    model = config["model"]
    losses = config["losses"]
    metrics = config["metrics"]

    print("Resolved run configuration:")
    print(f"  mode: {run.get('mode')}")
    print(f"  init_model_path: {run.get('init_model_path')}")
    print(f"  output_dir: {output_dir}")
    print(
        "  data filters: "
        f"min_train_valid_labels={config['data'].get('min_train_valid_labels')}, "
        f"min_train_label_coverage={config['data'].get('min_train_label_coverage')}, "
        f"max_train_targets={config['data'].get('max_train_targets')}, "
        f"train_selection={config['data'].get('train_selection')}, "
        f"train_targets_per_group={config['data'].get('train_targets_per_group')}, "
        f"train_similarity_top_k_per_group={config['data'].get('train_similarity_top_k_per_group')}"
    )
    print(f"  contact_map_dir: {config['data'].get('contact_map_dir')}")
    print(
        "  extra train data: "
        f"dirs={config['data'].get('extra_train_data_dirs')}, "
        f"max_targets={config['data'].get('max_extra_train_targets')}, "
        f"selection={config['data'].get('extra_train_selection')}, "
        f"seed={config['data'].get('extra_train_seed')}"
    )
    print(
        "  structure reinforcement: "
        f"extra_targets={config['data'].get('structure_reinforcement_extra_targets')}, "
        f"targets_per_group={config['data'].get('structure_reinforcement_targets_per_group')}, "
        f"anchor_groups={config['data'].get('structure_reinforcement_anchor_groups')}"
    )
    print(
        "  training: "
        f"epochs={training.get('epochs')}, batch_size={training.get('batch_size')}, "
        f"lr={training.get('lr')}, patience={training.get('patience')}, "
        f"max_len={training.get('max_len')}, weight_decay={training.get('weight_decay')}"
    )
    print("  model:")
    for key in [
        "spot_bias_scale",
        "use_graph",
        "graph_layers",
        "graph_scale",
        "spot_edge_threshold",
        "spot_top_k",
        "local_edge_max_sep",
        "coord_refine_steps",
        "coord_refine_hidden",
        "coord_refine_dropout",
        "coord_refine_local_window",
        "coord_refine_delta_scale",
    ]:
        print(f"    {key}: {model.get(key)}")
    print("  losses:")
    for key in [
        "raw_loss_weight",
        "aligned_loss_weight",
        "bond_loss_weight",
        "spot_loss_weight",
        "distmap_loss_weight",
        "adj_loss_weight",
        "short_range_loss_weight",
        "medium_range_loss_weight",
        "curvature_loss_weight",
        "spread_loss_weight",
        "bond_lower",
        "bond_upper",
        "short_range_max_sep",
        "medium_range_min_sep",
        "medium_range_max_sep",
    ]:
        print(f"    {key}: {losses.get(key)}")
    print(f"  accuracy_thresholds: {metrics.get('accuracy_thresholds')}")


def add_shared_training_args(parser, include_mode=True):
    """Purpose: Add shared YAML/CLI training options to an argparse parser.

    Input:
        parser: argparse parser to update.
        include_mode: Whether to expose the run mode CLI override.
    Output:
        The same parser, updated with shared options.
    """
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    if include_mode:
        parser.add_argument("--mode", default=None)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--msa-dir", default=None)
    parser.add_argument("--val-labels-path", default=None)
    parser.add_argument("--contact-map-dir", default=None)
    parser.add_argument("--min-train-valid-labels", type=int, default=None)
    parser.add_argument("--min-train-label-coverage", type=float, default=None)
    parser.add_argument("--max-train-targets", type=int, default=None)
    parser.add_argument("--train-selection", default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--train-length-bin-size", type=int, default=None)
    parser.add_argument("--train-targets-per-group", type=int, default=None)
    parser.add_argument("--train-similarity-top-k-per-group", type=int, default=None)
    parser.add_argument("--extra-train-data-dirs", nargs="*", default=None)
    parser.add_argument("--max-extra-train-targets", type=int, default=None)
    parser.add_argument("--extra-train-selection", default=None)
    parser.add_argument("--extra-train-seed", type=int, default=None)
    parser.add_argument("--extra-train-length-bin-size", type=int, default=None)
    parser.add_argument("--structure-reinforcement-extra-targets", type=int, default=None)
    parser.add_argument("--structure-reinforcement-targets-per-group", type=int, default=None)
    parser.add_argument("--spot-bias-scale", type=float, default=None)
    parser.add_argument("--use-graph", action="store_true", default=None)
    parser.add_argument("--graph-layers", type=int, default=None)
    parser.add_argument("--graph-scale", type=float, default=None)
    parser.add_argument("--spot-edge-threshold", type=float, default=None)
    parser.add_argument("--spot-top-k", type=int, default=None)
    parser.add_argument("--local-edge-max-sep", type=int, default=None)
    parser.add_argument("--coord-refine-steps", type=int, default=None)
    parser.add_argument("--coord-refine-hidden", type=int, default=None)
    parser.add_argument("--coord-refine-dropout", type=float, default=None)
    parser.add_argument("--coord-refine-local-window", type=int, default=None)
    parser.add_argument("--coord-refine-delta-scale", type=float, default=None)
    parser.add_argument("--spot-loss-weight", type=float, default=None)
    parser.add_argument("--raw-loss-weight", type=float, default=None)
    parser.add_argument("--aligned-loss-weight", type=float, default=None)
    parser.add_argument("--bond-loss-weight", type=float, default=None)
    parser.add_argument("--distmap-loss-weight", type=float, default=None)
    parser.add_argument("--adj-loss-weight", type=float, default=None)
    parser.add_argument("--short-range-loss-weight", type=float, default=None)
    parser.add_argument("--medium-range-loss-weight", type=float, default=None)
    parser.add_argument("--curvature-loss-weight", type=float, default=None)
    parser.add_argument("--spread-loss-weight", type=float, default=None)
    parser.add_argument("--bond-lower", type=float, default=None)
    parser.add_argument("--bond-upper", type=float, default=None)
    parser.add_argument("--short-range-max-sep", type=int, default=None)
    parser.add_argument("--medium-range-min-sep", type=int, default=None)
    parser.add_argument("--medium-range-max-sep", type=int, default=None)
    parser.add_argument("--accuracy-thresholds", type=float, nargs="+", default=None)
    return parser
