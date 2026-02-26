#!/usr/bin/env python3
"""
parse_config.py — Read YAML experiment configs, merge overrides, and output
in formats consumable by shell scripts (flat), W&B (JSON), and log files (header).

Usage examples:
    # Flat output for shell consumption (verl CLI args)
    python parse_config.py configs/base/grpo_7b.yaml --format flat

    # With overrides
    python parse_config.py configs/base/grpo_7b.yaml --format flat \
        --override actor.optim.lr=5e-7 --override rollout.n=16

    # W&B JSON config
    python parse_config.py configs/base/grpo_7b.yaml --format wandb

    # Log header
    python parse_config.py configs/base/grpo_7b.yaml --format header \
        --run_name my_run --gpus 0,1,2,3 --num_gpus 4

    # Sweep mode
    python parse_config.py --sweep configs/sweeps/lr_sweep.yaml

Dependencies: pyyaml + stdlib only.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Prefix mapping: short keys in our YAML → verl CLI prefixes
# ---------------------------------------------------------------------------
# Keys under these top-level sections get rewritten when producing verl CLI args.
# actor.*, rollout.*, ref.*, model.* → actor_rollout_ref.<section>.*
# Everything else passes through as-is.
VERL_PREFIX_MAP = {
    "actor":   "actor_rollout_ref.actor",
    "rollout": "actor_rollout_ref.rollout",
    "ref":     "actor_rollout_ref.ref",
    "model":   "actor_rollout_ref.model",
}

# Note: top-level keys not in VERL_PREFIX_MAP pass through as-is automatically
# (see to_verl_key). No explicit passthrough set is needed.


# ---------------------------------------------------------------------------
# Value parsing: string → bool / int / float / None / string
# ---------------------------------------------------------------------------
def parse_value(v):
    """Coerce a string value to the most specific Python type possible."""
    if not isinstance(v, str):
        return v

    # Booleans
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False

    # None / null
    if v.lower() in ("none", "null", "~"):
        return None

    # Integer (but not scientific notation like 1e6)
    if "e" not in v.lower() and "." not in v:
        try:
            return int(v)
        except ValueError:
            pass

    # Float (includes scientific notation like 1e-6, 5e-7, 3.14)
    try:
        return float(v)
    except ValueError:
        pass

    return v


# ---------------------------------------------------------------------------
# YAML loading + flattening
# ---------------------------------------------------------------------------
def load_yaml(path):
    """Load a YAML file, returning a dict (empty dict if file is empty)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dict, coercing string values.

    Example: {actor: {optim: {lr: 1e-6}}} → {"actor.optim.lr": 1e-6}

    String values that look like bool/int/float/None are coerced to native
    Python types (YAML safe_load sometimes leaves scientific notation like
    ``1e-6`` as strings).
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict) and v:
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, parse_value(v) if isinstance(v, str) else v))
    return dict(items)


# ---------------------------------------------------------------------------
# Prefix mapping (short key → verl CLI key)
# ---------------------------------------------------------------------------
def to_verl_key(short_key):
    """Map a short dotted key to the verl CLI key.

    actor.optim.lr        → actor_rollout_ref.actor.optim.lr
    data.train_batch_size → data.train_batch_size  (passthrough)
    """
    parts = short_key.split(".")
    top = parts[0]

    if top in VERL_PREFIX_MAP:
        return VERL_PREFIX_MAP[top] + "." + ".".join(parts[1:]) if len(parts) > 1 else VERL_PREFIX_MAP[top]
    # Passthrough sections and anything unrecognized keep their key as-is
    return short_key


# ---------------------------------------------------------------------------
# Override merging
# ---------------------------------------------------------------------------
def apply_overrides(flat_config, overrides):
    """Apply override strings (key=value) on top of the flat config.

    Overrides use our *short* keys (e.g. actor.optim.lr=5e-7), not verl
    prefixed keys.
    """
    for override in overrides:
        if "=" not in override:
            print(f"WARNING: Ignoring malformed override (no '='): {override}", file=sys.stderr)
            continue
        key, _, value = override.partition("=")
        key = key.strip()
        value = value.strip()
        flat_config[key] = parse_value(value)
    return flat_config


# ---------------------------------------------------------------------------
# Value serialization for output
# ---------------------------------------------------------------------------
def serialize_value(v):
    """Convert a Python value to a string suitable for shell / verl CLI."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        # :g format preserves scientific notation for very small/large values
        return f"{v:g}"
    if isinstance(v, list):
        # verl expects list-like syntax, e.g. ['console','wandb']
        return repr(v)
    return str(v)


# ---------------------------------------------------------------------------
# Output: flat (shell / verl CLI args)
# ---------------------------------------------------------------------------
def output_flat(flat_config):
    """Print one verl_key=value per line, sorted by verl key."""
    pairs = []
    for short_key, val in flat_config.items():
        verl_key = to_verl_key(short_key)
        pairs.append((verl_key, serialize_value(val)))
    pairs.sort(key=lambda p: p[0])
    print("\n".join(f"{k}={v}" for k, v in pairs))


# ---------------------------------------------------------------------------
# Output: wandb (JSON of original / short keys)
# ---------------------------------------------------------------------------
def output_wandb(flat_config):
    """Print a JSON dict of original (non-prefixed) keys for W&B config upload."""
    # Convert values for JSON serialization
    wandb_dict = {}
    for k, v in sorted(flat_config.items()):
        # JSON can handle native types; just ensure no Python-specific oddities
        if isinstance(v, float) and (v != v):  # NaN check
            wandb_dict[k] = None
        else:
            wandb_dict[k] = v
    print(json.dumps(wandb_dict, indent=2, default=str))


# ---------------------------------------------------------------------------
# Output: header (formatted metadata for log files)
# ---------------------------------------------------------------------------
def get_git_commit():
    """Return the short git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def safe_get(flat_config, key, default="N/A"):
    """Safely retrieve a value from the flat config, returning default if missing.

    Uses ``key in`` check so that explicit None/null values are preserved
    (serialized as "null") rather than being conflated with missing keys.
    """
    if key not in flat_config:
        return default
    return serialize_value(flat_config[key])


def output_header(flat_config, run_name=None, gpus=None, num_gpus=None,
                  config_path=None, overrides=None):
    """Print a formatted metadata block for log file headers."""
    # Resolve values with graceful fallbacks
    run_name = run_name or "unnamed_run"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_path = safe_get(flat_config, "model.path")
    algorithm_name = safe_get(flat_config, "algorithm.adv_estimator",
                              safe_get(flat_config, "algorithm.name", "N/A"))
    num_gpus_str = str(num_gpus) if num_gpus is not None else "N/A"
    gpus_str = gpus if gpus else "N/A"
    config_path_str = config_path or "N/A"
    overrides_str = ", ".join(overrides) if overrides else "none"
    git_commit = get_git_commit()

    # Key params with graceful fallbacks
    lr = safe_get(flat_config, "actor.optim.lr")
    kl_loss_coef = safe_get(flat_config, "actor.kl_loss_coef")
    rollout_n = safe_get(flat_config, "rollout.n")
    batch_size = safe_get(flat_config, "data.train_batch_size")
    max_prompt = safe_get(flat_config, "data.max_prompt_length")
    max_response = safe_get(flat_config, "data.max_response_length")
    temperature = safe_get(flat_config, "rollout.temperature")
    entropy_coeff = safe_get(flat_config, "actor.entropy_coeff")

    header = f"""\
=============================================
Run: {run_name}
Date: {date_str}
Model: {model_path}
Algorithm: {algorithm_name}
GPUs: {num_gpus_str} (CUDA_VISIBLE_DEVICES={gpus_str})
Config: {config_path_str}
Overrides: {overrides_str}
Key params:
  lr={lr}, kl_loss_coef={kl_loss_coef}, rollout_n={rollout_n}
  batch_size={batch_size}, max_prompt={max_prompt}, max_response={max_response}
  temperature={temperature}, entropy_coeff={entropy_coeff}
Git commit: {git_commit}
============================================="""
    print(header)


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------
def output_sweep(sweep_path):
    """Read a sweep config and output one param=value per line.

    Expected sweep config format:
        sweep_name: "lr_sweep"
        sweep_param: "actor.optim.lr"
        values:
          - 5e-7
          - 1e-6
    """
    sweep = load_yaml(sweep_path)

    if "sweep_param" not in sweep or "values" not in sweep:
        print("ERROR: Sweep config must have 'sweep_param' and 'values' keys.",
              file=sys.stderr)
        sys.exit(1)

    param = sweep["sweep_param"]
    values = sweep["values"]

    for v in values:
        # Coerce string values (YAML safe_load leaves 5e-7 as strings)
        v = parse_value(v) if isinstance(v, str) else v
        print(f"{param}={serialize_value(v)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Parse YAML experiment configs for verl training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Output formats:
  flat    One verl_key=value per line (sorted), for shell consumption (default)
  wandb   JSON dict of original (non-prefixed) keys for W&B config upload
  header  Formatted metadata text block for log file headers

Examples:
  %(prog)s config.yaml --format flat
  %(prog)s config.yaml --format flat --override actor.optim.lr=5e-7
  %(prog)s config.yaml --format wandb
  %(prog)s config.yaml --format header --run_name my_run --gpus 0,1 --num_gpus 2
  %(prog)s --sweep sweeps/lr_sweep.yaml
""",
    )

    parser.add_argument("config", nargs="?", default=None,
                        help="Path to the YAML experiment config file")
    parser.add_argument("--format", choices=["flat", "wandb", "header"],
                        default="flat", dest="output_format",
                        help="Output format (default: flat)")
    parser.add_argument("--override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override a config value (can be repeated). "
                             "Uses short keys, e.g. actor.optim.lr=5e-7")
    parser.add_argument("--sweep", default=None, metavar="SWEEP.yaml",
                        help="Sweep mode: read a sweep config and output "
                             "one param=value per line")

    # Header-specific args
    header_group = parser.add_argument_group("header format options")
    header_group.add_argument("--run_name", default=None,
                              help="Run name for header output")
    header_group.add_argument("--gpus", default=None,
                              help="CUDA_VISIBLE_DEVICES value for header output")
    header_group.add_argument("--num_gpus", default=None, type=int,
                              help="Number of GPUs for header output")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---- Sweep mode (independent of config file) ----
    if args.sweep:
        output_sweep(args.sweep)
        return

    # ---- Normal mode: config file required ----
    if not args.config:
        parser.error("config file is required (unless using --sweep)")

    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load and flatten
    raw_config = load_yaml(config_path)
    flat_config = flatten_dict(raw_config)

    # Apply overrides
    flat_config = apply_overrides(flat_config, args.override)

    # Output in the requested format
    if args.output_format == "flat":
        output_flat(flat_config)
    elif args.output_format == "wandb":
        output_wandb(flat_config)
    elif args.output_format == "header":
        output_header(
            flat_config,
            run_name=args.run_name,
            gpus=args.gpus,
            num_gpus=args.num_gpus,
            config_path=config_path,
            overrides=args.override,
        )


if __name__ == "__main__":
    main()
