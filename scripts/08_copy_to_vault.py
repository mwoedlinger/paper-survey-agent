#!/usr/bin/env python3
"""
Step 8: Copy the staged literature tree into the configured Obsidian vault.

Mirrors ``staging/literature/`` into ``<vault>/literature/`` using rsync so
the operation is incremental, deletion-safe (only with --delete), and prints
exactly what would change in --dry-run mode.

Usage:
    python scripts/08_copy_to_vault.py --config config.yaml --dry-run
    python scripts/08_copy_to_vault.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Show changes without copying")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete vault files that no longer exist in staging (DESTRUCTIVE)",
    )
    parser.add_argument(
        "--checkpoint",
        default="work/checkpoint.json",
        help="Checkpoint file to mark vault_copied=True on success",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    staging = Path(config.get("staging_dir", "./staging")).resolve()
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH") or config["obsidian_vault_path"]
    vault = Path(vault_path).expanduser()
    src = staging / "literature"
    dst = vault / "literature"

    if not src.exists():
        sys.exit(f"Staging tree not found: {src}")
    if not vault.exists():
        sys.exit(f"Vault path does not exist: {vault}")

    if not shutil.which("rsync"):
        sys.exit("rsync is required but not found in PATH")

    cmd = ["rsync", "-av"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.delete:
        cmd.append("--delete")
    # Trailing slash on src copies *contents* into dst.
    cmd += [f"{src}/", f"{dst}/"]

    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    if args.dry_run:
        print("\n[dry-run] No files were copied.")
        return

    # Mark checkpoint
    cp_path = Path(args.checkpoint)
    if cp_path.exists():
        try:
            data = json.loads(cp_path.read_text())
            data["vault_copied"] = True
            cp_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Warning: failed to update checkpoint: {e}", file=sys.stderr)

    print(f"\n✓ Copied {src} → {dst}")


if __name__ == "__main__":
    main()
