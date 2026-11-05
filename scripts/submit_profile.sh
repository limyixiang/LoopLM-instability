#!/usr/bin/env bash
# Same saved settings/optional account behavior as submit_reproduction.sh.
set -euo pipefail
cd "${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source scripts/load_cluster_env.sh
submit_args=(--export=ALL --chdir="$REPO_DIR")
if [[ -n "${PILOT_ACCOUNT:-}" ]]; then
    submit_args+=(--account="$PILOT_ACCOUNT")
fi
if [[ -n "${PILOT_PARTITION:-}" ]]; then
    submit_args+=(--partition="$PILOT_PARTITION")
fi
mkdir -p "$REPO_DIR/logs"
exec sbatch "${submit_args[@]}" "$@" "$REPO_DIR/scripts/profile_reproduction.sh"
