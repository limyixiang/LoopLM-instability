#!/usr/bin/env bash
# Source from the repository root. Optional config; existing env-only use works.
export REPO_DIR="${REPO_DIR:-$PWD}"
# Array tasks receive a frozen experiment configuration after the batch wrapper
# loads site modules. Do not let later edits to cluster.env change queued cells.
if [[ "${OURO_ARRAY_FROZEN_ENV:-0}" == 1 ]]; then
    return 0
fi
if [[ -n "${CLUSTER_ENV_FILE:-}" ]]; then
    if [[ ! -f "$CLUSTER_ENV_FILE" ]]; then
        echo "Cluster configuration not found: $CLUSTER_ENV_FILE" >&2
        return 1
    fi
    source "$CLUSTER_ENV_FILE"
elif [[ -f "$REPO_DIR/cluster.env" ]]; then
    source "$REPO_DIR/cluster.env"
fi
