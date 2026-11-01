#!/usr/bin/env bash
# Source from the repository root. Optional config; existing env-only use works.
export REPO_DIR="${REPO_DIR:-$PWD}"
if [[ -n "${CLUSTER_ENV_FILE:-}" ]]; then
    if [[ ! -f "$CLUSTER_ENV_FILE" ]]; then
        echo "Cluster configuration not found: $CLUSTER_ENV_FILE" >&2
        return 1
    fi
    source "$CLUSTER_ENV_FILE"
elif [[ -f "$REPO_DIR/cluster.env" ]]; then
    source "$REPO_DIR/cluster.env"
fi
