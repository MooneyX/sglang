#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sudo $0 /sys/fs/cgroup/sglang-hicache command [args...]" >&2
  exit 2
fi

cgroup_path=$1
shift

if [[ ! -d "$cgroup_path" || ! -w "$cgroup_path/cgroup.procs" ]]; then
  echo "$cgroup_path is missing or cgroup.procs is not writable" >&2
  exit 1
fi

echo $$ > "$cgroup_path/cgroup.procs"
exec "$@"
