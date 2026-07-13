#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sudo $0 /sys/fs/cgroup/sglang-hicache" >&2
  exit 2
fi

cgroup_path=$1
parent_path=$(dirname "$cgroup_path")

if [[ $(stat -fc %T /sys/fs/cgroup) != cgroup2fs ]]; then
  echo "cgroup v2 is required" >&2
  exit 1
fi

if [[ -f "$parent_path/cgroup.subtree_control" ]] && ! grep -qw io "$parent_path/cgroup.subtree_control"; then
  echo +io > "$parent_path/cgroup.subtree_control"
fi
mkdir -p "$cgroup_path"
if [[ ! -w "$cgroup_path/io.max" ]]; then
  echo "$cgroup_path/io.max is not writable; run this script as root" >&2
  exit 1
fi
printf 'Created cgroup: %s\n' "$cgroup_path"
