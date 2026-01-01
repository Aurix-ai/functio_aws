#!/usr/bin/env bash
#
# Profiles a process by PID for the duration of another command execution
# Usage: sudo ./flame_folded_pid.sh --pid <PID> -- <command>
#

set -euo pipefail

PID=""
EXEC_CMD=""

# Parse --pid <PID> -- <command>
if [[ "$1" == "--pid" ]]; then
    PID="$2"
    shift 2
    if [[ "$1" == "--" ]]; then
        shift
        EXEC_CMD="$*"
    fi
fi

if [[ -z "$PID" || -z "$EXEC_CMD" ]]; then
    echo "Usage: sudo $0 --pid <PID> -- <command>" >&2
    exit 1
fi

PERF_DATA="perf.data"
FOLDED="flamegraph.folded"
FLG_DIR=$(pwd)

echo "[*] Recording perf data for PID: $PID while executing: $EXEC_CMD"

# Start perf recording in background
sudo perf record -F 999 -g --call-graph fp -o "$PERF_DATA" -p "$PID" &
PERF_PID=$!

# Execute the command
echo "[*] Executing command: $EXEC_CMD"
bash -c "$EXEC_CMD" || true

# Stop perf recording
kill $PERF_PID 2>/dev/null
wait $PERF_PID 2>/dev/null || true

echo "[*] Converting perf.data to folded format..."
perf script -i "$PERF_DATA" | "${FLG_DIR}/stackcollapse-perf.pl" > "$FOLDED"

echo "[✓] Folded flamegraph written to: $FOLDED"
