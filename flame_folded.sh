#!/usr/bin/env bash
#
# flamegraph_record.sh
#
# Records perf data for a command or PID, converts it to a FlameGraph folded file,
# and opens it in Firefox as the invoking (non-root) user.
#
# Usage:
#   sudo ./flamegraph_record.sh "<command>" [output_basename] [flamegraph_dir]
#   sudo ./flamegraph_record.sh --pid <PID> --duration <seconds> [output_basename] [flamegraph_dir]
#
#   <command>        – The program to profile (quoted if it has args).
#   --pid <PID>      – Process ID to profile
#   --duration <sec> – Duration to record in seconds (used with --pid)
#   output_basename  – Base name for .perf / .folded / .svg (default: flamegraph)
#   flamegraph_dir   – Directory containing stackcollapse-perf.pl & flamegraph.pl
#                      (default: current directory)
#
# Requirements:
#   * perf (sudo apt install linux-tools-$(uname -r))
#   * Brendan Gregg's FlameGraph scripts
#   * Firefox installed

set -euo pipefail

# Parse arguments
CMD=""
PID=""
DURATION=""
BASE="flamegraph"
FLG_DIR=$(pwd)

while [[ $# -gt 0 ]]; do
    case $1 in
        --pid)
            PID="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        *)
            if [[ -z "$CMD" && -z "$PID" ]]; then
                CMD="$1"
            elif [[ "$CMD" != "" || "$PID" != "" ]] && [[ "$BASE" == "flamegraph" ]]; then
                BASE="$1"
            elif [[ "$FLG_DIR" == $(pwd) ]]; then
                FLG_DIR="$1"
            fi
            shift
            ;;
    esac
done

PERF_DATA="perf.data"
PERF_TXT="${BASE}.perf"
FOLDED="${BASE}.folded"

TARGET_USER=${SUDO_USER:-$USER}

# --------------------------- check arguments -----------------------------
if [[ -z "$CMD" && -z "$PID" ]]; then
    echo "Usage: sudo $0 \"<command>\" [output_basename] [flamegraph_dir]" >&2
    echo "   or: sudo $0 --pid <PID> --duration <seconds> [output_basename] [flamegraph_dir]" >&2
    exit 1
fi

if [[ -n "$PID" && -z "$DURATION" ]]; then
    echo "Error: --duration is required when using --pid" >&2
    exit 1
fi

if [[ -n "$CMD" && -n "$PID" ]]; then
    echo "Error: Cannot specify both command and PID" >&2
    exit 1
fi

if [[ ! -x "${FLG_DIR}/stackcollapse-perf.pl" ]] || [[ ! -x "${FLG_DIR}/flamegraph.pl" ]]; then
    echo "Error: stackcollapse-perf.pl or flamegraph.pl not found in '$FLG_DIR'." >&2
    exit 1
fi

if ! command -v perf &>/dev/null; then
    echo "Error: 'perf' not found. Install with: sudo apt install linux-tools-\$(uname -r)" >&2
    exit 1
fi

# --------------------------- record perf data ----------------------------
if [[ -n "$CMD" ]]; then
    echo "[*] Recording perf data for: $CMD"
    sudo perf record -F 999 -g --call-graph fp -o "$PERF_DATA" -- bash -c "$CMD"
else
    echo "[*] Recording perf data for PID: $PID (duration: ${DURATION}s)"
    
    # Check if PID exists
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Error: Process $PID does not exist or is not accessible" >&2
        exit 1
    fi
    
    # Try different perf record options to handle synthesis issues
    echo "[*] Attempting to record with Frame Pointer call-graph..."
    if sudo perf record -F 999 -g --call-graph fp -o "$PERF_DATA" -p "$PID" sleep "$DURATION" 2>/dev/null; then
        echo "[✓] Successfully recorded with Frame Pointer call-graph"
    else
        echo "[!] Frame Pointer failed, trying with DWARF..."
        if sudo perf record -F 999 -g --call-graph dwarf -o "$PERF_DATA" -p "$PID" sleep "$DURATION" 2>/dev/null; then
            echo "[✓] Successfully recorded with DWARF call-graph"
        else
            echo "[!] DWARF failed, trying basic call-graph..."
            sudo perf record -F 999 -g -o "$PERF_DATA" -p "$PID" sleep "$DURATION"
        fi
    fi
fi

# Verify perf.data was created
if [[ ! -f "$PERF_DATA" ]]; then
    echo "Error: perf record failed to create $PERF_DATA" >&2
    echo "Current directory: $(pwd)" >&2
    echo "Files in current directory:" >&2
    ls -la . >&2
    exit 1
fi

echo "[✓] perf.data created successfully: $PERF_DATA"

# Change ownership of perf.data to the real user
if [[ -n "${SUDO_USER:-}" ]] && [[ -f "$PERF_DATA" ]]; then
    sudo chown "$TARGET_USER:$TARGET_USER" "$PERF_DATA"
elif [[ -n "${SUDO_USER:-}" ]] && [[ ! -f "$PERF_DATA" ]]; then
    echo "Warning: $PERF_DATA not found, skipping ownership change" >&2
fi

# --------------------- generate flamegraph as user -----------------------
sudo -u "$TARGET_USER" bash <<EOF
set -euo pipefail

echo "[*] Converting perf.data to text..."
perf script -i "$PERF_DATA" > "$PERF_TXT"

echo "[*] Collapsing stacks..."
"${FLG_DIR}/stackcollapse-perf.pl" "$PERF_TXT" > "$FOLDED"

echo "[✓] FlameGraph written to: $FOLDED"
EOF
