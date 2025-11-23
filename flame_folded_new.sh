#!/usr/bin/env bash
#
# flamegraph_record.sh
#
# Records perf data for a command or PID, converts it to a FlameGraph folded file,
# and opens it in Firefox as the invoking (non-root) user.
#
# Usage:
#   sudo ./flamegraph_record.sh "<command>" [output_basename] [flamegraph_dir]
#   sudo ./flamegraph_record.sh --pid <PID> -- command <specific_command> [output_basename] [flamegraph_dir]
#
#   <command>        – The program to profile (quoted if it has args).
#   --pid <PID> -- command <specific_command> – Profile existing PID for the duration of command execution.
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
BASE="flamegraph"
FLG_DIR=$(pwd)
PROFILE_PID_WITH_CMD=""
EXEC_CMD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pid)
            PROFILE_PID_WITH_CMD="$2"
            shift 2
            # Look for -- separator
            if [[ $# -gt 0 && "$1" == "--" ]]; then
                shift
                # Everything after -- is the command
                EXEC_CMD="$*"
                break
            else
                echo "Error: --pid requires -- separator followed by command" >&2
                exit 1
            fi
            ;;
        *)
            if [[ -z "$CMD" && -z "$PROFILE_PID_WITH_CMD" ]]; then
                CMD="$1"
            elif [[ "$CMD" != "" || "$PROFILE_PID_WITH_CMD" != "" ]] && [[ "$BASE" == "flamegraph" ]]; then
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
if [[ -z "$CMD" && -z "$PROFILE_PID_WITH_CMD" ]]; then
    echo "Usage: sudo $0 \"<command>\" [output_basename] [flamegraph_dir]" >&2
    echo "   or: sudo $0 --pid <PID> -- command <specific_command> [output_basename] [flamegraph_dir]" >&2
    exit 1
fi

if [[ -n "$PROFILE_PID_WITH_CMD" ]]; then
    if ! [[ "$PROFILE_PID_WITH_CMD" =~ ^[0-9]+$ ]]; then
        echo "Error: PID must be a number" >&2
        exit 1
    fi
    if ! kill -0 "$PROFILE_PID_WITH_CMD" 2>/dev/null; then
        echo "Error: Process with PID $PROFILE_PID_WITH_CMD not found or not accessible" >&2
        exit 1
    fi
    if [[ -z "$EXEC_CMD" ]]; then
        echo "Error: --pid requires a command after --" >&2
        exit 1
    fi
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
if [[ -n "$PROFILE_PID_WITH_CMD" ]]; then
    echo "[*] Recording perf data for PID: $PROFILE_PID_WITH_CMD while executing: $EXEC_CMD"
    # Start perf recording in background
    sudo perf record -F 999 -g --call-graph fp -o "$PERF_DATA" -p "$PROFILE_PID_WITH_CMD" &
    PERF_PID=$!
    
    # Execute the command and wait for it to complete
    echo "[*] Executing command: $EXEC_CMD"
    if ! bash -c "$EXEC_CMD"; then
        echo "Error: Command failed: $EXEC_CMD" >&2
        kill $PERF_PID 2>/dev/null
        wait $PERF_PID 2>/dev/null || true
        exit 1
    fi
    
    # Stop perf recording
    echo "[*] Command completed, stopping perf recording..."
    kill $PERF_PID 2>/dev/null
    wait $PERF_PID 2>/dev/null || true
else
    echo "[*] Recording perf data for: $CMD"
    sudo perf record -F 999 -g --call-graph fp -o "$PERF_DATA" -- bash -c "$CMD"
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
