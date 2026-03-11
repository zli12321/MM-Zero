#!/bin/bash
# =============================================================================
# Monitored run of main_svg.sh: on failure, log to CHANGELOG_SVG_RUN.md and
# restart (main_svg skips completed steps). Stops after MAX_RESTARTS failures.
# Run from repo root: bash MM-zero_final/scripts/run_main_svg_monitored.sh
# =============================================================================

set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STORAGE_PATH="${STORAGE_PATH:-/workspace/selfAgent_Storage_svg_round1}"
export STORAGE_PATH
LOG_DIR="${STORAGE_PATH}/temp_results"
CHANGELOG="${SCRIPT_DIR}/../CHANGELOG_SVG_RUN.md"
RUN_LOG="${LOG_DIR}/main_svg_run.log"
MAIN_SVG="${REPO_ROOT}/MM-zero_final/scripts/main_svg.sh"

# After this many consecutive failures, stop (so you can fix and re-run)
MAX_RESTARTS="${MAX_RESTARTS:-5}"
# Last N lines of run log to append to changelog on failure
TAIL_LINES="${TAIL_LINES:-80}"

mkdir -p "$LOG_DIR"

append_changelog() {
    local content="$1"
    {
        echo ""
        echo "---"
        echo "$content"
    } >> "$CHANGELOG"
}

run_attempt=0
while true; do
    run_attempt=$((run_attempt + 1))
    echo "[monitor] Run attempt $run_attempt at $(date -Iseconds)"
    echo "[monitor] Run attempt $run_attempt at $(date -Iseconds)" >> "$RUN_LOG"

    cd "$REPO_ROOT" || exit 1
    bash "$MAIN_SVG" 2>&1 | tee -a "$RUN_LOG"
    exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -eq 0 ]; then
        append_changelog "[SUCCESS] Run finished at $(date -Iseconds) (attempt $run_attempt)."
        echo "[monitor] main_svg.sh completed successfully."
        exit 0
    fi
    echo "[monitor] main_svg.sh failed with exit code $exit_code"

    # Log failure to changelog
    {
        echo ""
        echo "### [FAILURE] $(date -Iseconds) (attempt $run_attempt)"
        echo "- **Exit code:** $exit_code"
        echo "- **Run log:** \`$RUN_LOG\`"
        echo ""
        echo "**Last $TAIL_LINES lines of run log:**"
        echo '```'
        tail -n "$TAIL_LINES" "$RUN_LOG" 2>/dev/null || echo "(could not read log)"
        echo '```'
        echo ""
        echo "**FIX APPLIED:** (none yet — fill in when you fix and re-run)"
        echo ""
    } >> "$CHANGELOG"

    if [ "$run_attempt" -ge "$MAX_RESTARTS" ]; then
        append_changelog "[STOP] Stopped after $MAX_RESTARTS failure(s). Please fix and re-run this script."
        echo "[monitor] Stopped after $MAX_RESTARTS failures. See $CHANGELOG"
        exit "$exit_code"
    fi

    echo "[monitor] Restarting in 30s (will skip already-completed steps)..."
    sleep 30
done
