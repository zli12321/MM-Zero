# Changelog: main_svg.sh monitored run

**Purpose:** Log each failure and each fix during unattended runs. When you're back, read this file to see what happened and what was changed.

---

## When you're back

1. **Check if the run completed:** Look for an entry below like `[SUCCESS] Run finished at ...`. If present, the full pipeline finished.
2. **If there are failure entries:** Each `[FAILURE]` block contains the time, exit code, and last lines of the run log. Fixes applied (by you or the assistant) are documented in the same block under **FIX APPLIED**.
3. **To resume after a fix:** Run the same command you used to start (e.g. `bash SelfAgent_svg/scripts/run_main_svg_monitored.sh`). The pipeline skips already-completed steps (existing checkpoints).

---

## Run log file

- **Current run stdout/stderr:** `$STORAGE_PATH/temp_results/main_svg_run.log` (or path set by the wrapper)
- **Service logs:** `$STORAGE_PATH/temp_results/*.log` (codegen_*.log, solver_*.log, etc.)

---

## Entries (newest at bottom)

