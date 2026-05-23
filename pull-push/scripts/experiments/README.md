# Experiment Launchers

- `run_ours7_true_jointaware.sh`: reference launcher for the current PICA main flow.

The launcher runs fresh v2c training followed by v2d joint-aware ARAM fine-tuning and unified evaluation over the 7-object manifest-style sample list. Override `SAMPLES`, `GPUS`, `BATCH_SIZE`, `MANIFEST`, `OUT_ROOT`, and `RUN_TAG` as needed.

Generated outputs under `output/`, `runs/`, logs, checkpoints, and videos are intentionally ignored by git.
