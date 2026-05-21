# Unified main-table eval

This directory freezes the eval protocol for rows 03-07.

Default protocol:
- canonical data manifest: `/data/dyj/zts/clean_data/v20260520/batch_manifest.csv`
- checkpoint kind: `best` (`HandDrag.pth`)
- modes: `det stoch`
- damping scales: `1 2 4`
- episodes: `20`
- seed: `42`
- max episode length: `300`
- evaluator: `/data/dyj/zts/main_table_worktrees/main_table_baselines/code/scripts/evaluate_ppo_unified.py`

Smoke test example:

```bash
ROWS=07 ONLY_SAMPLE=45936_handle_1 DAMPS="1" MODES="det stoch" EPISODES=1 GPU_ID=6 \
  /data/dyj/zts/main_table_worktrees/main_table_baselines/scripts/unified_eval/run_unified_eval.sh
```
