# Active Server Experiment Scripts

This folder keeps only the current reusable experiment entrypoints.
Old one-off scripts and failed/legacy experiment helpers were archived during cleanup.

## Environment
Always run from repo root and source:

```bash
source scripts/pica_server/local/env_cuda_kernel_eval.sh
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTORCH_JIT=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Current Scripts

- `run_manifest_v2d_both_ft50.sh`: current manifest/single-trajectory batch pipeline.
- `make_eval_core_table_eval_clip_detach.py`: rebuilds the compact eval table with eval-time `clip099` and `detach`.

## Current Outputs

- `reports/pica_handoff/current_multiobject_eval_table.md`: compact metric table.
- `output/pica_manifest_singletraj_v2d_both_ft50_eval`: 45936/45661 eval outputs.
- `output/pica_single_7310_rerun_eval`: 7310 eval outputs.
- `output/pica_single_45261_handle7_clean_eval`: 45261 handle_7 eval outputs.
- `output/pica_rerun_27044_41529_eval`: 27044/41529 rerun eval outputs.
