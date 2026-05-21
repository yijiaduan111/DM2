# Active Manifest Batch

Active routine batch script:

`bash scripts/pica_server/local/active/run_manifest_v2d_both_ft50.sh`

Default manifest:

`/data/dyj/zts/clean_data/v20260520/batch_manifest.csv`

This script trains `v2c 150ep`, then `v2d Both FT` to total epoch 200, then runs deterministic and stochastic eval at damping x1/x2/x4. Training and eval both pass `--trajectory` explicitly from the clean manifest; it no longer switches or writes `output/hand_drag/<object_id>/trajectory.json`.

Historical pre-clean-data batch scripts were moved to:

`scripts/pica_server/local/archive/legacy_pre_clean_data_20260520/`
