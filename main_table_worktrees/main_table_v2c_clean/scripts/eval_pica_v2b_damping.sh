#!/bin/bash
# scripts/eval_pica_v2b_damping.sh
#
# Damping sweep (x1, x2, x4) for a PICA v2b checkpoint, mirroring
# eval_long_damping.sh.
#
# v2b compatibility:
#   evaluate_ppo_baseline.py auto-detects PICA v2b aux-trained checkpoints by
#   scanning the state_dict for `a2c_network.aux_head.*` keys. When found,
#   it enables `physical_auxiliary` in HandDragTask -- this is *only* so the
#   env emits the same obs_dim (1750) the checkpoint was trained on. The
#   actor still slices the aux tail off before its forward, and is_train is
#   set to False inside GLAActor.forward, which short-circuits the aux head.
#   Net behaviour: action selection at eval is identical to v1 / v2a.
#
# DO NOT pass --phys_aux in production commands -- auto-detect handles
# both v2b checkpoints (aux=True) and older non-aux checkpoints (aux=False).
# A debug-only forced-on form is shown in a comment near the bottom.

set -e

# ---- Run / paths (override on command line if needed) ----
RUN_NAME="${RUN_NAME:-hand_drag_gla_45936_pica_drand12_aux_50ep}"
CKPT_DIR="${CKPT_DIR:-runs/${RUN_NAME}/nn}"
OUT_DIR="${OUT_DIR:-output/pica_v2b_eval/${RUN_NAME}}"

# ---- Eval protocol (parity with output/gla_tuning_45936/long_damping) ----
OBJ=45936
EP=20
SEED=42
MAX_LEN=300
GLA_POOL=last

# ---- Environment ----
PYBIN="${PYBIN:-/home/plote/miniconda3/envs/isaacgym/bin/python}"
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:${LD_LIBRARY_PATH:-}

cd /home/plote/new/2
mkdir -p "${OUT_DIR}"

echo "  RUN_NAME = ${RUN_NAME}"
echo "  CKPT_DIR = ${CKPT_DIR}"
echo "  OUT_DIR  = ${OUT_DIR}"
echo

if [ ! -d "${CKPT_DIR}" ]; then
    echo "  [error] checkpoint dir does not exist: ${CKPT_DIR}" >&2
    exit 1
fi

eval_one() {
    local D="$1"
    local PREFIX="${RUN_NAME}_damp${D}_det"
    local LOG_CSV="${OUT_DIR}/${PREFIX}_metrics.csv"
    local SUM_JSON="${OUT_DIR}/${PREFIX}_summary.json"
    local CMD_LOG="${OUT_DIR}/${PREFIX}.cmd"
    local STDOUT_LOG="${OUT_DIR}/${PREFIX}.log"

    if [ -f "${SUM_JSON}" ]; then
        echo "[skip] ${PREFIX} (summary already exists)"
        return
    fi
    echo "[eval] ${PREFIX}"

    # Persist exact command for reproducibility before running.
    {
        echo "${PYBIN} scripts/evaluate_ppo_baseline.py \\"
        echo "    --checkpoint \"${CKPT_DIR}\" --checkpoint-kind latest \\"
        echo "    --object_id ${OBJ} --episodes ${EP} --seed ${SEED} \\"
        echo "    --max_episode_length ${MAX_LEN} \\"
        echo "    --object_damping_scale ${D} \\"
        echo "    --gla_pool ${GLA_POOL} \\"
        echo "    --log_csv \"${LOG_CSV}\" --summary_json \"${SUM_JSON}\""
    } > "${CMD_LOG}"
    cat "${CMD_LOG}"

    "${PYBIN}" scripts/evaluate_ppo_baseline.py \
        --checkpoint "${CKPT_DIR}" --checkpoint-kind latest \
        --object_id "${OBJ}" --episodes "${EP}" --seed "${SEED}" \
        --max_episode_length "${MAX_LEN}" \
        --object_damping_scale "${D}" \
        --gla_pool "${GLA_POOL}" \
        --log_csv "${LOG_CSV}" --summary_json "${SUM_JSON}" \
        2>&1 | tee "${STDOUT_LOG}" | tail -12

    # Debug-only forced-on form (DO NOT enable in production):
    # "${PYBIN}" scripts/evaluate_ppo_baseline.py --phys_aux 1 \
    #     --checkpoint "${CKPT_DIR}" --checkpoint-kind latest ...
}

DAMPS=(1.0 2.0 4.0)
for D in "${DAMPS[@]}"; do
    eval_one "${D}"
done

echo
echo "  DONE damping eval. Building summary markdown..."

# ---- Summary markdown -----------------------------------------------------
# Pulls success/return/progress/length from each *_summary.json and derives
# clip099 / detach / mean action_abs_max / mean palm dist from the matching
# *_metrics.csv (same logic as scripts/eval_postprocess.py, inlined to avoid
# a second subprocess call).
#
# We use a *quoted* heredoc and pass the dynamic values via env vars so the
# shell does not try to interpret Python f-string braces or array syntax.
DAMPS_JSON="[$(IFS=, ; echo "${DAMPS[*]}")]"
export RUN_NAME OUT_DIR DAMPS_JSON
"${PYBIN}" - <<'PY'
import csv, json, os
from pathlib import Path
from statistics import mean

RUN_NAME = os.environ["RUN_NAME"]
OUT_DIR  = Path(os.environ["OUT_DIR"])
DAMPS    = json.loads(os.environ["DAMPS_JSON"])

def per_csv_stats(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return None
    n = len(rows)
    by_ep = {}
    for r in rows:
        by_ep.setdefault(int(r["episode"]), []).append(r)
    eps = []
    for ep, ep_rows in sorted(by_ep.items()):
        ep_rows.sort(key=lambda r: int(r["step"]))
        min_r = min(float(r["reward"]) for r in ep_rows)
        any_s = any(int(r["success"]) for r in ep_rows)
        max_p = max(float(r["palm_to_handle_dist"]) for r in ep_rows)
        eps.append({"min_r": min_r, "any_s": any_s, "max_p": max_p})
    clip095 = sum(1 for r in rows if float(r["action_abs_max"]) >= 0.95) / n
    clip099 = sum(1 for r in rows if float(r["action_abs_max"]) >= 0.99) / n
    detach  = sum(
        1 for e in eps if (not e["any_s"]) and e["min_r"] <= -40.0
    ) / max(1, len(eps))
    palm_max = mean(e["max_p"] for e in eps)
    act_max  = mean(float(r["action_abs_max"]) for r in rows)
    return {
        "clip095": clip095, "clip099": clip099, "detach": detach,
        "palm_max": palm_max, "action_max_mean": act_max,
    }

lines = []
lines.append(f"# PICA v2b damping eval -- {RUN_NAME}")
lines.append("")
parts = RUN_NAME.split("_")
# First all-numeric token is the object id (e.g. "45936" inside
# hand_drag_gla_45936_pica_drand12_aux_50ep). Falls back to "N/A".
obj_id = next((p for p in parts if p.isdigit()), "N/A")
lines.append(
    f"Object {obj_id}, {len(DAMPS)} damping scales, "
    f"20-episode deterministic eval, seed=42, max_episode_length=300, "
    f"gla_pool=last. obs_dim and aux state are auto-detected from the "
    f"checkpoint via evaluate_ppo_baseline.py."
)
lines.append("")
lines.append("| method | damping | success | progress | return  | steps  | action_l2 | clip099 | detach |")
lines.append("|---     |---:     |---:     |---:      |---:     |---:    |---:       |---:     |---:    |")
for d in DAMPS:
    prefix = f"{RUN_NAME}_damp{d}_det"
    sum_p  = OUT_DIR / f"{prefix}_summary.json"
    csv_p  = OUT_DIR / f"{prefix}_metrics.csv"
    if not sum_p.exists():
        lines.append(f"| {RUN_NAME} | {d} | MISSING | | | | | | |")
        continue
    s = json.load(open(sum_p))
    cs = per_csv_stats(csv_p) or {}
    lines.append(
        f"| {RUN_NAME} | {d} | {s.get('success_rate', 0):.2f} | "
        f"{s.get('normalized_progress_mean', 0):.3f} | "
        f"{s.get('return_mean', 0):.2f} | "
        f"{s.get('steps_mean', 0):.1f} | "
        f"{s.get('mean_action_l2', 0):.3f} | "
        f"{cs.get('clip099', 0):.3f} | "
        f"{cs.get('detach', 0):.2f} |"
    )

out_md = OUT_DIR / "damping_eval_table.md"
out_md.write_text("\n".join(lines) + "\n")
print(f"  wrote: {out_md}")
PY

echo "DONE eval_pica_v2b_damping"
