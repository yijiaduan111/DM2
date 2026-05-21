#!/usr/bin/env python3
"""
scripts/pica_server/summarize_pica_calibration.py

After the 20-epoch PICA-GLA calibration runs in run_pica_gla_calibration_20ep.sh
have completed, this script reads each run's epoch_rewards.csv, picks epoch 20
(or the latest available epoch if 20 is missing), and writes a Markdown table
+ heuristic interpretation to:

    reports/pica_handoff/pica_calibration_20ep_summary.md

The script does NOT run any training or evaluation. It only reads CSVs that
were produced earlier on the server.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (experiment_name, lambda_bound) — keep in sync with the calibration runner.
EXPERIMENTS = [
    ("hand_drag_gla_45936_pica_ab1_contact05_smoke20",  1.0),
    ("hand_drag_gla_45936_pica_ab5_contact05_smoke20",  5.0),
    ("hand_drag_gla_45936_pica_ab20_contact05_smoke20", 20.0),
]

COLUMNS_OUT = [
    "reward_mean",
    "r_task_mean",
    "r_act_mean",
    "r_phys_bound_mean",
    "r_phys_contact_mean",
    "r_phys_slip_mean",
    "r_phys_smooth_mean",
    "r_phys_total_mean",
    "success_mean",
    "normalized_progress_mean",
    "final_joint_pos_mean",
    "length_mean",
]

TARGET_EPOCH = 20


def read_target_row(csv_path: Path):
    """Return (row_dict, epoch) for epoch=TARGET_EPOCH or the latest row."""
    if not csv_path.exists():
        return None, None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    target = None
    latest = None
    for r in rows:
        try:
            ep = int(r.get("epoch", -1))
        except (TypeError, ValueError):
            continue
        latest = (r, ep)
        if ep == TARGET_EPOCH:
            target = (r, ep)
            break
    return target if target is not None else latest


def fmt_num(x, fmt="{:.4f}"):
    if x is None or x == "":
        return "N/A"
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return str(x)


def main():
    out_dir = PROJECT_ROOT / "reports" / "pica_handoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pica_calibration_20ep_summary.md"

    rows_for_table = []
    for exp_name, lb in EXPERIMENTS:
        csv_path = PROJECT_ROOT / "runs" / exp_name / "epoch_rewards.csv"
        row, ep = read_target_row(csv_path)
        if row is None:
            print(f"  [warn] missing or empty: {csv_path}", file=sys.stderr)
            rows_for_table.append({
                "exp_name": exp_name,
                "lambda_bound": lb,
                "epoch": None,
                "missing": True,
            })
            continue

        rows_for_table.append({
            "exp_name": exp_name,
            "lambda_bound": lb,
            "epoch": ep,
            "missing": False,
            **{k: row.get(k) for k in COLUMNS_OUT},
        })

    # --- Build Markdown ----------------------------------------------------
    lines = []
    lines.append("# PICA-GLA action_bound calibration — epoch-20 summary")
    lines.append("")
    lines.append(
        "Object 45936, num_envs=64, 20 training epochs, "
        "`train_config_gla_pica.yaml` with `lambda_bound` overridden via CLI. "
        "All other PICA weights kept at default (`contact_distance.weight=0.5`, "
        "slip and smoothness disabled). Numbers are read from "
        "`runs/<exp>/epoch_rewards.csv` at epoch=20 (or the latest available)."
    )
    lines.append("")
    header = (
        "| experiment | lambda_bound | epoch | reward | r_task | r_act | "
        "r_phys_bound | r_phys_contact | r_phys_total | success | "
        "progress | final_joint | length |"
    )
    sep = (
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines.append(header)
    lines.append(sep)
    for r in rows_for_table:
        if r["missing"]:
            lines.append(
                f"| {r['exp_name']} | {r['lambda_bound']} | MISSING | | | | | | | | | | |"
            )
            continue
        lines.append(
            "| {exp} | {lb} | {ep} | "
            "{rw} | {rt} | {ra} | "
            "{rb} | {rc} | {rp} | "
            "{su} | {pr} | {fj} | {ln} |".format(
                exp=r["exp_name"],
                lb=r["lambda_bound"],
                ep=r["epoch"],
                rw=fmt_num(r.get("reward_mean")),
                rt=fmt_num(r.get("r_task_mean")),
                ra=fmt_num(r.get("r_act_mean"), "{:.5f}"),
                rb=fmt_num(r.get("r_phys_bound_mean"), "{:.5f}"),
                rc=fmt_num(r.get("r_phys_contact_mean"), "{:.5f}"),
                rp=fmt_num(r.get("r_phys_total_mean"), "{:.5f}"),
                su=fmt_num(r.get("success_mean"), "{:.3f}"),
                pr=fmt_num(r.get("normalized_progress_mean"), "{:.3f}"),
                fj=fmt_num(r.get("final_joint_pos_mean"), "{:.4f}"),
                ln=fmt_num(r.get("length_mean"), "{:.1f}"),
            )
        )

    lines.append("")
    lines.append("## How to read this table")
    lines.append("")
    lines.append(
        "- If `r_phys_bound` is still near zero **and** progress / saturation "
        "are unchanged versus the disabled-PICA baseline (`gla_long_bounds01` "
        "at epoch 20), `lambda_bound` is too weak: raise it.\n"
        "- If `progress` collapses near zero relative to disabled-PICA at "
        "epoch 20, `lambda_bound` is too strong: lower it.\n"
        "- Choose the middle setting where `r_phys_bound` is visibly non-zero "
        "(say |value| > 1e-3) **and** progress is comparable to disabled-PICA.\n"
        "- Compare against the smoke baselines in "
        "`smoke_csv/smoke_pica_disabled.csv` and "
        "`smoke_csv/smoke_pica_enabled.csv` for sanity."
    )
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append(
        "Once a winning `lambda_bound` is chosen, edit "
        "`ppo/train_config_gla_pica.yaml::physical_regularization.action_bound.weight` "
        "and run a 100-epoch PICA-GLA training. Then run damping x1/x2/x4 "
        "deterministic eval with the same protocol used in "
        "`gla_tuning/long_damping/long_damping_table.md`."
    )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
