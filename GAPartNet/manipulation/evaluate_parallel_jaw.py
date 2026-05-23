import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from object_gym import ObjectGym
from utils import prepare_gsam_model, read_yaml_config
import torch

try:
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert
except ModuleNotFoundError:
    def matrix_to_quaternion(matrix):
        """Minimal PyTorch3D-compatible wxyz conversion for 3x3 rotation matrices."""
        m00 = matrix[..., 0, 0]
        m11 = matrix[..., 1, 1]
        m22 = matrix[..., 2, 2]
        qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0)) / 2.0
        qx = torch.sign(matrix[..., 2, 1] - matrix[..., 1, 2]) * torch.sqrt(
            torch.clamp(1.0 + m00 - m11 - m22, min=0.0)
        ) / 2.0
        qy = torch.sign(matrix[..., 0, 2] - matrix[..., 2, 0]) * torch.sqrt(
            torch.clamp(1.0 - m00 + m11 - m22, min=0.0)
        ) / 2.0
        qz = torch.sign(matrix[..., 1, 0] - matrix[..., 0, 1]) * torch.sqrt(
            torch.clamp(1.0 - m00 - m11 + m22, min=0.0)
        ) / 2.0
        quat = torch.stack((qw, qx, qy, qz), dim=-1)
        return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-8)

    def quaternion_invert(quaternion):
        return quaternion * quaternion.new_tensor([1.0, -1.0, -1.0, -1.0])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GAPartNet-style parallel-jaw pulling.")
    parser.add_argument("--object_id", default="45661")
    parser.add_argument("--config", default="config")
    parser.add_argument("--asset_root", default="gapartnet_example")
    parser.add_argument("--output", default="output/parallel_jaw_eval")
    parser.add_argument("--bbox_id", type=int, default=-1)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument("--settle_steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def init_gym(cfgs, task_cfg, device):
    if cfgs["INFERENCE_GSAM"]:
        grounded_dino_model, sam_predictor = prepare_gsam_model(device=device)
    else:
        grounded_dino_model, sam_predictor = None, None

    selected_obj_names = task_cfg["selected_obj_names"]
    selected_obj_urdfs = task_cfg["selected_urdfs"]
    selected_obj_num = len(selected_obj_names)
    selected_ob_poses = task_cfg["init_obj_pos"]
    selected_ob_pose_rs = [pose[3:] for pose in selected_ob_poses]

    cfgs["asset"]["position_noise"] = [0, 0, 0]
    cfgs["asset"]["rotation_noise"] = 0
    cfgs["asset"]["asset_files"] = selected_obj_urdfs
    cfgs["asset"]["asset_seg_ids"] = [2 + i for i in range(selected_obj_num)]
    cfgs["asset"]["obj_pose_ps"] = selected_ob_poses
    cfgs["asset"]["obj_pose_rs"] = selected_ob_pose_rs

    gym = ObjectGym(cfgs, grounded_dino_model, sam_predictor)
    gym.refresh_observation(get_visual_obs=False)
    gym.run_steps(pre_steps=10, refresh_obs=False, print_step=False)
    gym.refresh_observation(get_visual_obs=False)
    return gym


def get_articulated_dof(gym):
    gym.refresh_observation(get_visual_obs=False)
    start = gym.franka_num_dofs + gym.obj_num_dofs
    end = start + gym.arti_obj_num_dofs
    dof = gym.dof_states.reshape(gym.num_envs, -1, 2)[0, start:end, :]
    return dof[:, 0].detach().cpu().numpy(), dof[:, 1].detach().cpu().numpy()


def get_gripper_distance(gym, target_position):
    gym.refresh_observation(get_visual_obs=False)
    hand = gym.hand_pos[0].detach().cpu().numpy()
    return float(np.linalg.norm(hand - target_position))


def record_stage(rows, stage, step_num, gym, target_position):
    qpos, qvel = get_articulated_dof(gym)
    rows.append({
        "stage": stage,
        "step": int(step_num),
        "gripper_target_dist": get_gripper_distance(gym, target_position),
        **{f"dof_{i}_pos": float(v) for i, v in enumerate(qpos)},
        **{f"dof_{i}_vel": float(v) for i, v in enumerate(qvel)},
    })


def compute_part_geometry(gym, cfgs, bbox_id):
    gym.get_gapartnet_anno()
    all_bbox_now = gym.gapart_init_bboxes[0] * cfgs["asset"]["arti_obj_scale"]
    rotation = R.from_quat(gym.arti_init_obj_rot_list[0])
    all_bbox_now = np.dot(all_bbox_now, rotation.as_matrix().T)
    all_bbox_now = all_bbox_now + gym.arti_init_obj_pos_list[0]

    all_bbox_now = torch.tensor(all_bbox_now, dtype=torch.float32, device=gym.device).reshape(-1, 8, 3)
    all_bbox_center_front_face = torch.mean(all_bbox_now[:, 0:4, :], dim=1)
    handle_out = all_bbox_now[:, 0, :] - all_bbox_now[:, 4, :]
    handle_out /= torch.norm(handle_out, dim=1, keepdim=True)
    handle_long = all_bbox_now[:, 0, :] - all_bbox_now[:, 1, :]
    handle_long /= torch.norm(handle_long, dim=1, keepdim=True)
    handle_short = all_bbox_now[:, 0, :] - all_bbox_now[:, 3, :]
    handle_short /= torch.norm(handle_short, dim=1, keepdim=True)
    rotations = quaternion_invert(matrix_to_quaternion(torch.cat((
        handle_long.reshape((-1, 1, 3)),
        handle_short.reshape((-1, 1, 3)),
        -handle_out.reshape((-1, 1, 3)),
    ), dim=1)))

    return {
        "bbox_id": bbox_id,
        "category": gym.gapart_cates[0][bbox_id],
        "link_name": gym.gapart_link_names[0][bbox_id],
        "init_position": all_bbox_center_front_face[bbox_id].detach().cpu().numpy(),
        "handle_out": handle_out[bbox_id].detach().cpu().numpy(),
        "rotation": rotations[bbox_id].detach().cpu().numpy(),
    }


def write_outputs(output_dir, rows, summary):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    # ObjectGym calls gymutil.parse_arguments internally; keep only Isaac Gym args visible to it.
    sys.argv = [sys.argv[0], "--headless"] if args.headless else [sys.argv[0]]

    cfgs = read_yaml_config(f"{args.config}.yaml")
    with open("task_config.json", "r") as f:
        task_cfg = json.load(f)

    cfgs["HEADLESS"] = args.headless
    cfgs["USE_CUROBO"] = False
    cfgs["asset"]["arti_obj_root"] = args.asset_root
    cfgs["asset"]["arti_position_noise"] = 0.0
    cfgs["asset"]["arti_rotation_noise"] = 0.0
    cfgs["asset"]["arti_obj_scale"] = 0.4
    cfgs["asset"]["arti_rotation"] = 0
    cfgs["asset"]["arti_gapartnet_ids"] = [args.object_id]

    min_z = -1.5
    if os.path.exists("gapartnet_obj_min_z.json"):
        with open("gapartnet_obj_min_z.json", "r") as f:
            min_z = json.load(f).get(args.object_id, min_z)
    cfgs["asset"]["arti_obj_pose_ps"] = [[0.8, 0, -0.4 * min_z]]

    gym = init_gym(cfgs, task_cfg, args.device)
    rows = []
    try:
        part = compute_part_geometry(gym, cfgs, args.bbox_id)
        initial_qpos, _ = get_articulated_dof(gym)
        record_stage(rows, "init", 0, gym, part["init_position"])

        step_num = 0
        pre_grasp_position = part["init_position"] + 0.2 * part["handle_out"]
        for _ in range(10):
            step_num, _ = gym.control_to_pose(
                np.array([*pre_grasp_position, *part["rotation"]]),
                close_gripper=False,
                save_video=False,
                save_root=None,
                step_num=step_num,
                use_ik=True,
            )
        record_stage(rows, "pre_grasp", step_num, gym, part["init_position"])

        grasp_position = part["init_position"] + 0.1 * part["handle_out"]
        for _ in range(10):
            step_num, _ = gym.control_to_pose(
                np.array([*grasp_position, *part["rotation"]]),
                close_gripper=False,
                save_video=False,
                save_root=None,
                step_num=step_num,
                use_ik=True,
            )
        record_stage(rows, "grasp_pose", step_num, gym, part["init_position"])

        for _ in range(10):
            step_num = gym.move_gripper(close_gripper=True, save_video=False, save_root=None, start_step=step_num)
        record_stage(rows, "gripper_closed", step_num, gym, part["init_position"])

        for i in range(30):
            step_num, _ = gym.control_to_pose(
                np.array([*(part["init_position"] + (0.1 + i * 0.01) * part["handle_out"]), *part["rotation"]]),
                close_gripper=True,
                save_video=False,
                save_root=None,
                step_num=step_num,
                use_ik=True,
            )
            if i in {0, 9, 19, 29}:
                record_stage(rows, f"pull_{i + 1:02d}", step_num, gym, part["init_position"])

        if args.settle_steps > 0:
            gym.run_steps(pre_steps=args.settle_steps, refresh_obs=False, print_step=False)
        record_stage(rows, "final", step_num + args.settle_steps, gym, part["init_position"])

        final_qpos, _ = get_articulated_dof(gym)
        lower = np.asarray(gym.arti_obj_dof_props["lower"], dtype=np.float32)
        upper = np.asarray(gym.arti_obj_dof_props["upper"], dtype=np.float32)
        dof_range = np.maximum(np.abs(upper - lower), 1e-6)
        delta = final_qpos - initial_qpos
        norm_abs_progress = np.abs(delta) / dof_range
        target_dof = int(np.argmax(norm_abs_progress))
        success = bool(norm_abs_progress[target_dof] >= args.success_threshold)

        summary = {
            "method": "GT Part Pose + Parallel-Jaw Pulling",
            "object_id": args.object_id,
            "asset_root": args.asset_root,
            "bbox_id": int(args.bbox_id),
            "bbox_category": part["category"],
            "bbox_link_name": part["link_name"],
            "success_threshold": args.success_threshold,
            "success": success,
            "target_dof": target_dof,
            "target_dof_initial": float(initial_qpos[target_dof]),
            "target_dof_final": float(final_qpos[target_dof]),
            "target_dof_delta": float(delta[target_dof]),
            "target_dof_range": float(dof_range[target_dof]),
            "normalized_abs_progress": float(norm_abs_progress[target_dof]),
            "final_gripper_target_dist": rows[-1]["gripper_target_dist"],
            "min_gripper_target_dist": float(min(row["gripper_target_dist"] for row in rows)),
            "steps": int(step_num + args.settle_steps),
            "all_initial_qpos": initial_qpos.tolist(),
            "all_final_qpos": final_qpos.tolist(),
            "all_normalized_abs_progress": norm_abs_progress.tolist(),
        }
        write_outputs(args.output, rows, summary)
        print(json.dumps(summary, indent=2))
    finally:
        gym.clean_up()


if __name__ == "__main__":
    main()
