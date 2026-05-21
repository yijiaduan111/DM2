# Project Structure

现有数据生成代码保持不动，仍然使用根目录这几个文件：

- `run_hand_drag.py`
- `hand_object_gym.py`
- `build_hand_urdf.py`
- `hand_config.yaml`

生成出来的数据仍然在：

- `output/hand_drag/`

为了后面把 PPO 作为主代码继续扩展，目录先按职责预留成下面这样：

- `ppo/`: 后续 PPO 主代码
- `dataset/`: 数据集处理、统计、转换脚本
- `docs/`: 实验记录、结构说明

建议后续这么放：

- PPO 环境封装放 `ppo/env.py`
- 观测和奖励放 `ppo/obs.py`、`ppo/reward.py`
- 训练入口放 `ppo/train_ppo.py`
- 数据筛选和分析放 `dataset/`

这样做的原则是：

- 不动当前能跑通的数据生成原型
- PPO 单独长成主线
- 数据处理和训练逻辑分开
