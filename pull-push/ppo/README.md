把后续 PPO 主代码放在这里。

建议最少拆成：

- `env.py`: 把现有仿真封成 RL 接口
- `obs.py`: 观测拼接
- `reward.py`: 奖励函数
- `train_ppo.py`: 训练入口
- `eval.py`: 评估脚本
