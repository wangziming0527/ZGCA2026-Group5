# MyHolo / HoloMotion 项目说明

本仓库围绕 **HoloMotion** 的核心流程组织，覆盖：
- 基于 IsaacLab 的速度追踪 / 动作追踪环境定义；
- 基于 PPO 的训练；
- 数据整理与 SMPL 相关预处理；
- 评估脚本与离线指标分析；
- 大量可组合的 Hydra 配置。

## 1）代码结构

```text
MyHolo/
├── config/                         # Hydra 配置中心
│   ├── algo/                       # 算法配置（如 PPO）
│   ├── env/                        # 环境配置（观测、奖励、终止、地形、域随机化）
│   ├── evaluation/                 # 评估配置（motion/velocity/sim2sim）
│   ├── modules/                    # 模型结构配置（如 MLP actor/critic）
│   ├── motion_retargeting/         # 动作重定向流程配置
│   ├── robot/                      # 机器人参数（如 Unitree G1 29DoF）
│   ├── training/                   # 训练入口配置（任务组合）
│   └── data_curation/              # 数据清洗/拟合配置
│
├── src/
│   ├── training/                   # 训练入口与数据加载
│   │   ├── train.py                # Hydra 主入口，实例化算法并启动学习
│   │   └── h5_dataloader.py        # HDF5 数据读取
│   ├── algo/                       # 算法实现
│   │   └── ppo.py                  # PPO 训练主逻辑
│   ├── env/                        # 环境包装与 IsaacLab 组件
│   │   ├── velocity_tracking.py    # 速度追踪环境
│   │   ├── motion_tracking.py      # 动作追踪环境
│   │   └── isaaclab_components/    # 奖励、观测、命令、终止、地形等组件拼装
│   ├── modules/                    # 网络与 agent 组件
│   │   ├── network_modules.py
│   │   └── agent_modules.py
│   ├── evaluation/                 # 在线/离线评估、指标与报告
│   ├── motion_retargeting/         # 动作重定向与后处理工具
│   ├── data_curation/              # 视频到 SMPL、过滤、可视化等
│   └── utils/                      # 配置、旋转/数学工具、Isaac 工具
│
├── scripts/                        # 常用 shell 工作流封装
│   ├── training/                   # 训练脚本
│   ├── evaluation/                 # 评估脚本
│   ├── data_curation/              # 数据处理脚本
│   └── motion_retargeting/         # 动作重定向脚本
│
└── tests/                          # 测试占位（当前仅基础结构）
```

### 核心执行路径（训练）

1. 使用 `scripts/training/train_velocity_tracking.sh` 或 `train_motion_tracking.sh` 组织启动参数；
2. 启动 `src/training/train.py`（Hydra 入口），加载 `config/training/...` 下任务配置；
3. 在 `config` 中组合算法、环境、观测、奖励、终止等子配置；
4. 由 `src/algo/ppo.py` 创建 PPO Actor/Critic 与 rollout，驱动学习流程；
5. 环境由 `src/env/velocity_tracking.py` / `motion_tracking.py` 构建 IsaacLab `ManagerBasedRLEnv` 并执行交互。

## 2）要求

### 2.1 关于运行

对此项目，你无需运行代码，只需根据我的要求对代码进行阅读和修改即可。

### 2.2 关于 branch

在对话开始前，先询问用户：
- 这次对话的目的是什么；
- 是否需要建立 branch。

如果需要建 branch，可参考以下命名示例：
- `exp/ppo-lr-1e-4`
- `fix/ik-solver-bug`
