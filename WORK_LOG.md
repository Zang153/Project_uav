# UAV Project Development Log

## 2026-03-19

### 1. 控制器 PyTorch 张量化重构 (Controller PyTorch Refactoring)
- 将核心控制器的底层运算从 NumPy 迁移至 PyTorch 张量 (Tensor)，涉及文件包括：
  - `uav_project/controllers/cascade_controller.py`
  - `uav_project/controllers/delta_controller.py`
  - `uav_project/controllers/pid.py`
- 统一矩阵运算形状（如明确 `(3, 1)` 列向量），并引入 `torch.no_grad()` 优化前向控制计算的性能。
- 修复了 NumPy 数组与 PyTorch 张量混合运算导致的 `TypeError` 问题，保证了端到端张量计算。

### 2. 数学函数与运动学库统一 (Math Utilities Standardization)
- 对项目中的基础数学运算和运动学库进行了 PyTorch 原生化重构，涉及文件：
  - `uav_project/utils/SimpleMath.py`
  - `uav_project/utils/DeltaKinematics.py`
  - `uav_project/utils/trajectory.py`
- 将原有的 `math.sin`、`math.cos`、`math.pi` 等替换为原生的 `torch.sin`、`torch.cos`、`torch.pi`，确保整个控制和轨迹生成链路的数据类型及设备一致性，进一步提升计算效率。
- 在自定义技能 `matrix-control` 中增加了处理数学函数和常数的约束，确保后续代码的张量计算标准。

### 3. SSH 远程开发无头模式适配 (SSH Headless Mode Adaptation)
- 针对 SSH 远程开发环境下无法调用显示器 (X11 DISPLAY 报错) 的问题，在 `uav_project/main.py` 中为 MuJoCo 仿真开启了 `headless=True`。
- 修改了 `uav_project/utils/logger.py`，将 Matplotlib 的 backend 设置为 `Agg`，使得仿真绘图结果可以直接无界面保存为本地图片 (`simulation_results.png`)。

### 4. 环境配置与文档规范化 (Environment & Documentation)
- 统一了开发环境，明确所有后续开发任务使用 `mujoco-sim` conda 虚拟环境。
- 更新了项目根目录的 `/environment.yml`，明确加入了核心依赖项 `torch>=2.0.0` 和 `mujoco`。
- 全面更新了 `uav_project/readme.md`，加入了详细的新手配置指南、依赖安装步骤以及运行方式说明，形成闭环的 Onboarding 文档。

## 2026-03-20

### 1. 推力计算与物理参数动态绑定
- **动态质量读取**：修复了 Delta 模型下无人机推力不足无法起飞的问题。将 `cascade_controller.py` 中硬编码的 `MASS` 修改为通过 `sum(self.uav.model.body_mass)` 动态读取 MuJoCo 模型所有刚体质量的总和，确保无论是否挂载机械臂，前馈推力都能正确计算。
- **电机参数解耦计算**：在 `config.py` 中，不再硬编码 $C_T$ 和 $C_D$。改为由 `MAX_THRUST_PER_MOTOR`、`MAX_TORQUE_PER_MOTOR` 和 `MAX_MOTOR_SPEED_KRPM` 动态计算出对应的 $C_T$ 和 $C_D$。
- **底层物理引擎参数同步**：在 `uav_model.py` 的初始化逻辑中，加入了动态覆盖 MuJoCo XML 中 `motor` 驱动器 `gear` 比例的逻辑。使得底层仿真物理引擎严格与 `config.py` 中的数学模型对齐。

### 2. Delta 机械臂运动学公式与坐标系说明文档
- 修复了 `DeltaKinematics.py` 中的轨迹跟踪问题：恢复了故意加入的符号反转和 90 度坐标系旋转矩阵，以适配 MuJoCo 中的坐标系定义。
- 编写了详尽的 `DELTA_KINEMATICS_MATH.md`，使用 LaTeX 公式完整推导了 Delta 机械臂的逆运动学（IK）和正运动学（FK），并特别记录了代码实现中的“折中策略”（如为了适配底层 XML 定义而故意导致的 IK/FK 不闭环问题）。

### 3. 数据可视化 (Logger) 与终端打印深度优化
- **自适应终端打印**：修改 `uav_model.py` 的打印逻辑，如果加载的是无机械臂模型 (`UAV.xml`)，则自动隐藏关于末端平台的冗余全 0 数据。
- **PID 串级分层可视化**：重构 `logger.py` 的 Matplotlib 绘图布局。现在生成的 `simulation_results.png` 严格按照串级 PID 的层级排列，依次展示：位置(XYZ)、速度(Vx/Vy/Vz)、姿态欧拉角(Roll/Pitch/Yaw)、角速度(P/Q/R)、控制输出(推力F/力矩M) 以及 四个电机的单独推力分配。极大提升了调试效率。

### 4. 隔离排查：四转子混控模式 vs 质心直接控制模式
为了排查无人机在引入姿态控制后出现的抖动与不稳定现象，进行了控制模式的隔离测试：
- **四转子电机混控模式 (Mixer Control)**：控制流为 `PID -> F/M -> Mixer逆解 -> 4个电机 kRPM^2 -> MuJoCo rotor 驱动器`。这是实际物理机的控制方式，但目前表现不稳定。
- **质心直接控制模式 (Direct Force Control)**：修改 XML 开启 `forcex`~`Mz` 驱动器，修改代码直接将 PID 算出的 F/M 发送给质心，跳过 Mixer。
- **测试结论**：在“质心直接控制模式”下，无人机能够稳定追踪轨迹（尽管响应略慢）。这**证明了外环 PID 控制器（位置、速度、姿态、角速度）的大致逻辑与参数是正确的**。
- **下一步方向**：确认导致不稳定的核心问题出在内环的 **Mixer 混控解算**、**电机参数映射**或 **XML 驱动器定义（如扭矩正负号方向与 Mixer 矩阵不匹配）** 上。目前已将代码回退为“四转子电机混控模式”以便集中修复 Mixer 问题。

### 5. 修复姿态控制逻辑与跨模型 PID 参数自适应
- **修复机体系四元数误差计算**：在 `controllers/pid.py` 的 `AttitudePID3D` 中，将世界系下的旋转误差计算 `quat_error = tar_q * cur_conj` 修正为了机体系下的标准 SO(3) 误差计算公式 `quat_error = cur_conj * tar_q`，彻底解决了无人机发生偏航后姿态补偿方向错误导致“倒扣”炸机的致命 Bug。
- **解决挂载 Delta 机械臂后的飘逸问题**：
  - 修复了 `cascade_controller.py` 中动态质量的计算逻辑，通过 `sum(m for m in self.uav.model.body_mass if m > 0)` 过滤了 `worldbody` 和无质量刚体，确保系统能够获取最准确的总挂载质量（纯无人机约为 0.4kg，挂载 Delta 后更重）。
  - **UAV 与 Delta 模型的 PID 刚度对比总结**：由于挂载 Delta 机械臂后整个飞行器系统的总质量和转动惯量发生了巨大的改变（特别是机械臂挥动带来的耦合扰动），原先针对轻量级模型的软 PID 参数会导致严重飘逸。
  - 最终确立的强鲁棒性 PID 参数 (`config.py`)：
    - **位置与速度环 (外环)**：放缓起飞瞬间的过冲，但增强了积分项（`VEL_KI=[0.5, 0.5, 1.0]`）以对抗重载情况下的稳态下坠。`POS_KP` 提升至 `2.0`，`VEL_KP` 提升至 `[2.0, 2.0, 3.0]`。
    - **姿态与角速度环 (内环)**：必须大幅提高刚度来压制机械臂的扰动力矩。`ATT_KP` 提升至 `[10.0, 10.0, 6.0]`，`RATE_KP` 暴增至 `[0.15, 0.15, 0.05]`。这套硬参数能保证无人机即使在下挂重物剧烈甩动时也能维持极高的水平稳定性。

## 2026-03-21

### 1. 不同模型下的 PID 参数总结与调优
针对无人机在执行圆形轨迹时出现的“飘移”和“倒扣”问题，主要原因是原有的 PID 参数无法同时适应轻载的 `UAV.xml` 和重载且具有扰动的 `Delta.xml` 模型。为了保证控制器的鲁棒性，针对两种模型分别总结了推荐的 PID 参数：

#### 针对纯无人机模型 (`UAV.xml`)
纯无人机质量较轻（约 0.4kg），转动惯量小，机动性强。使用过大的内环 PID 增益会导致高频振荡，甚至在剧烈机动时因电机饱和而失控（倒扣）。
推荐参数（较柔和）：
- **位置控制 (POS)**: `KP = [2.0, 2.0, 2.0]`
- **速度控制 (VEL)**: `KP = [2.0, 2.0, 3.0]`, `KI = [0.2, 0.2, 0.5]`, `KD = [0.05, 0.05, 0.05]`
- **姿态控制 (ATT)**: `KP = [6.0, 6.0, 4.0]`
- **角速度控制 (RATE)**: `KP = [0.08, 0.08, 0.03]`, `KI = [0.02, 0.02, 0.01]`, `KD = [0.001, 0.001, 0.0005]`

#### 针对挂载机械臂模型 (`Delta.xml`)
挂载 Delta 机械臂后，系统总质量显著增加，且机械臂下垂导致重心改变，在飞行时会产生钟摆效应和强烈的力矩耦合扰动。此时必须大幅提高内外环刚度，特别是积分项（对抗稳态误差）和内环角速度的微分项（增加阻尼，抑制抖动）。
推荐参数（强刚度）：
- **位置控制 (POS)**: `KP = [2.5, 2.5, 2.5]`
- **速度控制 (VEL)**: `KP = [3.0, 3.0, 4.0]`, `KI = [1.0, 1.0, 1.5]`, `KD = [0.1, 0.1, 0.1]`
- **姿态控制 (ATT)**: `KP = [15.0, 15.0, 8.0]`
- **角速度控制 (RATE)**: `KP = [0.25, 0.25, 0.1]`, `KI = [0.1, 0.1, 0.05]`, `KD = [0.01, 0.01, 0.005]`

*后续优化建议：可以在代码中根据加载的 XML 模型名称，动态加载对应的 PID 参数配置，以避免频繁手动修改 `config.py`。*

### 2. 强化学习 (RL) 框架搭建与集成
为了支持后续的强化学习科研需求，在项目中引入了基于 Gymnasium 和 Stable-Baselines3 的 RL 训练框架。
- **层次化环境设计**：参考 `gym-pybullet-drones` 架构，创建了 `rl_envs` 包，包含三层继承结构：
  - `BaseMujocoAviary`: 底层物理封装，接管 MuJoCo 步进 (`mj_step`) 和 Viewer 渲染逻辑。
  - `BaseRLMujocoAviary`: RL 适配层，定义 `[-1, 1]` 的标准化动作空间、扁平化的 12 维运动学观测空间，并实现了基于 `deque` 的历史动作缓存 (Action Buffer) 拼接，解决控制延迟带来的非马尔可夫问题。
  - `HoverMujocoAviary`: 具体任务层，实现了单机悬停任务 (`TARGET_POS = [0, 0, 1]`)，基于欧氏距离定义了 Reward 函数，并增加了防翻车/出界的 Truncated 截断条件以提升训练效率。
- **环境验证**：编写 `test_rl_env.py`，顺利通过 `stable-baselines3` 的 `check_env` 接口测试。
- **并行训练流水线**：编写 `train_rl.py`，使用 `make_vec_env` 开启多进程环境加速采样，配置 PPO (MlpPolicy) 算法，并挂载 `EvalCallback` 实现最佳模型的自动评估与保存。
- **可视化与测试**：编写 `enjoy_rl.py` 和配套的启动脚本 `run_enjoy.sh`。解决了 SSH 远程调用渲染时的 `mujoco-python-viewer` 依赖缺失问题，并利用 `LD_PRELOAD` 成功修复了 conda 环境中 `matplotlib` 引起的 `libstdc++.so.6: version CXXABI_1.3.15 not found` C++ 库版本冲突。

---

## 2026-03-23

### 1. 强化学习 (RL) 框架扩展：Delta 机械臂环境适配
在基础单机悬停环境之上，为了让强化学习算法能够控制搭载了 Delta 机械臂的无人机，完成了从单机到“无人机+机械臂”耦合系统的环境扩展。

- **动作空间扩展与推力映射重构** (`HoverDeltaMujocoAviary`)：
  - 动作空间从单纯的 4 个旋翼扩展到了 **7 维**（4 个旋翼 + 3 个 Delta 机械臂关节）。
  - **动态推力基准计算**：在 `BaseRLMujocoAviary` 中，通过 `mujoco.mj_getTotalmass()` 动态读取当前加载模型的总质量，并基于配置的 $C_T$ 常数计算出标准的悬停转速 `hover_rpm`。
  - **推力映射放宽**：为了让无人机有足够的动力起飞并抵抗机械臂带来的重力矩，将旋翼的动作映射范围从悬停转速的 `±5%` 大幅放宽至 `±50%`，即 `target_rpm = hover_rpm * (1.0 + 0.5 * action)`。
  - **机械臂关节映射**：将 `[-1, 1]` 的 RL 动作输出线性映射到 Delta 机械臂安全的物理关节限位 `[-1.57, 0.523]` (rad) 范围内。

### 2. 强化学习任务 1：带载悬停 (Hover Delta Task)
**任务目标**：让挂载了 Delta 机械臂的无人机在目标高度 (`z=1.5m`) 稳定悬停，且不发生严重抖动。
- **状态空间 (Observation Space)**：
  - 包含无人机的 13 维运动学状态（位置 3、四元数 4、线速度 3、角速度 3）。
  - 拼接历史动作缓存（Action Buffer），帮助策略网络推断动态趋势和系统延迟。
- **奖励函数设计 (Reward Function)**：
  - **位置奖励**：基于到目标点的欧氏距离平方，距离越近奖励越高，最大为 2.0。
  - **稳定性惩罚**：对过高的角速度（防抖动）、线速度（防乱飞）以及动作变化率（防电机剧烈震荡）进行惩罚。
  - **坠机惩罚**：如果高度低于 0.2m，给予强烈的负向惩罚（-2.0），以打破早期智能体“趴在地上不飞以换取零速度惩罚”的局部最优解。
- **训练成果**：通过调整推力上限并修复时间截断逻辑，模型成功在 PPO 算法下达到了 1800+ 的高分，实现了稳定的带载悬停。

### 3. 强化学习任务 2：末端轨迹追踪 (Track Delta Task)
**任务目标**：在无人机保持空间悬停（不晃动）的同时，控制下挂的 Delta 机械臂末端执行器（End-Effector）在机体下方画出一个预设的圆形轨迹。
- **状态空间扩展**：
  - 继承悬停环境的状态，并额外拼接了 9 维数据：**目标末端位置 (3维)**、**当前末端误差 (3维)** 以及最重要的 **目标轨迹速度 (3维，提供前馈预测信息)**。
- **坐标系校准**：修复了世界坐标系与机体坐标系的混淆问题。目标轨迹方程被定义为相对于无人机当前位置的相对偏移，从而使末端误差计算 `ee_dist` 具有正确的物理指导意义。
- **重塑奖励函数 (Reward Shaping)**：
  - **防局部最优**：初期智能体发现保持机械臂不动（误差约 0.12m）比乱动导致的姿态失稳更“划算”。为此，将追踪奖励的最大值从 2.0 提高到 10.0，并将惩罚变陡峭：当误差大于 0.05m 时，奖励直接变为负数（最低 -5.0），逼迫智能体必须移动机械臂。
  - **姿态倾斜惩罚 (Tilt Penalty)**：为了满足“无人机不发生晃动”的要求，直接提取无人机四元数的 Z 轴分量，增加 `tilt_penalty = 5.0 * (1.0 - up_z)`。强迫智能体通过更精细的 4 旋翼差速来抵抗机械臂挥动产生的力矩，而不是通过倾斜整个机身来代偿。

### 4. 训练基础设施与工程优化
- **时间预估与进度追踪**：为长时间（如 10 小时以上）的强化学习训练实现了自定义的 `TimeLoggerCallback`，能够实时计算剩余时间（ETA）、完成百分比以及总耗时。
- **多进程训练环境加固**：
  - 将环境并行数提升至 32 (`SubprocVecEnv`) 以榨干 CPU 性能。
  - 修复了因为 `__init__` 初始化时序问题导致的子进程环境找不到 `self.uav` 属性而频繁引发的 `EOFError` 崩溃问题。
- **环境参数动态化**：在所有的 RL 环境中实现了 `episode_duration` 的动态配置，允许在训练（如 10s）和测试（如 30s）时使用不同的回合长度，从而检验模型在超出训练时长的外推泛化能力。

---

## 2026-03-24

### 1. 强化学习抗扰动悬停任务重构 (Disturbance Rejection RL Task)
**任务目标**：让强化学习智能体仅控制无人机的 4 个旋翼实现极度稳定的悬停，同时完全解耦下挂 Delta 机械臂的控制权，将其作为“不可控的外部物理扰动源”。
- **动作空间降维**：创建 `DisturbanceHoverDeltaMujocoAviary` 环境，动作空间从 7 维降回 4 维（仅限旋翼推力）。此举彻底封死了智能体试图“锁死机械臂不动”的走捷径行为（策略坍缩）。
- **状态空间对齐与动态推断**：
  - 修复了 `BaseRLMujocoAviary` 在多态继承下硬编码动作维度导致的 `ValueError` 广播错误和 PyTorch `mat1 and mat2 shapes cannot be multiplied` 矩阵维度不匹配问题。
  - 实现了基于 `_computeObs()` 试运行的观测空间维度动态推断（Obs Dim = 88），并加入了针对 PPO 严格输入校验的 NumPy `pad/slice` 边界保护。

### 2. 独立笛卡尔空间轨迹生成器 (Cartesian Trajectory Generator)
- **第一性原理重构**：为了生成符合物理现实的机械臂扰动，废弃了直接向三个电机注入独立正弦波的错误做法（会导致不可达奇点和无穷大内力导致 MuJoCo 崩溃）。
- **模块化解耦**：抽取出了独立的 `DeltaRandomTrajectoryGenerator` 类。它负责在 3D 笛卡尔任务空间内生成随机的李萨如（Lissajous）曲线轨迹，并提供基于连续时间 $t$ 的状态查询接口。
- **预见性工作空间校验 (Workspace Validation)**：生成器在每次 `reset` 时，会主动前瞻性地对未来 10 秒的轨迹进行 100 次离散采样，利用 `DeltaKinematics.ik()` 进行逆解。只有当所有点都物理可达（不为 -1）时，才会放行该轨迹供 RL 环境或 PID 控制器使用。

### 3. 强化学习训练与泛化验证增强
- **强鲁棒性 Reward 函数重塑**：
  - 将位置误差的奖励函数从二次多项式改为指数衰减 (`3.0 * np.exp(-3.0 * dist)`)，在目标点附近形成极陡的梯度，逼迫智能体追求“钉子般”的极致悬停。
  - 引入显式的姿态倾斜惩罚 (`tilt_penalty = 0.5 * (1.0 - up_z)`)，强迫智能体学习使用旋翼差速来抵消机械臂挥动的反作用扭矩，而不是通过“歪着机身”来代偿。
- **强制跑满全量训练**：为了让网络充分见识到各种随机生成的机械臂扰动轨迹，在 `train_disturbance_hover_rl.py` 中将 `reward_threshold` 设置为无穷大，禁用了提前停止（Early Stopping），并将总训练步数提升了数倍以保证深度收敛。
- **多场景连续测试脚本**：重构 `enjoy_disturbance_hover_rl.py`，支持在一次运行中连续执行多次独立的 15 秒测试。每次测试都会被注入一组全新随机化（不同振幅、频率、相位）的机械臂扰动，并利用更新后的 `Logger` 自动生成包含 UAV 位置偏差与 Delta 末端 3D 轨迹的详细对比图表。

### 4. MuJoCo 原生仿真渲染时钟同步 (Phase-Locked Loop)
- **解决渲染“快进”与卡顿问题**：在 `simulator.py` 中引入了软件层面的锁相环（PLL）机制。通过对比仿真物理计算的挂钟时间（Wall-Clock Time）与设定的渲染帧率（如 60 FPS），动态插入 `time.sleep()`。这使得纯 C++ 驱动的高速 `main.py` 仿真也能像带有网络推理开销的 RL 环境一样，完美保持 1.0x 的实时因子（Real-Time Factor），实现了丝滑的可视化体验。

---

## 2026-03-25

### 1. 强化学习基础框架规范化与对齐 (RL Framework Standardization)
- **测试/评估管线工程化升级**：重构了早期单机任务的测试脚本 (`enjoy_rl.py` 和 `enjoy_track_rl.py`)，使其完全对齐抗扰动任务 (`enjoy_disturbance_hover_rl.py`) 的高标准工程架构：
  - 引入规范的多回合测试流程（如 `NUM_TESTS`、`TEST_DURATION`）。
  - 废弃强制 `time.sleep()`，改用基于控制频率计算的“抽帧渲染”机制，彻底释放物理引擎算力。
  - 全面集成 `Logger` 模块，实现所有测试流程的数据落盘与自动化图表生成。
- **训练管线统一与多进程加速**：更新 `train_rl.py` 和 `train_track_rl.py`：
  - 弃用单进程的 `DummyVecEnv`，全面升级为 `SubprocVecEnv` (32核并行)，大幅缩短数据采样时间。
  - 引入全局配置 (`config.py`) 动态计算 `total_timesteps` 和评估频率。
  - 统一禁用提前停止 (`reward_threshold=np.inf`) 并引入 `TimeLoggerCallback` 以实时监控训练 ETA 进度。
- **重塑早期任务的奖励函数**：将单机悬停与轨迹跟踪环境的 `_computeReward` 和 `_computeTruncated` 逻辑升级为抗扰动任务的最新“严苛”标准。引入基于指数衰减的位置奖励，并增加针对线速度、角速度、动作突变及姿态倾斜的显式惩罚项，从根源上消除了无人机高频震荡和“横冲直撞”的次优策略。

### 2. 物理引擎步长 (Timestep) 与控制频率解耦架构优化
- **核心问题解决**：发现了之前所有无机械臂模型 (`UAV.xml`) 被默认降级到 100Hz (0.01s) 物理更新率导致动力学积分粗糙的问题。
- **基于第一性原理的架构修正 (Single Source of Truth)**：
  - 废除了在 Python 代码中硬编码传入 `freq` 覆盖物理引擎精度的错误做法。
  - 确立 **XML 文件为物理属性的唯一真实来源**：在 `UAV.xml` 中将 `timestep` 定义为 `0.001` (1000Hz)，在 `Delta.xml` 中保持 `0.0001` (10000Hz)。
  - 重构 `BaseMujocoAviary.__init__`：自动读取 `self.model.opt.timestep` 并转换为物理运行频率 `self.freq`。
- **完美的齿轮比映射**：上层强化学习网络仍由 `config.py` 中的 `RL_CONTROL_FREQ = 100` 决定（100Hz），底层环境 (`BaseRLMujocoAviary`) 会自动计算并执行 `physics_steps_per_control = int(self.freq / self.control_freq)`。这保证了不同复杂度模型能在各自最适宜的物理精度下计算，且对上层 AI 智能体完全透明。

### 3. 排查并明确 `__init__` 与 `reset` 调用的初始化时序问题
- 分析了 Gymnasium `make_vec_env` 实例化期间自动调用 `reset()`（进而调用 `_computeObs`）可能导致的类属性缺失报错（即 Dirty Pattern 小瑕疵）。
- 确认当前项目通过在子类中预设 `if not hasattr(self, 'uav'):` 保护逻辑，以及在 `BaseRLMujocoAviary` 顶层提前赋值 `self.control_freq` 的做法已经实现了安全的降级防御（Fallback），保证了多进程环境初始化的绝对稳定。

---

*注：本次更新标志着 UAV-Delta 仿真核心控制链路的 PyTorch 化重构基本完成，并成功跑通了首个基于 MuJoCo 的强化学习单机悬停模型训练与渲染闭环。*
