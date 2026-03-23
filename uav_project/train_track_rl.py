import os
import sys

# 确保能正确导入当前项目的包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from uav_project.rl_envs.TrackCircularMujocoAviary import TrackCircularMujocoAviary

def main():
    # 1. 定义保存路径
    save_path = "./rl_results/track_models/"
    log_path = "./rl_results/track_logs/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("[INFO] 初始化环境...")
    # 2. 创建向量化环境 (Vectorized Environment)
    # n_envs=4 表示同时开启 4 个仿真环境并行收集数据，大大加快训练速度
    env = make_vec_env(TrackCircularMujocoAviary, n_envs=4)

    # 3. 创建评估环境与回调函数
    # 训练过程中，我们需要一个独立的环境来测试模型当前的能力，以保存最好的模型
    eval_env = make_vec_env(TrackCircularMujocoAviary, n_envs=1)
    
    # 当模型的平均奖励达到某个阈值时，自动停止训练（可选）
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1800, verbose=1)
    
    # 评估回调：每 10000 步评估一次，保存表现最好的模型到 save_path
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=10000,
        deterministic=True, # 测试时使用确定性动作，而不是随机采样
        render=False
    )

    print("[INFO] 初始化 PPO 模型...")
    # 4. 初始化 PPO 模型
    # MlpPolicy: 表示使用多层感知机 (全连接神经网络)
    # tensorboard_log: 用于后续使用 Tensorboard 查看训练曲线
    # device="cpu" 强制使用 CPU 进行训练，避免因为驱动崩溃导致的 CUDA error
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="cuda"
    )

    print("[INFO] 开始训练...")
    # 5. 开始训练
    # total_timesteps 是与环境交互的总步数
    try:
        model.learn(total_timesteps=1_000_000, callback=eval_callback)
    except KeyboardInterrupt:
        print("\n[INFO] 用户手动中断了训练。")

    # 6. 保存最终模型
    final_model_path = os.path.join(save_path, "ppo_track_final")
    model.save(final_model_path)
    print(f"[INFO] 训练结束。最终模型已保存至: {final_model_path}")
    print(f"[INFO] 表现最好的模型保存在: {os.path.join(save_path, 'best_model.zip')}")

    env.close()

if __name__ == "__main__":
    main()
