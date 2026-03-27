import os
import sys
import time
import numpy as np

# 确保能正确导入当前项目的包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from uav_project.rl_envs.HoverMujocoAviary import HoverMujocoAviary
from uav_project.config import TRAINING_CONFIGS

class TimeLoggerCallback(BaseCallback):
    """
    Custom callback for logging the real time elapsed, ETA, and progress.
    """
    def __init__(self, total_timesteps: int, n_envs: int, n_steps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.steps_per_iteration = n_envs * n_steps
        self.total_iterations = total_timesteps // self.steps_per_iteration
        
        self.training_start_time = None
        self.iteration_start_time = None
        self.iteration_count = 0

    def _on_training_start(self) -> None:
        self.training_start_time = time.time()
        self.iteration_start_time = time.time()
        print(f"\n[INFO] Training started at {time.strftime('%X')}")
        print(f"[INFO] Total Iterations to complete: {self.total_iterations}")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        total_elapsed = time.time() - self.training_start_time
        hours, rem = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\n[INFO] Training finished at {time.strftime('%X')}")
        print(f"[INFO] Total time spent: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
    def on_rollout_start(self) -> None:
        if self.iteration_start_time is not None:
            current_time = time.time()
            elapsed = current_time - self.iteration_start_time
            self.iteration_count += 1
            
            total_elapsed = current_time - self.training_start_time
            avg_time_per_iter = total_elapsed / self.iteration_count
            remaining_iters = self.total_iterations - self.iteration_count
            eta_seconds = remaining_iters * avg_time_per_iter
            
            eta_h, eta_rem = divmod(eta_seconds, 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            
            progress = (self.iteration_count / self.total_iterations) * 100
            
            print(f"\n[TIME] Iteration {self.iteration_count}/{self.total_iterations} ({progress:.1f}%) completed in {elapsed:.2f}s. "
                  f"ETA: {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s")
            
        self.iteration_start_time = time.time()

def main():
    # 0. Load Task Config
    cfg = TRAINING_CONFIGS["uav_hover"]

    # 1. 定义保存路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(project_dir, "rl_results", cfg["model_save_dir"])
    log_path = os.path.join(project_dir, "rl_results", cfg["log_save_dir"])
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("[INFO] 初始化环境...")
    
    # 2. 创建向量化环境 (Vectorized Environment)
    num_envs = cfg["num_envs"]
    
    # 将 config 中的时长传递给环境
    env_kwargs = {"episode_duration": cfg["episode_duration_sec"]}
    
    # 使用 SubprocVecEnv 以实现真正的多进程并行加速
    env = make_vec_env(HoverMujocoAviary, n_envs=num_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    # 3. 创建评估环境与回调函数
    eval_env = make_vec_env(HoverMujocoAviary, n_envs=1, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    
    # 禁用提前停止，让模型充分训练
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=cfg["reward_threshold"], verbose=1)
    
    control_freq = eval_env.get_attr('control_freq')[0]
    eval_freq_steps = int(cfg["eval_freq_sec"] * control_freq)
    
    # 为了保证训练充分，增加训练时长
    total_train_steps = int(cfg["total_train_sec"] * control_freq)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq_steps,
        deterministic=True,
        render=False
    )
    
    n_steps = cfg["n_steps"]
    time_callback = TimeLoggerCallback(
        total_timesteps=total_train_steps, 
        n_envs=num_envs, 
        n_steps=n_steps
    )

    print("[INFO] 初始化 PPO 模型...")
    # 4. 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=cfg["learning_rate"],
        n_steps=n_steps,
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        device="cpu"
    )

    print("[INFO] 开始训练...")
    # 5. 开始训练
    try:
        model.learn(total_timesteps=total_train_steps, callback=[time_callback, eval_callback])
    except KeyboardInterrupt:
        print("\n[INFO] 用户手动中断了训练。")

    # 6. 保存最终模型
    final_model_path = os.path.join(save_path, cfg["model_name"])
    model.save(final_model_path)
    print(f"[INFO] 训练结束。最终模型已保存至: {final_model_path}")
    print(f"[INFO] 表现最好的模型保存在: {os.path.join(save_path, 'best_model.zip')}")

    env.close()

if __name__ == "__main__":
    main()
