import os
import sys
import time

# Ensure correct import of current project packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
import numpy as np
from uav_project.rl_envs.DisturbanceHoverDeltaMujocoAviary import DisturbanceHoverDeltaMujocoAviary
from uav_project.config import RL_EVAL_FREQ_SEC, RL_TOTAL_TRAIN_SEC, RL_EPISODE_DURATION

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
            
            # Calculate ETA
            total_elapsed = current_time - self.training_start_time
            avg_time_per_iter = total_elapsed / self.iteration_count
            remaining_iters = self.total_iterations - self.iteration_count
            eta_seconds = remaining_iters * avg_time_per_iter
            
            # Format times
            eta_h, eta_rem = divmod(eta_seconds, 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            
            progress = (self.iteration_count / self.total_iterations) * 100
            
            print(f"\n[TIME] Iteration {self.iteration_count}/{self.total_iterations} ({progress:.1f}%) completed in {elapsed:.2f}s. "
                  f"ETA: {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s")
            
        self.iteration_start_time = time.time()

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. Define save paths
    save_path = os.path.join(project_dir, "rl_results", "disturbance_hover_models")
    log_path = os.path.join(project_dir, "rl_results", "disturbance_hover_logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("[INFO] Initializing environment...")
    RENDER_TRAINING = False
    render_mode = "human" if RENDER_TRAINING else None

    # 2. Create Vectorized Environment
    num_envs = 32
    
    env_kwargs = {
        "render_mode": render_mode,
        "episode_duration": RL_EPISODE_DURATION
    }
    env = make_vec_env(DisturbanceHoverDeltaMujocoAviary, n_envs=num_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    # 3. Create evaluation environment and callbacks
    eval_env_kwargs = {"episode_duration": RL_EPISODE_DURATION}
    eval_env = make_vec_env(DisturbanceHoverDeltaMujocoAviary, n_envs=1, vec_env_cls=SubprocVecEnv, env_kwargs=eval_env_kwargs)
    
    # We want to force it to train fully, so we set a practically unreachable reward threshold
    # or just disable the threshold stopping by setting it to infinity
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=np.inf, verbose=1)
    
    control_freq = eval_env.get_attr('control_freq')[0]
    eval_freq_steps = int(RL_EVAL_FREQ_SEC * control_freq)
    
    # Increase training time significantly due to the more complex Cartesian disturbance space
    increased_train_sec = RL_TOTAL_TRAIN_SEC * 5 # Train for much longer to ensure robustness
    total_train_steps = int(increased_train_sec * control_freq)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq_steps,
        deterministic=True,
        render=False
    )
    
    n_steps = 4096
    time_callback = TimeLoggerCallback(
        total_timesteps=total_train_steps, 
        n_envs=num_envs, 
        n_steps=n_steps
    )

    print("[INFO] Initializing PPO model...")
    # 4. Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=10,
        device="cpu"
    )

    print("[INFO] Starting training...")
    try:
        model.learn(total_timesteps=total_train_steps, callback=[time_callback, eval_callback])
    except KeyboardInterrupt:
        print("\n[INFO] User manually interrupted training.")

    # 6. Save final model
    final_model_path = os.path.join(save_path, "ppo_disturbance_hover_final")
    model.save(final_model_path)
    print(f"[INFO] Training finished. Final model saved to: {final_model_path}")
    print(f"[INFO] Best model saved to: {os.path.join(save_path, 'best_model.zip')}")

    env.close()

if __name__ == "__main__":
    main()