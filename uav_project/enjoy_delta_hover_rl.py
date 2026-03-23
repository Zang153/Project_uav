import os
import sys
import time

# Ensure correct import of current project packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from uav_project.rl_envs.HoverDeltaMujocoAviary import HoverDeltaMujocoAviary
from uav_project.config import RL_EPISODE_DURATION

def main():
    # 1. Find the best trained model
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "rl_results", "delta_hover_models", "best_model.zip")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("Please run train_delta_hover_rl.py first to train the model!")
        # For testing purposes, if model doesn't exist, we will just use an untrained model or exit
        return

    print(f"[INFO] Loading model: {model_path}")
    # 2. Load model
    model = PPO.load(model_path)

    # 3. Create test environment
    print("[INFO] Initializing test environment...")
    
    # Set to True if you want to test for 20s instead of the config's 10s
    TEST_DURATION = 30.0 

    # Initialize environment with the custom duration
    env_kwargs = {
        "render_mode": "human",
        "episode_duration": TEST_DURATION
    }
    env = HoverDeltaMujocoAviary(**env_kwargs)
    
    # Calculate test steps based on episode duration and control frequency
    test_steps = int(TEST_DURATION * env.control_freq)

    # 4. Test loop
    obs, info = env.reset()
    
    print(f"[INFO] Starting test for {TEST_DURATION} seconds ({test_steps} steps)...")
    
    # Target FPS for rendering (e.g., 60 FPS)
    TARGET_FPS = 30 # Reduced to lower rendering overhead
    render_interval = max(1, int(env.control_freq / TARGET_FPS))  # Steps between renders
    
    episode_start_time = time.time()
    steps_in_current_episode = 0
    
    for i in range(test_steps):
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        steps_in_current_episode += 1
        
        # --- Live Terminal Output ---
        # Only print every 10 steps (0.1s simulation time) to avoid flooding the terminal
        if steps_in_current_episode % 10 == 0:
            pos = info.get("current_pos", [0,0,0])
            err = info.get("pos_error", 0)
            ang_vel = info.get("angular_velocity", [0,0,0])
            
            print(f"\r[Step {steps_in_current_episode:4d}] "
                  f"Pos: ({pos[0]: 5.2f}, {pos[1]: 5.2f}, {pos[2]: 5.2f}) | "
                  f"Err: {err:5.3f}m | "
                  f"AngVel: ({ang_vel[0]: 5.1f}, {ang_vel[1]: 5.1f}, {ang_vel[2]: 5.1f})", end="")
            
        # Render at target FPS without strict time.sleep to allow maximum speed
        if i % render_interval == 0:
            env.render()

        if terminated or truncated:
            episode_end_time = time.time()
            real_time_elapsed = episode_end_time - episode_start_time
            # Fix simulation time calculation to use control_freq instead of freq
            sim_time_elapsed = steps_in_current_episode / env.control_freq
            
            print(f"\n[INFO] Episode finished.")
            print(f"       - Steps survived: {steps_in_current_episode}")
            print(f"       - Simulation time: {sim_time_elapsed:.2f}s")
            print(f"       - Real world time: {real_time_elapsed:.2f}s")
            print(f"       - Real-time Factor (RTF): {(sim_time_elapsed / real_time_elapsed):.2f}x")
            print("       Resetting environment...")
            
            obs, info = env.reset()
            episode_start_time = time.time()
            steps_in_current_episode = 0

    env.close()
    print("[INFO] Test finished.")

if __name__ == "__main__":
    main()
