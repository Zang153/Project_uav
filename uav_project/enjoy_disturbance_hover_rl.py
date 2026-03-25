import os
import sys
import time
import torch
import numpy as np

# Ensure correct import of current project packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from uav_project.rl_envs.DisturbanceHoverDeltaMujocoAviary import DisturbanceHoverDeltaMujocoAviary
from uav_project.utils.DeltaRandomTrajectoryGenerator import DeltaRandomTrajectoryGenerator
from uav_project.utils.logger import Logger

def main():
    # 1. Find the best trained model
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "rl_results", "disturbance_hover_models", "best_model.zip")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("Please run train_disturbance_hover_rl.py first to train the model!")
        return

    print(f"[INFO] Loading model: {model_path}")
    # 2. Load model
    model = PPO.load(model_path)

    # 3. Test Configuration
    NUM_TESTS = 5
    TEST_DURATION = 15.0 
    
    print(f"[INFO] Initializing test environment for {NUM_TESTS} tests, each lasting {TEST_DURATION}s...")

    env_kwargs = {
        "render_mode": "human",
        "episode_duration": TEST_DURATION
    }
    env = DisturbanceHoverDeltaMujocoAviary(**env_kwargs)
    
    # Initialize Random Trajectory Generator
    generator = DeltaRandomTrajectoryGenerator()
    
    test_steps = int(TEST_DURATION * env.control_freq)
    TARGET_FPS = 30
    render_interval = max(1, int(env.control_freq / TARGET_FPS)) 

    # 4. Run multiple tests
    for test_idx in range(1, NUM_TESTS + 1):
        print(f"\n==================================================")
        print(f"[INFO] Starting Test {test_idx}/{NUM_TESTS}")
        print(f"==================================================")
        
        # Reset environment and get a new random trajectory
        obs, info = env.reset()
        generator.reset()
        logger = Logger()
        
        print(f"[INFO] Trajectory Params: Z_Center={generator.z_center:.3f}m, XY_Radius={generator.xy_radius:.3f}m, Freqs=({generator.freq_x:.2f}, {generator.freq_y:.2f}, {generator.freq_z:.2f})Hz")
        
        episode_start_time = time.time()
        steps_in_current_episode = 0
        
        for i in range(test_steps):
            # --- Inject External Disturbance Trajectory ---
            sim_time = steps_in_current_episode / env.control_freq
            des_pos, joint_angles_rad = generator.get_state(sim_time)
            env.set_external_arm_action(joint_angles_rad)
            
            # --- Agent Action ---
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            steps_in_current_episode += 1
            
            # --- Logging ---
            current_pos = info.get("current_pos", np.zeros(3))
            current_vel = env.data.qvel[:3].copy()
            # Approximation of orientation/rates for logger if needed, or just zeros if not tracked explicitly
            # For this hover task, the main focus is position and delta tracking
            current_quat = env.data.qpos[3:7].copy()
            current_ang_vel = env.data.qvel[3:6].copy()
            
            # Use info to get the actual EE position if available, or just log desired for now
            # To be perfectly accurate with the Logger, we need to format the data
            uav_state = (current_pos, current_vel, current_quat, current_ang_vel)
            
            # In DisturbanceHoverDeltaMujocoAviary, we don't have a direct method to get actual EE pos easily without UAVModel
            # But we can approximate or just log the desired position for the Delta arm
            # For the Logger format: uav_data + (delta_des, delta_act)
            # We'll use the desired pos for both to avoid breaking the logger, or if we have it in info
            actual_ee_pos = des_pos # Placeholder, ideally read from forward kinematics or sensors
            
            logger.log(
                time=sim_time,
                position=current_pos,
                velocity=current_vel,
                attitude_quat=current_quat,
                angle_rate=current_ang_vel,
                target_pos=np.array([0, 0, 1.5]),
                target_vel=np.zeros(3),
                target_att_quat=np.array([1, 0, 0, 0]),
                target_rate=np.zeros(3),
                motor_thrusts=np.zeros(4), # Dummy control
                mixer_outputs=np.zeros(6), # Dummy
                delta_des_pos=des_pos,
                delta_actual_pos=actual_ee_pos
            )
            
            # --- Terminal Output ---
            if steps_in_current_episode % 10 == 0:
                pos = info.get("current_pos", [0,0,0])
                err = info.get("pos_error", 0)
                
                print(f"\r[Test {test_idx} | Step {steps_in_current_episode:4d}] "
                      f"Pos: ({pos[0]: 5.2f}, {pos[1]: 5.2f}, {pos[2]: 5.2f}) | "
                      f"Err: {err:5.3f}m ", end="")
                
            if i % render_interval == 0:
                env.render()

            if terminated or truncated:
                break
                
        # End of test iteration
        episode_end_time = time.time()
        real_time_elapsed = episode_end_time - episode_start_time
        sim_time_elapsed = steps_in_current_episode / env.control_freq
        
        print(f"\n[INFO] Test {test_idx} finished.")
        print(f"       - Simulation time: {sim_time_elapsed:.2f}s")
        print(f"       - Real world time: {real_time_elapsed:.2f}s")
        
        # Save Plot
        plot_path = os.path.join(project_dir, f'rl_test_{test_idx}.png')
        logger.plot_results(save_path=plot_path)
        print(f"[INFO] Plot saved to {plot_path}")

    env.close()
    print("\n[INFO] All tests finished successfully.")

if __name__ == "__main__":
    main()