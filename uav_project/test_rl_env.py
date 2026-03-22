import sys
import os

# Add the parent directory to sys.path so that 'uav_project' is recognized as a package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.env_checker import check_env
from uav_project.rl_envs.HoverMujocoAviary import HoverMujocoAviary

def main():
    print("Initializing HoverMujocoAviary...")
    env = HoverMujocoAviary()
    
    print("Running check_env...")
    try:
        check_env(env, warn=True)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return
    
    print("Running random action loop for 100 steps...")
    obs, info = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            
    print("Random action loop completed successfully. No crashes!")
    env.close()

if __name__ == "__main__":
    main()
