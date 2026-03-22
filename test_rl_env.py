import os
import sys

# Add project root to sys.path to ensure absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uav_project.rl_envs import BaseRLMujocoAviary

class TestEnv(BaseRLMujocoAviary):
    def _computeReward(self): return 0.0
    def _computeTerminated(self): return False
    def _computeTruncated(self): return False
    def _computeInfo(self): return {}

try:
    env = TestEnv()
    print("Environment created successfully!")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    
    action = env.action_space.sample()
    print(f"Sample Action: {action}")
    
    obs, reward, term, trunc, info = env.step(action)
    print(f"Post-step Observation Shape: {obs.shape}")
    print("Test passed.")
except Exception as e:
    print(f"Test failed with error: {e}")
