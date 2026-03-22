import numpy as np
from .BaseRLMujocoAviary import BaseRLMujocoAviary

class HoverMujocoAviary(BaseRLMujocoAviary):
    """
    Hover task environment for MuJoCo UAV.
    The goal is to hover at a target position [0, 0, 1].
    """
    def __init__(self, max_steps=1000, **kwargs):
        self.TARGET_POS = np.array([0, 0, 1], dtype=np.float32)
        self.max_steps = max_steps
        self.step_counter = 0
        super().__init__(**kwargs)
        
    def reset(self, seed=None, options=None):
        self.step_counter = 0
        return super().reset(seed=seed, options=options)
        
    def step(self, action):
        self.step_counter += 1
        return super().step(action)
        
    def _computeReward(self) -> float:
        """
        Computes the reward for hovering near the target.
        e.g. max(0, 2 - norm(target_pos - current_pos)**4)
        """
        current_pos = self.data.qpos[:3]
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        reward = float(np.max([0.0, 2.0 - dist**4]))
        return reward
        
    def _computeTerminated(self) -> bool:
        """
        Computes whether the episode is terminated.
        e.g. distance < 0.01
        """
        current_pos = self.data.qpos[:3]
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        return bool(dist < 0.01)
        
    def _computeTruncated(self) -> bool:
        """
        Computes whether the episode is truncated.
        e.g. out of bounds x,y > 1.5, z > 2.0, or tilted too much, or max steps reached.
        """
        current_pos = self.data.qpos[:3]
        x, y, z = current_pos
        
        # Check boundaries
        out_of_bounds = abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0 or z < 0.0
        
        # Check tilt (simplified by checking the z-component of the up vector)
        # The quaternion in MuJoCo is [w, x, y, z] (scalar first usually, but checking formula)
        # If [w, x, y, z], then up_z = 1 - 2*(x^2 + y^2)
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        too_tilted = up_z < 0.5 # Tilted more than ~60 degrees
        
        # Check max steps
        max_steps_reached = self.step_counter >= self.max_steps
        
        return bool(out_of_bounds or too_tilted or max_steps_reached)
        
    def _computeInfo(self) -> dict:
        """
        Computes the info dictionary.
        """
        return {}
