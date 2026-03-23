import numpy as np
import gymnasium as gym
from .BaseRLMujocoAviary import BaseRLMujocoAviary

class TrackCircularMujocoAviary(BaseRLMujocoAviary):
    """
    Tracking task environment for MuJoCo UAV.
    The goal is to track a dynamic target moving in a circular trajectory.
    """
    def __init__(self, 
                 max_steps=2000, 
                 center=np.array([0.0, 0.0, 1.0]),
                 radius=1.0,
                 angular_velocity=0.5,
                 **kwargs):
        
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.angular_velocity = float(angular_velocity)
        
        self.max_steps = max_steps
        self.step_counter = 0
        
        super().__init__(**kwargs)
        
    def _getTargetPos(self) -> np.ndarray:
        """
        Computes the target position at the current simulation time.
        """
        t = self.data.time
        x = self.center[0] + self.radius * np.cos(self.angular_velocity * t)
        y = self.center[1] + self.radius * np.sin(self.angular_velocity * t)
        z = self.center[2]
        return np.array([x, y, z], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        return super().reset(seed=seed, options=options)
        
    def step(self, action):
        self.step_counter += 1
        return super().step(action)
        
    def _observationSpace(self) -> gym.Space:
        """
        Overrides the base observation space to include the dynamic target position
        and relative error.
        Original obs + target_pos (3) + relative_pos (3)
        """
        base_space = super()._observationSpace()
        obs_dim = base_space.shape[0] + 6  # 3 for target_pos, 3 for relative_pos
        
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def _computeObs(self) -> np.ndarray:
        """
        Computes the current observation, appending target position and relative position.
        """
        base_obs = super()._computeObs()
        target_pos = self._getTargetPos()
        current_pos = self.data.qpos[:3].copy()
        
        relative_pos = target_pos - current_pos
        
        # Concatenate base observations with target and relative positions
        obs = np.concatenate([base_obs, target_pos, relative_pos])
        return obs.astype(np.float32)
        
    def _computeReward(self) -> float:
        """
        Computes the reward to penalize distance from the dynamic target.
        """
        target_pos = self._getTargetPos()
        current_pos = self.data.qpos[:3]
        dist = np.linalg.norm(target_pos - current_pos)
        
        # Penalize distance, encouraging the UAV to stay close to the target.
        # Max reward of 2.0 when distance is 0, smoothly decreasing.
        reward = float(np.max([0.0, 2.0 - dist**4]))
        return reward
        
    def _computeTerminated(self) -> bool:
        """
        Computes whether the episode is terminated.
        For tracking, we usually do not terminate early just because we are close to the target,
        as the target keeps moving.
        """
        return False
        
    def _computeTruncated(self) -> bool:
        """
        Computes whether the episode is truncated.
        Out of bounds, tilted too much, or max steps reached.
        """
        current_pos = self.data.qpos[:3]
        x, y, z = current_pos
        
        # Check boundaries (looser boundaries to allow following the circular path)
        out_of_bounds = abs(x) > 3.0 or abs(y) > 3.0 or z > 4.0 or z < 0.0
        
        # Check tilt
        quat = self.data.qpos[3:7]
        # In MuJoCo, quat is typically [w, x, y, z]
        w, qx, qy, qz = quat
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        too_tilted = up_z < 0.0 # Tilted more than 90 degrees
        
        # Check max steps
        max_steps_reached = self.step_counter >= self.max_steps
        
        return bool(out_of_bounds or too_tilted or max_steps_reached)
        
    def _computeInfo(self) -> dict:
        """
        Computes the info dictionary.
        """
        target_pos = self._getTargetPos()
        current_pos = self.data.qpos[:3]
        dist = np.linalg.norm(target_pos - current_pos)
        
        return {
            "target_pos": target_pos.tolist(),
            "current_pos": current_pos.tolist(),
            "tracking_error": float(dist)
        }
