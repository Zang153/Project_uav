import numpy as np
from .BaseRLMujocoAviary import BaseRLMujocoAviary

class HoverMujocoAviary(BaseRLMujocoAviary):
    """
    Hover task environment for MuJoCo UAV.
    The goal is to hover at a target position [0, 0, 1].
    """
    def __init__(self, episode_duration=10.0, **kwargs):
        self.TARGET_POS = np.array([0, 0, 1], dtype=np.float32)
        
        super().__init__(**kwargs)
        
        self.episode_duration = episode_duration
        self.max_steps = int(self.episode_duration * self.control_freq)
        
    def _computeReward(self) -> float:
        """
        Computes the reward for hovering near the target.
        Includes penalties for jitter, velocity, action diff, and tilt to ensure smooth flight.
        """
        current_pos = self.data.qpos[:3]
        current_vel = self.data.qvel[:3]
        current_ang_vel = self.data.qvel[3:6]
        
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        
        # 1. Base position reward
        pos_reward = float(3.0 * np.exp(-3.0 * dist))
        
        # 2. Penalty for jitter and high angular velocity
        ang_vel_penalty = 0.1 * np.linalg.norm(current_ang_vel)
        
        # 3. Penalty for high linear velocity
        lin_vel_penalty = 0.1 * np.linalg.norm(current_vel)
        
        # 4. Action smoothness penalty
        action_diff_penalty = 0.0
        if len(self.action_buffer) >= 2:
            action_diff = self.action_buffer[-1] - self.action_buffer[-2]
            action_diff_penalty = 0.2 * np.linalg.norm(action_diff)
            
        # 5. Posture penalty
        tilt_penalty = 0.5 * (1.0 - up_z)
            
        reward = pos_reward - ang_vel_penalty - lin_vel_penalty - action_diff_penalty - tilt_penalty
        
        # Heavy penalty for staying on the ground
        if current_pos[2] < 0.2:
            reward -= 5.0
            
        return float(reward)
        
    def _computeTerminated(self) -> bool:
        """
        We do not terminate on success to encourage continuous hovering.
        """
        return False
        
    def _computeTruncated(self) -> bool:
        """
        Computes whether the episode is truncated.
        """
        current_pos = self.data.qpos[:3]
        x, y, z = current_pos
        
        # Check boundaries (relax z lower bound to allow starting on ground)
        out_of_bounds = abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0 or z < -0.1
        
        # Check tilt (terminate if upside down)
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        too_tilted = up_z < 0.0 
        
        # Check max steps
        max_steps_reached = self.step_counter >= self.max_steps
        
        return bool(out_of_bounds or too_tilted or max_steps_reached)
        
    def _computeInfo(self) -> dict:
        """
        Computes the info dictionary.
        """
        current_pos = self.data.qpos[:3].copy()
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        
        return {
            "current_pos": current_pos,
            "target_pos": self.TARGET_POS.copy(),
            "pos_error": float(dist),
            "angular_velocity": self.data.qvel[3:6].copy()
        }
