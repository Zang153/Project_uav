import numpy as np
import gymnasium as gym
from .BaseRLMujocoAviary import BaseRLMujocoAviary

class HoverDeltaMujocoAviary(BaseRLMujocoAviary):
    """
    Hover task environment for Delta MuJoCo UAV.
    The goal is to hover at a target position [0, 0, 1.5].
    """
    def __init__(self, model_filename="Delta.xml", episode_duration=10.0, **kwargs):
        self.TARGET_POS = np.array([0, 0, 1.5], dtype=np.float32)
        
        # BaseMujocoAviary takes xml_path, not model_filename
        # We rely on the timestep defined in Delta.xml (0.0001s)
        xml_path = f"../meshes/{model_filename}"
        super().__init__(xml_path=xml_path, **kwargs)
        
        # max_steps is calculated dynamically based on control_freq and the provided episode_duration
        self.episode_duration = episode_duration
        self.max_steps = int(self.episode_duration * self.control_freq)
        self.step_counter = 0
        
    def reset(self, seed=None, options=None):
        # BaseRLMujocoAviary handles the step_counter reset now
        return super().reset(seed=seed, options=options)
        
    def step(self, action):
        # BaseRLMujocoAviary handles the step_counter increment now
        return super().step(action)
        
    def _actionSpace(self) -> gym.Space:
        """
        Defines the RL action space: [-1, 1] for 4 rotors and 3 arm positions.
        Total 7 dimensions.
        """
        return gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )

    def _observationSpace(self) -> gym.Space:
        """
        Defines the flattened kinematic observation space + action buffer.
        """
        state_dim = self.model.nq + self.model.nv
        buffer_dim = self.act_hist_len * self.model.nu
        obs_dim = state_dim + buffer_dim
        
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """
        Maps the [-1, 1] RL action to actual MuJoCo control inputs.
        - 4 rotors: scaled hover_rpm to krpm^2
        - 3 arm positions: map [-1, 1] to [-1.57, 0.523]
        """
        # Clip action for safety
        action = np.clip(action, -1.0, 1.0)
        
        # Append to action history buffer
        self.action_buffer.append(action.copy())
        
        # 1. Rotors (indices 0 to 3)
        # Increased authority from 0.05 (5%) to 0.3 (30%) to allow the drone to generate enough thrust to take off
        target_rpm = self.hover_rpm * (1.0 + 0.5 * action[:4])
        target_krpm = target_rpm / 1000.0
        rotor_action = target_krpm ** 2
        
        # 2. Arm positions (indices 4 to 6)
        # Map [-1, 1] to [-1.57, 0.523]
        min_pos = -1.57
        max_pos = 0.523
        arm_action = (action[4:7] + 1.0) / 2.0 * (max_pos - min_pos) + min_pos
        
        processed_action = np.concatenate([rotor_action, arm_action])
        return processed_action.astype(np.float32)

    def _computeObs(self) -> np.ndarray:
        """
        Computes and returns the current observation.
        """
        # Kinematic state from MuJoCo (full qpos and qvel)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Action buffer
        if len(self.action_buffer) < self.act_hist_len:
            pad_len = self.act_hist_len - len(self.action_buffer)
            for _ in range(pad_len):
                self.action_buffer.append(np.zeros(self.model.nu, dtype=np.float32))
                
        act_buf = np.concatenate(self.action_buffer)
        
        # Concatenate into flattened observation
        obs = np.concatenate([qpos, qvel, act_buf])
        return obs.astype(np.float32)

    def _computeReward(self) -> float:
        """
        Computes the reward for hovering near the target.
        """
        current_pos = self.data.qpos[:3]
        current_vel = self.data.qvel[:3]
        current_ang_vel = self.data.qvel[3:6]
        
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        
        # 1. Base position reward
        # Using dist**2 instead of dist**4 to make the gradient smoother at further distances
        pos_reward = float(np.max([0.0, 2.0 - dist**2]))
        
        # 2. Penalty for jitter and high angular velocity
        # This forces the drone to find a stable, smooth hovering posture
        ang_vel_penalty = 0.05 * np.linalg.norm(current_ang_vel)
        
        # 3. Penalty for high linear velocity (we want it to hover statically, not fly around wildly)
        lin_vel_penalty = 0.05 * np.linalg.norm(current_vel)
        
        # 4. Action smoothness penalty
        # Penalize large changes in action between consecutive steps
        action_diff_penalty = 0.0
        if len(self.action_buffer) >= 2:
            action_diff = self.action_buffer[-1] - self.action_buffer[-2]
            action_diff_penalty = 0.1 * np.linalg.norm(action_diff)
            
        reward = pos_reward - ang_vel_penalty - lin_vel_penalty - action_diff_penalty
        
        # Heavy penalty for staying on the ground to break out of local optima
        # The target height is 1.0. If the drone's z is below 0.2, it is heavily penalized.
        if current_pos[2] < 0.2:
            reward -= 2.0
            
        return float(reward)
        
    def _computeTerminated(self) -> bool:
        """
        Computes whether the episode is terminated.
        """
        current_pos = self.data.qpos[:3]
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        # We don't terminate on success (dist < 0.01) to encourage continuous hovering
        return False
        
    def _computeTruncated(self) -> bool:
        """
        Computes whether the episode is truncated.
        """
        current_pos = self.data.qpos[:3]
        x, y, z = current_pos
        
        # Check boundaries (z > 2.0 or z < 0.0)
        # Note: the drone might be initialized at z=0 (ground) before taking off,
        # so checking z < 0.0 might immediately trigger truncation if it drops slightly below 0.
        # Let's relax the lower bound to z < -0.1 to allow sitting on the ground initially.
        out_of_bounds = abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0 or z < -0.1
        
        # Check tilt (simplified by checking the z-component of the up vector)
        # MuJoCo quat is [w, x, y, z]
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat
        # The z-component of the up vector rotated by quat is 1 - 2(x^2 + y^2)
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        too_tilted = up_z < 0.0 # Tilted more than 90 degrees (upside down)
        
        # Check max steps
        max_steps_reached = self.step_counter >= self.max_steps
        
        return bool(out_of_bounds or too_tilted or max_steps_reached)
        
    def _computeInfo(self) -> dict:
        """
        Computes the info dictionary.
        """
        current_pos = self.data.qpos[:3]
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        
        return {
            "current_pos": current_pos.copy(),
            "target_pos": self.TARGET_POS.copy(),
            "pos_error": float(dist),
            "angular_velocity": self.data.qvel[3:6].copy()
        }
