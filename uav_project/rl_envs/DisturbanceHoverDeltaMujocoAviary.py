import numpy as np
import gymnasium as gym
import math
import torch
from .BaseRLMujocoAviary import BaseRLMujocoAviary
from uav_project.utils.DeltaRandomTrajectoryGenerator import DeltaRandomTrajectoryGenerator

class DisturbanceHoverDeltaMujocoAviary(BaseRLMujocoAviary):
    """
    Hover task environment for Delta MuJoCo UAV with internal arm disturbances.
    The goal is to hover at a target position [0, 0, 1.5] using only 4 rotors,
    while the 3 arm joints are driven internally to simulate disturbances.
    """
    def __init__(self, model_filename="Delta.xml", episode_duration=10.0, **kwargs):
        self.TARGET_POS = np.array([0, 0, 1.5], dtype=np.float32)
        
        xml_path = f"../meshes/{model_filename}"
        super().__init__(xml_path=xml_path, **kwargs)
        
        self.episode_duration = episode_duration
        self.max_steps = int(self.episode_duration * self.control_freq)

        # Define internal arm movement state in Cartesian space
        self.trajectory_generator = DeltaRandomTrajectoryGenerator(rod_b=0.1, rod_ee=0.2, r_b=0.074577, r_ee=0.02495)
        self.external_arm_action = None

    def set_external_arm_action(self, arm_action: np.ndarray):
        """Allows an external controller to dictate the arm movement during testing."""
        self.external_arm_action = arm_action

    def reset(self, seed=None, options=None):
        # Randomize arm disturbance parameters on each reset
        self.trajectory_generator.reset()
        
        # Super reset handles the Mujoco data and action buffer clearing
        obs, info = super().reset(seed=seed, options=options)
        
        return obs, info
        
    def _actionSpace(self) -> gym.Space:
        """
        Defines the RL action space: [-1, 1] for 4 rotors ONLY.
        Total 4 dimensions.
        """
        return gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )

    def _observationSpace(self) -> gym.Space:
        """
        Defines the flattened kinematic observation space + action buffer.
        """
        state_dim = self.model.nq + self.model.nv
        buffer_dim = self.act_hist_len * 4  # ONLY 4 rotor actions
        obs_dim = state_dim + buffer_dim
        
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """
        Maps the [-1, 1] 4D RL action to actual MuJoCo control inputs (7D).
        - 4 rotors: scaled hover_rpm to krpm^2
        - 3 arm positions: internally generated disturbance
        """
        # Clip action for safety
        action = np.clip(action, -1.0, 1.0)
        
        # We need to make sure we append only 4D action, not 7D, so diff calculation works
        # If action buffer is full of 7D from a bad init or reset, clear it and rebuild
        if len(self.action_buffer) > 0 and len(self.action_buffer[0]) != 4:
            self.action_buffer.clear()
            for _ in range(self.act_hist_len - 1):
                self.action_buffer.append(np.zeros(4, dtype=np.float32))
                
        # Append to action history buffer (4D)
        self.action_buffer.append(action.copy())
        
        # 1. Rotors (indices 0 to 3)
        target_rpm = self.hover_rpm * (1.0 + 0.5 * action)
        target_krpm = target_rpm / 1000.0
        rotor_action = target_krpm ** 2
        
        # 2. Arm positions (indices 4 to 6) generated internally or externally
        if self.external_arm_action is not None:
            arm_action = self.external_arm_action
        else:
            # Calculate current sim time based on the base class's step_counter
            current_sim_time = self.step_counter / self.control_freq
            
            # Use the standalone trajectory generator
            _, arm_action = self.trajectory_generator.get_state(current_sim_time)
            
        processed_action = np.concatenate([rotor_action, arm_action])
        return processed_action.astype(np.float32)

    def _computeObs(self) -> np.ndarray:
        """
        Computes and returns the current observation.
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        if len(self.action_buffer) < self.act_hist_len:
            pad_len = self.act_hist_len - len(self.action_buffer)
            for _ in range(pad_len):
                self.action_buffer.append(np.zeros(4, dtype=np.float32))
                
        act_buf = np.concatenate(self.action_buffer)
        
        # Flatten qpos (length 14 for Delta model: 7 for UAV + 7 for arm)
        # Flatten qvel (length 13 for Delta model: 6 for UAV + 7 for arm)
        # Ensure all arrays are flat and 1D
        qpos_flat = np.array(qpos).flatten()
        qvel_flat = np.array(qvel).flatten()
        act_buf_flat = np.array(act_buf).flatten()
        
        obs = np.concatenate([qpos_flat, qvel_flat, act_buf_flat])
        
        # Clip or pad to exactly match obs_dim = 88 to avoid PPO shape mismatch
        # Stable-Baselines3 checks the exact shape.
        obs_dim = 88
        if obs.shape[0] > obs_dim:
            obs = obs[:obs_dim]
        elif obs.shape[0] < obs_dim:
            obs = np.pad(obs, (0, obs_dim - obs.shape[0]), 'constant')
            
        return obs.astype(np.float32)

    def _computeReward(self) -> float:
        """
        Computes the reward for hovering near the target.
        """
        current_pos = self.data.qpos[:3]
        current_vel = self.data.qvel[:3]
        current_ang_vel = self.data.qvel[3:6]
        
        # Up vector check to penalize tilt (MuJoCo quat is [w, x, y, z])
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        
        # 1. Base position reward (sharper peak around zero error to encourage pinpoint hover)
        # Using exponential decay instead of polynomial for a smoother, tighter gradient
        pos_reward = float(3.0 * np.exp(-3.0 * dist))
        
        # 2. Penalty for jitter and high angular velocity
        ang_vel_penalty = 0.1 * np.linalg.norm(current_ang_vel)
        
        # 3. Penalty for high linear velocity (we want it to hover statically, not fly around wildly)
        lin_vel_penalty = 0.1 * np.linalg.norm(current_vel)
        
        # 4. Action smoothness penalty
        action_diff_penalty = 0.0
        if len(self.action_buffer) >= 2:
            action_diff = self.action_buffer[-1] - self.action_buffer[-2]
            action_diff_penalty = 0.2 * np.linalg.norm(action_diff)
            
        # 5. Posture penalty (encourage the drone to stay flat)
        # If up_z is 1 (perfectly flat), penalty is 0. 
        tilt_penalty = 0.5 * (1.0 - up_z)
            
        reward = pos_reward - ang_vel_penalty - lin_vel_penalty - action_diff_penalty - tilt_penalty
        
        # Heavy penalty for staying on the ground to break out of local optima
        if current_pos[2] < 0.2:
            reward -= 5.0
            
        return float(reward)
        
    def _computeTerminated(self) -> bool:
        return False
        
    def _computeTruncated(self) -> bool:
        current_pos = self.data.qpos[:3]
        x, y, z = current_pos
        
        out_of_bounds = abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0 or z < -0.1
        
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        too_tilted = up_z < 0.0 
        
        max_steps_reached = self.step_counter >= self.max_steps
        
        return bool(out_of_bounds or too_tilted or max_steps_reached)
        
    def _computeInfo(self) -> dict:
        current_pos = self.data.qpos[:3]
        dist = np.linalg.norm(self.TARGET_POS - current_pos)
        
        return {
            "current_pos": current_pos.copy(),
            "target_pos": self.TARGET_POS.copy(),
            "pos_error": float(dist),
            "angular_velocity": self.data.qvel[3:6].copy()
        }
