import os
import gymnasium as gym
import numpy as np
from abc import ABC
from collections import deque

from .BaseMujocoAviary import BaseMujocoAviary
from ..config import MASS, GRAVITY, CT, RL_CONTROL_FREQ

class BaseRLMujocoAviary(BaseMujocoAviary, ABC):
    """
    Base class for Reinforcement Learning in MuJoCo UAV environments.
    Adds action spaces, observation spaces, and an action history buffer.
    """
    def __init__(self, act_hist_len=2, hover_rpm=None, **kwargs):
        self.act_hist_len = act_hist_len
        self.control_freq = RL_CONTROL_FREQ
        
        self.action_buffer = deque(maxlen=self.act_hist_len)
        
        # Superclass init will call _actionSpace and _observationSpace
        super().__init__(**kwargs)
        
        if hover_rpm is None:
            # Automatically calculate the hover RPM based on the actual mass of the MuJoCo model
            # This ensures that if a heavier model (like Delta.xml) is loaded, the hover_rpm adapts automatically.
            import mujoco
            total_mass = mujoco.mj_getTotalmass(self.model)
            # Thrust required per motor (N)
            hover_thrust_per_motor = (total_mass * GRAVITY) / 4.0
            # Thrust = CT * krpm^2 -> krpm^2 = Thrust / CT
            hover_krpm_sq = hover_thrust_per_motor / CT
            self.hover_rpm = np.sqrt(hover_krpm_sq) * 1000.0
            print(f"[INFO] BaseRLMujocoAviary: Dynamically calculated total mass: {total_mass:.3f} kg, hover_rpm: {self.hover_rpm:.1f}")
        else:
            self.hover_rpm = hover_rpm
        
        # Calculate how many physics steps to run per RL action step
        self.physics_steps_per_control = int(self.freq / self.control_freq)

    def reset(self, seed=None, options=None):
        """Resets the environment and the action buffer."""
        self.action_buffer.clear()
        # Initialize buffer with zeros
        for _ in range(self.act_hist_len):
            self.action_buffer.append(np.zeros(self.model.nu, dtype=np.float32))
            
        self.step_counter = 0
            
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        """
        Executes an action in the environment.
        Advances the physics simulation by `physics_steps_per_control` steps.
        """
        processed_action = self._preprocessAction(action)
        
        # Apply the same action for multiple physics steps
        for _ in range(self.physics_steps_per_control):
            # Apply control
            self.data.ctrl[:] = processed_action
            
            # Step MuJoCo physics
            import mujoco
            mujoco.mj_step(self.model, self.data)
            
            # Check early termination during the skipped steps
            if self._computeTerminated():
                break
                
        # Update internal step counter ONCE per control step
        self.step_counter += 1

        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        return obs, reward, terminated, truncated, info

    def _actionSpace(self) -> gym.Space:
        """
        Defines the RL action space: [-1, 1] for each actuator.
        """
        return gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )

    def _observationSpace(self) -> gym.Space:
        """
        Defines the flattened kinematic observation space + action buffer.
        State consists of:
        - Position (3)
        - Quaternion (4)
        - Linear Velocity (3)
        - Angular Velocity (3)
        - Action buffer (act_hist_len * nu)
        """
        # qpos is 7 (pos + quat), qvel is 6 (vel + ang_vel)
        state_dim = 13
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
        Assumes the actuators expect squared krpm values.
        """
        # Clip action for safety
        action = np.clip(action, -1.0, 1.0)
        
        # Append to action history buffer
        self.action_buffer.append(action.copy())
        
        # Simple scaling: hover_rpm * (1 + 0.05 * action)
        target_rpm = self.hover_rpm * (1.0 + 0.05 * action)
        
        # Convert to krpm^2 for MuJoCo actuators (since gear = CT)
        target_krpm = target_rpm / 1000.0
        processed_action = target_krpm ** 2
        
        return processed_action.astype(np.float32)

    def _computeObs(self) -> np.ndarray:
        """
        Computes and returns the current observation.
        """
        # Kinematic state from MuJoCo
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()
        vel = self.data.qvel[:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        
        # Action buffer
        if len(self.action_buffer) < self.act_hist_len:
            pad_len = self.act_hist_len - len(self.action_buffer)
            for _ in range(pad_len):
                self.action_buffer.append(np.zeros(self.model.nu, dtype=np.float32))
                
        act_buf = np.concatenate(self.action_buffer)
        
        # Concatenate into flattened observation
        obs = np.concatenate([pos, quat, vel, ang_vel, act_buf])
        return obs.astype(np.float32)
