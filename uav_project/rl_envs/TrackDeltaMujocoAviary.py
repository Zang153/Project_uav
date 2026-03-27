import numpy as np
import gymnasium as gym
from .HoverDeltaMujocoAviary import HoverDeltaMujocoAviary
from ..models.uav_model import UAVModel

class TrackDeltaMujocoAviary(HoverDeltaMujocoAviary):
    """
    Tracking task environment for Delta MuJoCo UAV.
    The goal is to hover at a target position while the Delta arm tracks a trajectory.
    """
    def __init__(self, 
                 model_filename="Delta.xml", 
                 episode_duration=10.0,
                 traj_radius=0.12,
                 traj_z=-0.18,
                 traj_period=8.0,
                 **kwargs):
        
        self.traj_radius = traj_radius
        self.traj_z = traj_z
        self.traj_period = traj_period
        
        # Pass episode_duration up to the parent class
        super().__init__(model_filename=model_filename, **kwargs)
        
        # Max steps per episode to prevent infinite loops
        self.episode_duration = episode_duration
        self.max_steps = int(self.episode_duration * self.control_freq)
        
        # Instantiate UAVModel to easily access sensors like the end-effector position
        self.uav = UAVModel(self.model, self.data)
        
        # Override observation space after UAVModel is initialized
        self.observation_space = self._observationSpace()
        
    def _getTargetEEPos(self) -> np.ndarray:
        """
        Computes the target end-effector position in the UAV body frame.
        """
        t = self.data.time
        omega = 2 * np.pi / self.traj_period
        phase = omega * t
        
        # This is the target position relative to the UAV's current position
        rel_x = self.traj_radius * np.cos(phase)
        rel_y = self.traj_radius * np.sin(phase)
        rel_z = self.traj_z
        
        return np.array([rel_x, rel_y, rel_z], dtype=np.float32)

    def _getTargetEEVel(self) -> np.ndarray:
        """
        Computes the target end-effector velocity in the UAV body frame.
        """
        t = self.data.time
        omega = 2 * np.pi / self.traj_period
        phase = omega * t
        
        vx = -self.traj_radius * omega * np.sin(phase)
        vy = self.traj_radius * omega * np.cos(phase)
        vz = 0.0
        
        return np.array([vx, vy, vz], dtype=np.float32)

    def _observationSpace(self) -> gym.Space:
        """
        Overrides the base observation space to include the target EE position,
        relative EE error, and target EE velocity.
        Original obs + target_ee_pos (3) + relative_ee_pos (3) + target_ee_vel (3)
        """
        state_dim = self.model.nq + self.model.nv
        buffer_dim = self.act_hist_len * self.model.nu
        obs_dim = state_dim + buffer_dim + 9  # +3 target_pos, +3 relative_pos, +3 target_vel
        
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def _computeObs(self) -> np.ndarray:
        """
        Computes the current observation, appending target EE position, relative EE position, and target EE velocity.
        """
        base_obs = super()._computeObs()
        target_ee_pos = self._getTargetEEPos()
        current_ee_pos = self.uav.get_ee_sensor_pos()
        target_ee_vel = self._getTargetEEVel()
        
        relative_ee_pos = target_ee_pos - current_ee_pos
        
        obs = np.concatenate([base_obs, target_ee_pos, relative_ee_pos, target_ee_vel])
        return obs.astype(np.float32)

    def _computeReward(self) -> float:
        """
        Computes the reward combining UAV hover reward and EE tracking reward.
        """
        # UAV Hover Reward (from super class)
        uav_reward = super()._computeReward()
        
        # EE Tracking Reward
        target_ee_pos = self._getTargetEEPos()
        current_ee_pos = self.uav.get_ee_sensor_pos()
        ee_dist = np.linalg.norm(target_ee_pos - current_ee_pos)
        
        # Scale tracking error. We use (ee_dist / 0.085)**2 to penalize small errors more effectively.
        # e.g., an error > 0.12m will result in 0 reward.
        # We increase the max reward from 2.0 to 10.0 to make it a dominant signal compared to hovering.
        # We also make the penalty sharper so it forces the arm to move.
        ee_reward = float(np.max([-5.0, 10.0 - 10.0 * (ee_dist / 0.05)**2]))
        
        # Tilt penalty
        quat = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        tilt_penalty = 5.0 * (1.0 - up_z)
        
        return uav_reward + ee_reward - tilt_penalty
        
    def _computeInfo(self) -> dict:
        """
        Computes the info dictionary.
        """
        info = super()._computeInfo()
        
        # Use the world-frame target position we already defined
        tracking_target = self._getTargetEEPos()
        
        # Calculate arm tracking error
        ee_pos = self.uav.get_ee_sensor_pos()
        track_error = np.linalg.norm(tracking_target - ee_pos)
        info["ee_pos"] = ee_pos
        info["track_error"] = float(track_error)
            
        info["tracking_target"] = tracking_target
        
        return info

    def _computeTruncated(self) -> bool:
        """
        Truncate if out of bounds, too tilted, or max steps reached.
        """
        current_pos = self.data.qpos[:3]
        x, y, z = current_pos
        
        # Relax z lower bound to -0.1 to allow takeoff from ground without instant truncation
        out_of_bounds = abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0 or z < -0.1
        
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        too_tilted = up_z < 0.0 # Tilted more than 90 degrees
        
        max_steps_reached = self.step_counter >= self.max_steps
        
        return bool(out_of_bounds or too_tilted or max_steps_reached)
