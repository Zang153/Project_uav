import numpy as np
import torch
import math
from typing import Tuple
from uav_project.utils.DeltaKinematics import DeltaKinematics

class DeltaRandomTrajectoryGenerator:
    """
    Generates physically valid, continuous random trajectories for the Delta manipulator.
    Pre-validates the workspace to ensure Inverse Kinematics (IK) always succeeds.
    """
    def __init__(self, rod_b=0.1, rod_ee=0.2, r_b=0.074577, r_ee=0.02495):
        self.kinematics = DeltaKinematics(rod_b=rod_b, rod_ee=rod_ee, r_b=r_b, r_ee=r_ee)
        
        # Internal state for the current trajectory
        self.z_center = 0.0
        self.xy_radius = 0.0
        self.z_amp = 0.0
        self.freq_x = 0.0
        self.freq_y = 0.0
        self.freq_z = 0.0
        self.phase_x = 0.0
        self.phase_y = 0.0
        self.phase_z = 0.0
        
        # Pre-roll the first valid trajectory
        self.reset()

    def _generate_params(self, fixed_period=None):
        """Randomize the parameters for a 3D Lissajous-like curve."""
        # Z is usually negative (below the base) for Delta robots
        self.z_center = np.random.uniform(-0.20, -0.15)
        # xy_radius determines the horizontal reach
        self.xy_radius = np.random.uniform(0.02, 0.12)
        # z_amp determines the vertical oscillation
        self.z_amp = np.random.uniform(0.0, 0.04)
        
        if fixed_period is not None:
            # If a fixed period is requested, ensure frequencies are multiples of the base frequency
            base_freq = 1.0 / fixed_period
            # Pick integer multipliers to ensure the pattern repeats exactly every fixed_period
            self.freq_x = base_freq * np.random.randint(1, 4)  # e.g., 1x, 2x, 3x
            self.freq_y = base_freq * np.random.randint(1, 4)
            self.freq_z = base_freq * np.random.randint(1, 3)
        else:
            # Frequencies for X, Y, Z (Hz)
            self.freq_x = np.random.uniform(0.2, 1.0)
            self.freq_y = np.random.uniform(0.2, 1.0)
            self.freq_z = np.random.uniform(0.1, 0.5)
        
        # Phase offsets
        self.phase_x = np.random.uniform(0, 2 * math.pi)
        self.phase_y = np.random.uniform(0, 2 * math.pi)
        self.phase_z = np.random.uniform(0, 2 * math.pi)

    def _validate_workspace(self, fixed_period=None) -> bool:
        """
        Discretely samples the generated trajectory over its period to ensure
        all points are within the Delta robot's physical workspace.
        """
        # Determine a reasonable period to check. 
        if fixed_period is not None:
            max_period = fixed_period
        else:
            # The lowest frequency is freq_z (min 0.1Hz), so max period is 10s.
            max_period = 10.0 
            
        num_samples = 100
        times = np.linspace(0, max_period, num_samples)
        
        for t in times:
            pos = self._compute_cartesian_pos(t)
            # Check IK
            joint_angles_deg = self.kinematics.ik(torch.tensor(pos, dtype=torch.float32))
            # DeltaKinematics.ik returns -1 if unreachable
            if isinstance(joint_angles_deg, int) and joint_angles_deg == -1:
                return False
        return True

    def reset(self, fixed_period=None):
        """
        Generates new random parameters and loops until a fully valid 
        trajectory within the workspace is found.
        """
        max_attempts = 100
        for attempt in range(max_attempts):
            self._generate_params(fixed_period)
            if self._validate_workspace(fixed_period):
                return
        
        # Fallback to a very safe, small trajectory if we are extremely unlucky
        print("[WARNING] DeltaRandomTrajectoryGenerator: Failed to find valid trajectory after 100 attempts. Using safe fallback.")
        self.z_center = -0.18
        self.xy_radius = 0.02
        self.z_amp = 0.01
        self.freq_x = 0.5
        self.freq_y = 0.5
        self.freq_z = 0.25
        self.phase_x = 0.0
        self.phase_y = math.pi / 2
        self.phase_z = 0.0

    def _compute_cartesian_pos(self, t: float) -> np.ndarray:
        """Computes the exact Cartesian [x, y, z] position at time t."""
        x = self.xy_radius * math.cos(2 * math.pi * self.freq_x * t + self.phase_x)
        y = self.xy_radius * math.sin(2 * math.pi * self.freq_y * t + self.phase_y)
        z = self.z_center + self.z_amp * math.sin(2 * math.pi * self.freq_z * t + self.phase_z)
        return np.array([x, y, z], dtype=np.float32)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the current state of the Delta arm at time t.
        
        Args:
            t: Continuous time in seconds.
            
        Returns:
            cartesian_pos: [x, y, z] numpy array
            joint_angles_rad: [theta1, theta2, theta3] numpy array in radians
        """
        pos = self._compute_cartesian_pos(t)
        
        # We already pre-validated the workspace, so IK should mathematically always succeed.
        # But just in case of float precision edge cases, we handle it safely.
        joint_angles_deg = self.kinematics.ik(torch.tensor(pos, dtype=torch.float32))
        
        if isinstance(joint_angles_deg, int) and joint_angles_deg == -1:
            # Fallback to zero angles (straight down) if a precision error occurs
            joint_angles_rad = np.zeros(3, dtype=np.float32)
        else:
            joint_angles_rad = np.deg2rad(joint_angles_deg.numpy()).astype(np.float32)
            
        return pos, joint_angles_rad
