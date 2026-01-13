"""
Cascade PID Controller implementation for Quadrotor.
Structure: Position -> Velocity -> Attitude -> Angular Rate -> Mixer.
"""

import numpy as np
import quaternion
from ..config import (
    MASS, GRAVITY,
    FREQ_POSITION, FREQ_VELOCITY, FREQ_ATTITUDE, FREQ_ANGLE_RATE,
    POS_KP, VEL_KP, VEL_KI, VEL_KD,
    ATT_KP, RATE_KP, RATE_KI, RATE_KD
)
from .pid import PosPID, VelPID, AttitudePID3D, AngVelPID
from ..models.mixer import Mixer

class CascadeController:
    """
    Manages the cascade of PID controllers to control the UAV.
    """
    def __init__(self, uav_model):
        """
        Args:
            uav_model: Instance of UAVModel to interact with the drone.
        """
        self.uav = uav_model
        self.mixer = Mixer()
        
        # Simulation time tracking
        self.sim_time = 0.0
        
        # Update frequencies and periods
        self.freqs = {
            'pos': FREQ_POSITION,
            'vel': FREQ_VELOCITY,
            'att': FREQ_ATTITUDE,
            'rate': FREQ_ANGLE_RATE
        }
        self.dts = {k: 1.0/v for k, v in self.freqs.items()}
        
        # Last update times
        self.last_updates = {
            'pos': 0.0,
            'vel': 0.0,
            'att': 0.0,
            'rate': 0.0
        }
        
        # Initialize Controllers
        self.pos_pid = PosPID(POS_KP, self.dts['pos'])
        self.vel_pid = VelPID(VEL_KP, VEL_KI, VEL_KD, self.dts['vel'])
        self.att_pid = AttitudePID3D(ATT_KP, self.dts['att'])
        self.rate_pid = AngVelPID(RATE_KP, RATE_KI, RATE_KD, self.dts['rate'])
        
        # State Targets
        self.target_position = np.array([0.0, 0.0, 1.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_attitude = np.quaternion(1, 0, 0, 0) # [w, x, y, z]
        self.target_rate = np.array([0.0, 0.0, 0.0])
        self.target_yaw = 0.0 # Radians
        
        # Outputs
        self.total_thrust_vector = np.zeros(3) # World frame force
        self.torque_command = np.zeros(3)      # Body frame torque
        self.motor_thrusts = np.zeros(4)
        
        # For logging/debugging
        self.mixer_output_log = np.zeros(6) # [Fx, Fy, Fz, Mx, My, Mz]

    def reset(self):
        """Resets all sub-controllers and state."""
        self.pos_pid.reset()
        self.vel_pid.reset()
        self.att_pid.reset()
        self.rate_pid.reset()
        self.sim_time = 0.0
        for k in self.last_updates:
            self.last_updates[k] = 0.0

    def set_target_position(self, pos):
        self.target_position = np.array(pos)

    def set_target_yaw(self, yaw_rad):
        self.target_yaw = yaw_rad
        # Update target attitude quaternion based on new yaw? 
        # Usually attitude is computed from velocity controller output.
        # But if we are in hover (vel=0), we should respect this yaw.
        # The velocity controller logic calculates desired attitude including yaw.

    def update(self, sim_time):
        """
        Main update loop. Should be called every simulation step.
        Checks if individual controllers need to run based on their frequencies.
        """
        self.sim_time = sim_time
        
        # Get current state
        UAV_position, UAV_velocity, UAV_att_quat, UAV_rate = self.uav.get_uav_state()
        # UAV_att_quat is [w, x, y, z]
        UAV_current_att = np.quaternion(*UAV_att_quat)

        # 1. Position Controller (50Hz)
        if sim_time >= self.last_updates['pos'] + self.dts['pos']:
            self.target_velocity = self.pos_pid.update(self.target_position, UAV_position)
            self.last_updates['pos'] = sim_time

        # 2. Velocity Controller (50Hz)
        if sim_time >= self.last_updates['vel'] + self.dts['vel']:
            # Desired acceleration
            acc_cmd = self.vel_pid.update(self.target_velocity, UAV_velocity)
            
            # Gravity Compensation (World Frame: +Z opposes gravity)
            # Acceleration needed to counter gravity is [0, 0, 9.81]
            # But the PID output is "acceleration error correction".
            # Total desired acceleration = PID_out + Gravity_comp
            # Note: Original code subtracted gravity [0, 0, -9.81] -> added [0, 0, 9.81]
            gravity_comp = np.array([0, 0, GRAVITY]) 
            
            # Total desired acceleration in World Frame
            total_acc = acc_cmd + gravity_comp
            
            # Calculate Total Thrust Vector (World Frame)
            # F = m * a
            self.total_thrust_vector = MASS * total_acc
            
            # Calculate Desired Attitude from Thrust Vector
            self.target_attitude = self._calculate_desired_attitude(self.total_thrust_vector, self.target_yaw)
            
            self.last_updates['vel'] = sim_time

        # 3. Attitude Controller (250Hz)
        if sim_time >= self.last_updates['att'] + self.dts['att']:
            self.target_rate = self.att_pid.update(self.target_attitude, UAV_current_att)
            self.last_updates['att'] = sim_time

        # 4. Angular Rate Controller (1000Hz)
        if sim_time >= self.last_updates['rate'] + self.dts['rate']:
            self.torque_command = self.rate_pid.update(self.target_rate, UAV_rate)
            
            # Apply Control
            self._apply_controls()
            
            self.last_updates['rate'] = sim_time

        

    def print_state(self):
        return self.uav.print_uav_state()

    def _calculate_desired_attitude(self, thrust_vector, yaw_target):
        """
        Calculates the desired attitude quaternion to align the body Z-axis 
        with the desired thrust vector, while maintaining the target yaw.
        """
        # Z-axis (Body Z) should align with thrust vector
        z_b = thrust_vector / (np.linalg.norm(thrust_vector) + 1e-6)
        
        # Desired Yaw direction in World Frame (Body Y projected on XY plane)
        # Note: Original code uses specific logic for X/Y axes construction.
        # Standard approach:
        # x_c = [cos(yaw), sin(yaw), 0]
        # y_b = (z_b x x_c) / norm...
        # x_b = y_b x z_b
        
        # Original code logic:
        # yaw_axis = [-sin(yaw), cos(yaw), 0] (This looks like Y axis direction for 0 yaw is Y-world?)
        # Let's follow standard aviation or original?
        # Original: yaw_angle=0 -> yaw_axis = [0, 1, 0] (World Y)
        # x_axis = cross(yaw_axis, z_axis) -> cross(Y, Z) = X. Correct.
        
        yaw_axis = np.array([-np.sin(yaw_target), np.cos(yaw_target), 0])
        
        x_b = np.cross(yaw_axis, z_b)
        x_b_norm = np.linalg.norm(x_b)
        
        if x_b_norm < 1e-6:
            # Singularity (Thrust vector parallel to yaw axis - unlikely unless 90deg pitch/roll)
            x_b = np.array([1, 0, 0])
        else:
            x_b = x_b / x_b_norm
            
        y_b = np.cross(z_b, x_b)
        y_b = y_b / np.linalg.norm(y_b)
        
        # Rotation Matrix [x_b, y_b, z_b]
        rot_mat = np.column_stack((x_b, y_b, z_b))
        
        return quaternion.from_rotation_matrix(rot_mat)

    def _apply_controls(self):
        """
        Calculates motor commands and applies them to the UAV.
        """
        # Total Thrust Magnitude (Projected onto Body Z)
        # Actually, total_thrust_vector is in World Frame.
        # The force actuators in MuJoCo (forcex,y,z) are likely Body Frame (if defined at site) or World Frame?
        # My analysis of uav_model.py concluded they are likely applied at a site which rotates.
        # BUT, the original code computed `force_body = att_mat.T @ total_thrust`.
        # This confirms actuators are BODY FRAME.
        # So we need to convert World Frame Thrust Vector to Body Frame.
        
        _, _, UAV_att_quat, _ = self.uav.get_uav_state()
        q = np.quaternion(*UAV_att_quat)
        rot_mat = quaternion.as_rotation_matrix(q)
        
        # Transform World Thrust -> Body Frame
        # F_body = R^T * F_world
        force_body = rot_mat.T @ self.total_thrust_vector
        
        # Prepare Mixer Inputs
        # The mixer expects Total Thrust (Scalar) and Torques.
        # For simplified control, we send [Fx, Fy, Fz] directly to force actuators.
        # And [Mx, My, Mz] directly to torque actuators.
        
        # Update logging
        self.mixer_output_log[:3] = force_body
        self.mixer_output_log[3:] = self.torque_command
        
        # Apply to model
        self.uav.set_actuators(force_body, self.torque_command)
        
        # Calculate individual motor thrusts for logging/visualization
        # Total thrust magnitude for mixer
        thrust_mag = force_body[2] # Primarily Z component
        self.motor_thrusts = self.mixer.simple_mix(thrust_mag, self.torque_command)

    def get_log_data(self):
        """Returns data for logging."""
        UAV_pos, UAV_vel, UAV_att, UAV_rate = self.uav.get_uav_state()
        return (
            self.sim_time, UAV_pos, UAV_vel, UAV_att, UAV_rate,
            self.target_position, self.target_velocity, self.target_attitude, self.target_rate,
            self.motor_thrusts, self.mixer_output_log
        )
