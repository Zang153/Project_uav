"""
PID Controller implementations for UAV control.
Includes Position (P), Velocity (PID), Attitude (P-Quaternion), and Angular Rate (PID) controllers.
"""

import numpy as np
import quaternion # numpy-quaternion

class BasePID:
    """Base class for PID controllers."""
    def __init__(self, output_limits=(-float('inf'), float('inf'))):
        self.output_limits = output_limits

    def clip_output(self, output):
        """Clips the output to defined limits."""
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            return np.clip(output, self.output_limits[0], self.output_limits[1])
        return output
    
    def reset(self):
        """Resets the internal state of the controller."""
        pass


class PosPID(BasePID):
    """
    Position Controller (P-only).
    Outputs desired velocity based on position error.
    """
    def __init__(self, kp, dt, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.dt = dt

    def update(self, setpoint, current_value):
        """
        Updates the controller.
        
        Args:
            setpoint: Target position [x, y, z].
            current_value: Current position [x, y, z].
            
        Returns:
            output: Target velocity [vx, vy, vz].
        """
        error = np.array(setpoint) - np.array(current_value)
        output = self.kp * error
        return self.clip_output(output)


class VelPID(BasePID):
    """
    Velocity Controller (PID).
    Outputs desired acceleration (or thrust vector components) based on velocity error.
    """
    def __init__(self, kp, ki, kd, dt, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.ki = np.array(ki) if hasattr(ki, '__len__') else np.ones(3) * ki
        self.kd = np.array(kd) if hasattr(kd, '__len__') else np.ones(3) * kd
        self.dt = dt
        
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.last_value = np.zeros(3)

    def update(self, setpoint, current_value):
        """
        Updates the controller.
        
        Args:
            setpoint: Target velocity.
            current_value: Current velocity.
            
        Returns:
            output: Desired acceleration.
        """
        error = np.array(setpoint) - np.array(current_value)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative term (on measurement to avoid derivative kick)
        # v_dot = (v_current - v_last) / dt
        # D term = Kd * (0 - v_dot)  <-- assuming setpoint derivative is 0 or ignored
        v_dot = (current_value - self.last_value) / self.dt
        self.last_value = current_value
        d_term = self.kd * (0 - v_dot)
        
        output = p_term + i_term + d_term
        
        self.previous_error = error
        return self.clip_output(output)
    
    def reset(self):
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.last_value = np.zeros(3)


class AttitudePID3D(BasePID):
    """
    Attitude Controller (P-only on Quaternion Error).
    Outputs desired angular rates based on attitude error.
    """
    def __init__(self, kp, dt, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.dt = dt

    def update(self, setpoint, current_value):
        """
        Updates the controller using quaternion error.
        
        Args:
            setpoint: Target attitude (Quaternion [w, x, y, z] or object).
            current_value: Current attitude (Quaternion [w, x, y, z] or object).
            
        Returns:
            output: Desired angular rates [p, q, r].
        """
        # Ensure inputs are quaternion objects
        if isinstance(setpoint, np.quaternion):
            tar_quat = setpoint
        else:
            tar_quat = np.quaternion(*setpoint) # Assumes [w, x, y, z]

        if isinstance(current_value, np.quaternion):
            cur_quat = current_value
        else:
            cur_quat = np.quaternion(*current_value)

        # Compute error quaternion: q_err = q_target * q_current_inverse
        # Note: Original code used: quat_error = tar_quat * cur_conj
        cur_conj = cur_quat.conjugate()
        quat_error = tar_quat * cur_conj
        
        # Extract vector part of error quaternion (x, y, z)
        # This approximates the rotation vector for small angles
        error = np.array([quat_error.x, quat_error.y, quat_error.z])
        
        # Ensure shortest path (if w < 0, invert vector part)
        if quat_error.w < 0:
            error = -error
            
        output = self.kp * error
        return self.clip_output(output)


class AngVelPID(BasePID):
    """
    Angular Velocity Controller (PID).
    Outputs control torques based on angular rate error.
    """
    def __init__(self, kp, ki, kd, dt, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.ki = np.array(ki) if hasattr(ki, '__len__') else np.ones(3) * ki
        self.kd = np.array(kd) if hasattr(kd, '__len__') else np.ones(3) * kd
        self.dt = dt
        
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.last_value = np.zeros(3)

    def update(self, setpoint, current_value):
        """
        Updates the controller.
        
        Args:
            setpoint: Target angular rates.
            current_value: Current angular rates.
            
        Returns:
            output: Control torques [Mx, My, Mz].
        """
        error = np.array(setpoint) - np.array(current_value)
        
        p_term = self.kp * error
        
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative on measurement
        v_dot = (current_value - self.last_value) / self.dt
        self.last_value = current_value
        d_term = self.kd * (0 - v_dot)
        
        output = p_term + i_term + d_term
        
        self.previous_error = error
        return self.clip_output(output)
        
    def reset(self):
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.last_value = np.zeros(3)
