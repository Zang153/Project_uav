"""
Motor mixing logic for converting forces/torques to motor speeds.
"""
import numpy as np
from ..config import CT, CD, ARM_LENGTH, MAX_THRUST_PER_MOTOR, MAX_TORQUE_PER_MOTOR, MAX_MOTOR_SPEED_KRPM

class Mixer:
    """
    Handles mixing of control inputs (Thrust + Torques) to motor outputs (RPM/Force).
    """
    def __init__(self):
        self.Ct = CT
        self.Cd = CD
        self.L = ARM_LENGTH
        self.max_thrust = MAX_THRUST_PER_MOTOR
        self.max_torque = MAX_TORQUE_PER_MOTOR
        self.max_speed = MAX_MOTOR_SPEED_KRPM

        # Forward Mixing Matrix: [F_total, Mx, My, Mz]^T = Mat @ [w1^2, w2^2, w3^2, w4^2]^T
        # Motor order (Quad X config):
        # 1: Front-Left (CW or CCW? Standard is: 1(FR-CCW), 2(RL-CCW), 3(FL-CW), 4(RR-CW) - Wait, let's check original code)
        # Original Code Note:
        # prop1/prop3: anti_clockwise (CCW)
        # prop2/prop4: clockwise (CW)
        # 
        # Mat definition in original code:
        # [Ct, Ct, Ct, Ct] -> Force
        # [Ct*L, -Ct*L, -Ct*L, Ct*L] -> Mx (+ - - +)
        # [-Ct*L, -Ct*L, Ct*L, Ct*L] -> My (- - + +)
        # [-Cd, Cd, -Cd, Cd] -> Mz (- + - +)
        
        self.mat = np.array([
            [self.Ct, self.Ct, self.Ct, self.Ct],                                   # F total
            [self.Ct*self.L, -self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L],     # Mx
            [-self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L, self.Ct*self.L],     # My
            [-self.Cd, self.Cd, -self.Cd, self.Cd]                                  # Mz
        ])
        
        self.inv_mat = np.linalg.inv(self.mat)

    def calculate(self, thrust, mx, my, mz):
        """
        Calculates motor speeds (kRPM) from desired thrust and torques with saturation handling.
        """
        Mx, My = mx, my
        Mz = 0 # Prioritize X/Y, set Z to 0 initially
        
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        
        # --- Saturation Handling for X/Y ---
        max_value = np.max(motor_speed_squ)
        min_value = np.min(motor_speed_squ)
        ref_value = np.sum(motor_speed_squ) / 4.0
        
        max_trim_scale = 1.0
        min_trim_scale = 1.0
        
        if max_value > self.max_speed ** 2:
            max_trim_scale = (self.max_speed ** 2 - ref_value) / (max_value - ref_value) if (max_value - ref_value) != 0 else 0
            
        if min_value < 0:
            min_trim_scale = (ref_value) / (ref_value - min_value) if (ref_value - min_value) != 0 else 0
            
        scale = min(max_trim_scale, min_trim_scale)
        
        # Apply scaling to X/Y torques
        Mx = Mx * scale  
        My = My * scale
        
        # Re-calculate with scaled torques
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        
        if scale < 1.0:
            # If we had to scale down, we don't add Z torque
            motor_speed_squ = np.abs(motor_speed_squ)
            return np.sqrt(motor_speed_squ)
        else:
            # We have headroom, add Z torque
            Mz = mz
            control_input_withz = np.array([thrust, Mx, My, Mz])
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            
            # --- Saturation Handling for Z ---
            max_value = np.max(motor_speed_squ_withz)
            min_value = np.min(motor_speed_squ_withz)
            max_index = np.argmax(motor_speed_squ_withz)
            min_index = np.argmin(motor_speed_squ_withz)
            
            max_trim_scale_z = 1.0
            min_trim_scale_z = 1.0
            
            if max_value > self.max_speed ** 2:
                denom = max_value - motor_speed_squ[max_index]
                max_trim_scale_z = (self.max_speed ** 2 - motor_speed_squ[max_index]) / denom if denom != 0 else 0
                
            if min_value < 0:
                denom = motor_speed_squ[min_index] - min_value
                min_trim_scale_z = (motor_speed_squ[min_index]) / denom if denom != 0 else 0
                
            scale_z = min(max_trim_scale_z, min_trim_scale_z)
            
            Mz = Mz * scale_z
            
            control_input_withz = np.array([thrust, Mx, My, Mz])
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            
            motor_speed_squ = np.abs(motor_speed_squ_withz)
            return np.sqrt(motor_speed_squ)

    def simple_mix(self, total_thrust, torques):
        """
        A simpler mixing strategy that returns motor thrusts (N) instead of RPM.
        Used for simplified simulation control.
        """
        # Note: This matrix must match the one used in `apply_motor_thrusts` in original code
        # The original code uses:
        # [c, c, c, c]
        # [c*L, -c*L, -c*L, c*L]
        # [-c*L, -c*L, c*L, c*L]
        # [-b, b, -b, b]
        # This matches self.mat if c=Ct, b=Cd.
        
        forces_torques = np.array([total_thrust, torques[0], torques[1], torques[2]])
        
        # Solve for squared speeds first (or pseudo-thrusts if we consider T ~ w^2 directly)
        # If we want motor THRUSTS (N), we need to invert the relationship carefully.
        # But for simplified control where we just want to distribute forces:
        
        # Let's assume the mixing matrix relates motor THRUSTS to body wrench directly:
        # F_total = sum(Fi)
        # Mx = L * (F0 + F3 - F1 - F2)  (Based on signs in original matrix)
        # ...
        
        # Let's reconstruct the matrix for Force -> Wrench
        # Note: Original code uses self.thrust_coeff=1 for simple mixing
        c = 1.0
        b = 0.15 # torque_coeff
        L = self.L
        
        # Matrix mapping [F0, F1, F2, F3] -> [F_total, Mx, My, Mz]
        # Indices: 0:FrontLeft, 1:FrontRight, 2:BackRight, 3:BackLeft
        # Check signs from original:
        # F_total: + + + +
        # Mx: + - - +  => F0, F3 pos; F1, F2 neg
        # My: - - + +  => F2, F3 pos; F0, F1 neg
        # Mz: - + - +  => F1, F3 pos; F0, F2 neg
        
        mixing_matrix = np.array([
            [c, c, c, c],          # Total Thrust
            [c*L, -c*L, -c*L, c*L],         # Mx
            [-c*L, -c*L, c*L, c*L],         # My
            [-b, b, -b, b]         # Mz
        ])
        
        motor_thrusts = np.linalg.inv(mixing_matrix) @ forces_torques
        return motor_thrusts
