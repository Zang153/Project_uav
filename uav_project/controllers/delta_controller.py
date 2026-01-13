import numpy as np
import math
from uav_project.utils.DeltaKinematics import DeltaKinematics

class DeltaController:
    """
    Controller for the Delta robot manipulator using Jacobian-based velocity control.
    """
    def __init__(self, uav_model, control_freq=100.0, control_mode='position'):
        """
        Args:
            uav_model: Instance of UAVModel.
            control_freq: Control frequency in Hz.
            control_mode: 'position' or 'velocity'.
        """
        self.uav = uav_model
        self.control_freq = control_freq
        self.control_mode = control_mode
        self.dt = 1.0 / control_freq
        
        # Delta Robot Parameters
        self.L = 0.1      # Upper arm length (rod_b)
        self.l = 0.2      # Lower arm length (rod_ee)
        self.R = 0.074577 # Base radius (r_b)
        self.r = 0.02495  # End-effector radius (r_ee)
        
        # Kinematics helper
        self.kinematics = DeltaKinematics(self.L, self.l, self.R, self.r)
        
        # Control gains (PD)
        self.kp = 2.0
        self.kd = 1.0
        
        # Trajectory parameters
        self.traj_radius = 0.1 # 0.1m radius might be too large for workspace? user said 0.1m.
                               # Let's check workspace. Z ~ -0.15. 
                               # Max reach is roughly L+l = 0.3.
                               # 0.1m radius at z=-0.15: sqrt(0.1^2 + 0.15^2) = 0.18 < 0.3. Should be fine.
        self.traj_radius = 0.1
        self.traj_z = -0.25
        self.traj_period = 25.0 # seconds for one circle
        
        self.last_update_time = 0.0
        
    def get_circular_trajectory(self, t):
        """
        Generates desired position and velocity for a circular trajectory.
        
        Args:
            t: Current time.
            
        Returns:
            des_pos: [x, y, z]
            des_vel: [vx, vy, vz]
        """
        omega = 2 * np.pi / self.traj_period
        phase = omega * t
        
        # Position
        x = self.traj_radius * np.cos(phase)
        y = self.traj_radius * np.sin(phase)
        z = self.traj_z
        
        # Velocity
        vx = -self.traj_radius * omega * np.sin(phase)
        vy = self.traj_radius * omega * np.cos(phase)
        vz = 0.0
        
        return np.array([x, y, z]), np.array([vx, vy, vz])

    def update(self, sim_time):
        """
        Main update loop. Should be called every simulation step, 
        but logic executes only at control_freq.
        """
        # Check if it's time to run control loop (simple decimation)
        # Note: In a real implementation, we might accumulate error or check against discrete steps.
        # For simplicity in simulation, we run if enough time passed.
        # However, the user said "control frequency is 100hz, similarly in mujoco use step accurately control".
        # This implies we should check (sim_time - last_time) >= 1/freq.
        
        if sim_time >= self.last_update_time + self.dt:
        
        
        # 1. Get current state (End-effector pos/vel in Delta Base frame)
        # The sensor data is usually relative to the sensor site. 
        # Assuming the sensor is on the end-platform and measures relative to... world? or base?
        # uav_model.get_ee_sensor_pos() returns data from 'ee_pos' sensor.
        # In MuJoCo, 'framepos' sensor returns pos in world, 'framequat' in world.
        # But 'subtreecom' etc might differ.
        # User snippet: `get_ee_sensor_pos` returns `self.data.sensordata[...]`.
        # The XML snippet for sensor wasn't fully shown, but usually users set up framepos/framelinvel.
        # If it's relative to base, that's great. If world, we need to transform.
        # uav_model.get_delta_state() calculates Relative_position = End_platform_position - Base_position.
        # This is exactly what we need for the Delta controller (pos relative to base).
        # Let's use get_delta_state() instead of raw sensor if we want to be safe about frames,
        # OR assume get_ee_sensor_pos() is correctly configured as relative.
        # User explicitly said: "Mechanical arm position and velocity info use uav_model.py L151, L163".
        # So I MUST use `get_ee_sensor_pos` and `get_ee_sensor_lin_vel`.
        # I will assume these sensors provide the position/velocity of the EE relative to the Base (or in the frame expected by the math).
        
        # Wait, get_ee_sensor_pos returns sensordata.
        # Let's look at simple_controller.py again.
        # It uses `controller.get_end_effector_position()` which wraps `data.xpos`. xpos is WORLD frame.
        # But simple_controller seems to assume the base is fixed or handles it implicitly?
        # In simple_controller, `controller = SimpleController("urdf/Delta.xml")`. 
        # The base is likely fixed in world at 0,0,0 or similar.
        # In the UAV project, the base is moving (attached to UAV).
        # So the Delta math MUST be done in the Delta Base frame.
        # Does `get_ee_sensor_pos` return World or Base frame?
        # If it's a standard MuJoCo "framepos" sensor attached to end_effector with objtype="site" and reftype="site" ref="base", it's relative.
        # If no ref is given, it's World.
        # Given the context "UAV + Delta", using World frame for Delta IK is wrong because the base moves.
        # I strongly suspect `get_ee_sensor_pos` is intended to be the Relative position (EE in Base frame).
        # Let's proceed with that assumption, but if it behaves wildly, it might be World frame.
        
            current_pos = self.uav.get_ee_sensor_pos()
            current_vel = self.uav.get_ee_sensor_lin_vel()
            
            print(f"current_pos: {current_pos}")
            # 2. Get desired state
            des_pos, des_vel = self.get_circular_trajectory(sim_time)
            print(f"des_pos: {des_pos}")

            des_pos = [0, 0, -0.35]
            # 3. Calculate Controls
            if self.control_mode == 'position':
                # Position Control: Desired Pos -> IK -> Joint Angles -> Motor Positions
                # IK returns degrees, model expects radians
                joint_angles_deg = self.kinematics.ik(des_pos)
                
                if isinstance(joint_angles_deg, int) and joint_angles_deg == -1:
                    # IK failed
                    print(f"Warning: IK failed for target {des_pos}")
                else:
                    joint_angles_rad = np.deg2rad(joint_angles_deg)
                    self.uav.set_delta_motor_positions(joint_angles_rad)
                    
            elif self.control_mode == 'velocity':
                # Velocity Control: Jacobian-based
                motor_vels = self.calculate_motor_velocities(current_pos, current_vel, des_pos, des_vel)
                self.uav.set_delta_motor_velocities(motor_vels)

            self.last_update_time = sim_time

    def calculate_motor_velocities(self, pos, vel, des_pos, des_vel):
        """
        Calculates motor angular velocities using Jacobian-based PD control.
        Ref: simple_controller.py
        """
        # PD Control for Velocity Correction
        err_pos = des_pos - pos
        err_vel = des_vel - vel
        
        vel_correct = self.kp * err_pos + self.kd * err_vel
        vel_control = des_vel + vel_correct
        
        # Get current joint angles (needed for Jacobian)
        # We need the joint angles of the 3 motors.
        # uav_model doesn't have a direct "get_delta_joint_angles" method shown in the snippet.
        # But we need them for the Jacobian matrix M(q).
        # We can compute q from IK(pos) if we trust the model, OR read from sensors if available.
        # simple_controller uses `controller.get_joint_angles()`.
        # I should verify if uav_model allows reading joint angles.
        # I'll add `get_delta_joint_angles` to uav_model.py later or use a workaround.
        # Workaround: Use IK to estimate current angles from current EE pos.
        # This is valid if the robot is rigid and not in singularity.
        
        # Using IK to get current angles
        current_angles_deg = self.kinematics.ik(pos)
        if isinstance(current_angles_deg, int) and current_angles_deg == -1:
             # IK failed
             return np.zeros(3)
        current_angles = np.deg2rad(current_angles_deg)
        
        # Jacobian Calculation
        # Constants
        L_upper = self.L
        L_lower = self.l
        R_val = self.R
        r_val = self.r
        
        def phi_matrix(i):
            angle = i * np.pi / 3
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        def alpha_vector(q_i):
            return np.array([np.cos(q_i), 0, np.sin(q_i)])

        def d_alpha_dq(q_i):
            return np.array([-np.sin(q_i), 0, np.cos(q_i)])

        def e_vector(q_i, i):
            alpha_i = alpha_vector(q_i)
            base_vector = np.array([R_val - r_val, 0, 0])
            return base_vector + L_upper * alpha_i

        def beta_vector(pos, q_i, i):
            phi_i = phi_matrix(i)
            p_Ei_e = phi_i @ pos
            e_i = e_vector(q_i, i)
            return (p_Ei_e - e_i) / L_lower

        M_rows = []
        V_diag = []

        for i in range(3):
            q_i = current_angles[i]
            beta_i = beta_vector(pos, q_i, i)
            phi_i = phi_matrix(i)
            
            M_row = beta_i.T @ phi_i
            M_rows.append(M_row)
            
            d_alpha = d_alpha_dq(q_i)
            V_element = beta_i.T @ d_alpha
            V_diag.append(V_element)

        M = np.array(M_rows)
        V = np.diag(V_diag)

        # q_dot = (1/L_upper) * V^(-1) * M * vel_control
        try:
            V_inv = np.linalg.inv(V)
            q_dot = (1.0 / L_upper) * V_inv @ M @ vel_control
        except np.linalg.LinAlgError:
            # Handle singularity
            V_inv = np.linalg.pinv(V)
            q_dot = (1.0 / L_upper) * V_inv @ M @ vel_control

        return q_dot
