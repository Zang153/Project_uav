"""
Logger utility for recording simulation data and plotting results.
"""

import os
import platform
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation as R

# Configure matplotlib backend for macOS compatibility
if platform.system() == 'Darwin':
    matplotlib.use('Agg')
    # print("Detected macOS, using Agg backend for Matplotlib (saving to file instead of showing window)")

import matplotlib.pyplot as plt

class Logger:
    """
    Records simulation data and generates plots.
    """
    def __init__(self):
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'target_velocity': [],
            'euler': [],
            'target_position': [],
            'target_euler': [],
            'angle_rate': [],
            'target_angle_rate': [],
            'motor_thrusts': [],
            'motor_mix': []
        }

    def log(self, time, position, velocity, attitude_quat, angle_rate, 
            target_pos, target_vel, target_att_quat, target_rate, 
            motor_thrusts, mixer_outputs):
        """
        Appends a new data point to history.
        
        Args:
            time: Current simulation time.
            position: Current position [x, y, z].
            velocity: Current velocity [vx, vy, vz].
            attitude_quat: Current attitude quaternion [w, x, y, z].
            angle_rate: Current angular velocity [p, q, r].
            target_pos: Target position.
            target_vel: Target velocity.
            target_att_quat: Target attitude quaternion.
            target_rate: Target angular rate.
            motor_thrusts: Array of motor thrusts.
            mixer_outputs: Mixer outputs (Forces + Torques).
        """
        # Convert quaternions to Euler angles for logging/plotting (ZYX order)
        # Note: input quat is [w, x, y, z] (scalar first)
        
        # Current Euler
        try:
            r = R.from_quat([attitude_quat[1], attitude_quat[2], attitude_quat[3], attitude_quat[0]]) # Expects [x, y, z, w]
            euler = r.as_euler('ZYX', degrees=True)
        except Exception:
            euler = np.zeros(3)

        # Target Euler
        try:
            # Check if target_att_quat is a numpy-quaternion object or array
            if hasattr(target_att_quat, 'w'):
                t_q = [target_att_quat.x, target_att_quat.y, target_att_quat.z, target_att_quat.w]
            else:
                # Assuming [w, x, y, z]
                t_q = [target_att_quat[1], target_att_quat[2], target_att_quat[3], target_att_quat[0]]
            
            r_t = R.from_quat(t_q)
            target_euler = r_t.as_euler('ZYX', degrees=True)
        except Exception:
            target_euler = np.zeros(3)

        self.history['time'].append(time)
        self.history['position'].append(np.array(position).copy())
        self.history['velocity'].append(np.array(velocity).copy())
        self.history['target_velocity'].append(np.array(target_vel).copy())
        self.history['euler'].append(euler.copy())
        self.history['target_position'].append(np.array(target_pos).copy())
        self.history['target_euler'].append(target_euler.copy())
        self.history['angle_rate'].append(np.array(angle_rate).copy())
        self.history['target_angle_rate'].append(np.array(target_rate).copy())
        self.history['motor_thrusts'].append(np.array(motor_thrusts).copy())
        self.history['motor_mix'].append(np.array(mixer_outputs).copy())

    def plot_results(self, save_path='simulation_results.png'):
        """
        Plots the recorded history.
        """
        try:
            time_array = np.array(self.history['time'])
            if len(time_array) == 0:
                print("No data to plot.")
                return

            position = np.array(self.history['position'])
            target_position = np.array(self.history['target_position'])
            
            vel = np.array(self.history['velocity'])
            target_vel = np.array(self.history['target_velocity'])
            
            euler = np.array(self.history['euler'])
            target_euler = np.array(self.history['target_euler'])
            
            angle_rate = np.array(self.history['angle_rate'])
            target_angle_rate = np.array(self.history['target_angle_rate'])
            
            motor_thrusts = np.array(self.history['motor_thrusts'])
            motor_mix = np.array(self.history['motor_mix'])
            
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))
            
            # 1. Position Tracking
            axes[0, 0].plot(time_array, position[:, 0], 'r-', label='X')
            axes[0, 0].plot(time_array, position[:, 1], 'g-', label='Y')
            axes[0, 0].plot(time_array, position[:, 2], 'b-', label='Z')
            axes[0, 0].plot(time_array, target_position[:, 0], 'r--', label='Ref X')
            axes[0, 0].plot(time_array, target_position[:, 1], 'g--', label='Ref Y')
            axes[0, 0].plot(time_array, target_position[:, 2], 'b--', label='Ref Z')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].set_title('Position Tracking')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 2. Velocity Tracking
            axes[1, 0].plot(time_array, vel[:, 0], 'r-', label='Vx')
            axes[1, 0].plot(time_array, vel[:, 1], 'g-', label='Vy')
            axes[1, 0].plot(time_array, vel[:, 2], 'b-', label='Vz')
            axes[1, 0].plot(time_array, target_vel[:, 0], 'r--', label='Ref Vx')
            axes[1, 0].plot(time_array, target_vel[:, 1], 'g--', label='Ref Vy')
            axes[1, 0].plot(time_array, target_vel[:, 2], 'b--', label='Ref Vz')
            axes[1, 0].set_ylabel('Velocity (m/s)')
            axes[1, 0].set_title('Velocity Tracking')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 3. Attitude (Euler) - Separated by axis
            # Yaw
            axes[0, 2].plot(time_array, euler[:, 0], 'r-', label='Yaw')
            axes[0, 2].plot(time_array, target_euler[:, 0], 'b--', label='Ref Yaw')
            axes[0, 2].set_ylabel('Deg')
            axes[0, 2].set_title('Yaw')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # Pitch
            axes[1, 2].plot(time_array, euler[:, 1], 'r-', label='Pitch')
            axes[1, 2].plot(time_array, target_euler[:, 1], 'b--', label='Ref Pitch')
            axes[1, 2].set_ylabel('Deg')
            axes[1, 2].set_title('Pitch')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
            
            # Roll
            axes[2, 2].plot(time_array, euler[:, 2], 'r-', label='Roll')
            axes[2, 2].plot(time_array, target_euler[:, 2], 'b--', label='Ref Roll')
            axes[2, 2].set_ylabel('Deg')
            axes[2, 2].set_title('Roll')
            axes[2, 2].legend()
            axes[2, 2].grid(True)
            
            # 4. Angular Rates
            axes[0, 3].plot(time_array, angle_rate[:, 0], 'r-', label='P (Roll Rate)') # Note: check coordinate system convention
            axes[0, 3].plot(time_array, target_angle_rate[:, 0], 'b--', label='Ref P')
            axes[0, 3].set_title('Roll Rate (rad/s)')
            axes[0, 3].grid(True)
            
            axes[1, 3].plot(time_array, angle_rate[:, 1], 'r-', label='Q (Pitch Rate)')
            axes[1, 3].plot(time_array, target_angle_rate[:, 1], 'b--', label='Ref Q')
            axes[1, 3].set_title('Pitch Rate (rad/s)')
            axes[1, 3].grid(True)
            
            axes[2, 3].plot(time_array, angle_rate[:, 2], 'r-', label='R (Yaw Rate)')
            axes[2, 3].plot(time_array, target_angle_rate[:, 2], 'b--', label='Ref R')
            axes[2, 3].set_title('Yaw Rate (rad/s)')
            axes[2, 3].grid(True)
            
            # 5. Forces and Torques
            axes[0, 1].plot(time_array, motor_mix[:, 0], 'k-', label='Thrust Total')
            axes[0, 1].set_ylabel('Force (N)')
            axes[0, 1].set_title('Total Thrust')
            axes[0, 1].grid(True)
            
            axes[1, 1].plot(time_array, motor_mix[:, 3], 'r-', label='Mx')
            axes[1, 1].plot(time_array, motor_mix[:, 4], 'g-', label='My')
            axes[1, 1].plot(time_array, motor_mix[:, 5], 'b-', label='Mz')
            axes[1, 1].set_ylabel('Torque (Nm)')
            axes[1, 1].set_title('Control Torques')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            # 6. Motor Thrusts
            for i in range(4):
                axes[2, 1].plot(time_array, motor_thrusts[:, i], label=f'M{i+1}')
            axes[2, 1].set_ylabel('Thrust (N)')
            axes[2, 1].set_title('Individual Motor Thrusts')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
            
            # 7. Errors
            pos_error = np.linalg.norm(position - target_position, axis=1)
            axes[2, 0].plot(time_array, pos_error, 'k-')
            axes[2, 0].set_ylabel('Error (m)')
            axes[2, 0].set_title('Position Error Norm')
            axes[2, 0].grid(True)
            
            vel_error = np.linalg.norm(vel - target_vel, axis=1)
            axes[3, 0].plot(time_array, vel_error, 'm-')
            axes[3, 0].set_ylabel('Error (m/s)')
            axes[3, 0].set_title('Velocity Error Norm')
            axes[3, 0].grid(True)
            
            # 8. 3D Trajectory
            ax_3d = fig.add_subplot(4, 4, 14, projection='3d')
            ax_3d.plot(position[:, 0], position[:, 1], position[:, 2], 'b-', label='Actual')
            ax_3d.plot(target_position[:, 0], target_position[:, 1], target_position[:, 2], 'r--', label='Ref')
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title('3D Path')
            ax_3d.legend()
            
            # Hide unused subplots
            axes[3, 1].axis('off')
            axes[3, 2].axis('off')
            axes[3, 3].axis('off')
            
            plt.tight_layout()
            
            # Save/Show logic
            if platform.system() == 'Darwin':
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                try:
                    plt.show()
                except Exception:
                    plt.savefig(save_path)
                    print(f"Plot saved to {save_path}")

        except Exception as e:
            print(f"Error plotting results: {e}")
