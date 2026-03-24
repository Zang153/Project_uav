"""
Test script to verify the correctness of the DeltaRandomTrajectoryGenerator
and ensure the simulation runs smoothly without IK errors or physics crashes.
"""
import os
import sys
import mujoco
import numpy as np
import time

# Add the project root to sys.path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from uav_project.simulation.simulator import Simulator
from uav_project.controllers.cascade_controller import CascadeController
from uav_project.models.uav_model import UAVModel
from uav_project.utils.logger import Logger
from uav_project.utils.DeltaRandomTrajectoryGenerator import DeltaRandomTrajectoryGenerator

class TestController:
    """
    A simple controller wrapper that runs the UAV CascadeController 
    and drives the Delta arm using the DeltaRandomTrajectoryGenerator.
    """
    def __init__(self, uav_model, trajectory_generator):
        self.uav_model = uav_model
        
        # 1. Initialize UAV PID Controller to hover at [0, 0, 1.5]
        self.uav_controller = CascadeController(uav_model)
        self.uav_controller.set_target_position([0.0, 0.0, 1.5])
        
        # 2. Store the trajectory generator for the Delta arm
        self.trajectory_generator = trajectory_generator
        
        # 3. State logging for the Delta arm
        self.current_des_pos_log = np.zeros(3)
        self.current_actual_pos_log = np.zeros(3)

    def update(self, sim_time: float) -> None:
        """
        Updates both the UAV hover control and the Delta arm random trajectory.
        """
        # Update UAV PID
        self.uav_controller.update(sim_time)
        
        # Query the generator for current joint angles
        # The generator works in continuous time, so we just pass sim_time
        cartesian_pos, joint_angles_rad = self.trajectory_generator.get_state(sim_time)
        
        # Apply to the Delta arm in MuJoCo
        if getattr(self.uav_model, 'has_delta', False):
            self.uav_model.set_delta_motor_positions(joint_angles_rad)
            
            # Log the desired and actual positions for the logger
            self.current_des_pos_log = cartesian_pos
            # Use the UAVModel's method to get the actual EE position
            self.current_actual_pos_log = self.uav_model.get_ee_sensor_pos()

    def set_target_position(self, pos: list) -> None:
        self.uav_controller.set_target_position(pos)

    def get_log_data(self):
        # We return the UAV log data AND the Delta arm log data
        uav_data = self.uav_controller.get_log_data()
        return uav_data + (self.current_des_pos_log, self.current_actual_pos_log)

    def print_state(self):
        pass


def main():
    # 0. Test Parameters
    total_sim_time = 16.0
    trajectory_period = 8.0
    
    print(f"[INFO] Starting DeltaRandomTrajectoryGenerator Verification Test")
    print(f"[INFO] Total Simulation Time: {total_sim_time}s")
    print(f"[INFO] Random Trajectory Period: {trajectory_period}s")

    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = "Delta.xml" 
    model_path = os.path.join(current_dir, "meshes", model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 2. Initialize Components
    logger = Logger()
    simulator = Simulator(model_path, None, logger) 
    uav_model = UAVModel(simulator.model, simulator.data)
    
    # 3. Initialize Generator with fixed 8s period
    generator = DeltaRandomTrajectoryGenerator()
    generator.reset(fixed_period=trajectory_period)
    
    print(f"[INFO] Generated Random Trajectory Params:")
    print(f"       Center Z: {generator.z_center:.3f}m")
    print(f"       XY Radius: {generator.xy_radius:.3f}m")
    print(f"       Z Amp: {generator.z_amp:.3f}m")
    print(f"       Freqs (X, Y, Z): ({generator.freq_x:.3f}, {generator.freq_y:.3f}, {generator.freq_z:.3f}) Hz")

    # 4. Combine into custom controller
    controller = TestController(uav_model, generator)
    simulator.controller = controller

    # 5. Run Simulation (Headless=False to visually verify)
    print("\n[INFO] Launching MuJoCo Viewer. Watch the Delta arm move!")
    simulator.run(duration=total_sim_time, trajectory=None, headless=False, print_state_info=False)
    
    # 6. Plot Results
    logger.plot_results(save_path=os.path.join(current_dir, 'test_trajectory_results.png'))
    print(f"\n[INFO] Test finished successfully. No kinematics errors occurred.")

if __name__ == "__main__":
    main()