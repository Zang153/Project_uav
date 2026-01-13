"""
Simulator module managing the MuJoCo physics loop and rendering.
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
from ..config import SIM_TIMESTEP, RENDER_FPS

class Simulator:
    """
    Manages the simulation loop, timing, and rendering.
    """
    def __init__(self, model_path, controller, logger=None):
        """
        Args:
            model_path: Path to the XML model file.
            controller: Instance of the controller (must have .update(time) method).
            logger: Optional Logger instance.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.controller = controller
        self.logger = logger
        
        # Simulation parameters
        self.timestep = self.model.opt.timestep
        if abs(self.timestep - SIM_TIMESTEP) > 1e-6:
            print(f"Warning: XML timestep ({self.timestep}) differs from config ({SIM_TIMESTEP}). Using XML value.")
            
        self.render_interval = 1.0 / RENDER_FPS
        self.steps_per_render = max(1, int(self.render_interval / self.timestep))

    def run(self, duration=10.0, trajectory=None, headless=False):
        """
        Runs the simulation.
        
        Args:
            duration: Total simulation time (seconds).
            trajectory: List of (time, [x, y, z]) tuples defining the path.
            headless: If True, runs without viewer.
        """
        total_steps = int(duration / self.timestep)
        
        if headless:
            print(f"Simulation started (Headless). Duration: {duration}s, Timestep: {self.timestep}s")
            start_real_time = time.time()
            for step in range(total_steps):
                self._step_simulation(step, trajectory)
            elapsed = time.time() - start_real_time
            print(f"Simulation finished. Real time: {elapsed:.2f}s. Factor: {duration/elapsed:.2f}x")
        else:
            # Initialize Viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                print(f"Simulation started. Duration: {duration}s, Timestep: {self.timestep}s")
                start_real_time = time.time()
                
                for step in range(total_steps):
                    self._step_simulation(step, trajectory)
                    
                    # 5. Rendering
                    if step % self.steps_per_render == 0:
                        viewer.sync()
                        
                        # Optional: Print progress
                        if step % 5000 == 0:
                            progress = step / total_steps * 100
                            print(f"Progress: {progress:.1f}%")
                
                elapsed = time.time() - start_real_time
                print(f"Simulation finished. Real time: {elapsed:.2f}s. Factor: {duration/elapsed:.2f}x")

    def _step_simulation(self, step, trajectory):
        sim_time = step * self.timestep
        
        # 1. Update Trajectory Target
        if trajectory:
            self._update_target(sim_time, trajectory)
        
        # 2. Update Controller
        # The controller internals handle multi-rate updates
        self.controller.update(sim_time)
        
        # 3. Step Physics
        mujoco.mj_step(self.model, self.data)
        
        # 4. Logging (decimated, e.g., every 10 steps = 1000Hz)
        if self.logger and step % 10 == 0:
            log_data = self.controller.get_log_data()
            self.logger.log(*log_data)

        if step % 100 == 0:
            self.controller.print_state() # Disable print for cleaner output in headless

    def _update_target(self, current_time, trajectory):
        """
        Interpolates trajectory to find current target position.
        """
        # Linear interpolation between waypoints
        for i in range(len(trajectory) - 1):
            t_start, pos_start = trajectory[i]
            t_end, pos_end = trajectory[i+1]
            
            if t_start <= current_time < t_end:
                alpha = (current_time - t_start) / (t_end - t_start)
                target_pos = np.array(pos_start) + alpha * (np.array(pos_end) - np.array(pos_start))
                self.controller.set_target_position(target_pos)
                break
        
        # Hold last point
        if current_time >= trajectory[-1][0]:
            self.controller.set_target_position(trajectory[-1][1])
