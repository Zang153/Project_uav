import os
import gymnasium as gym
import mujoco
import numpy as np
from abc import ABC, abstractmethod

class BaseMujocoAviary(gym.Env, ABC):
    """
    Base class for MuJoCo-based UAV environments.
    Inspired by the gym-pybullet-drones architecture.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, 
                 xml_path: str = "../meshes/UAV.xml", 
                 render_mode: str = None, 
                 freq: int = 100):
        super().__init__()
        
        # Load MuJoCo model
        self.xml_path = os.path.join(os.path.dirname(__file__), xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.freq = freq
        self.model.opt.timestep = 1.0 / self.freq
        
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        
        # Initialize Spaces (must be implemented by subclasses)
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Compute initial observation
        # Do not compute obs here if the subclass needs more initialization first.
        # However, Gym requires reset to return obs, info. We will let subclass override reset if needed.
        obs = self._computeObs()
        info = self._computeInfo()
        
        if self.render_mode == "human":
            self.render()
            
        return obs, info

    def step(self, action):
        # Preprocess action
        processed_action = self._preprocessAction(action)
        
        # Apply action to the system (e.g., set control inputs)
        # Assuming the action space matches the ctrl space, or subclass handles it in _preprocessAction
        self.data.ctrl[:] = processed_action
        
        # Step the physics simulation
        mujoco.mj_step(self.model, self.data)
        
        # Compute new state, reward, termination, and truncation
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    import mujoco_viewer
                    self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                except ImportError:
                    gym.logger.warn("mujoco_viewer is not installed, please install it to use human render mode.")
                    return
            self.viewer.render()
        elif self.render_mode == "rgb_array":
            try:
                if self.renderer is None:
                    self.renderer = mujoco.Renderer(self.model)
                self.renderer.update_scene(self.data)
                return self.renderer.render()
            except Exception as e:
                gym.logger.warn(f"Failed to render rgb_array: {e}")
                return np.zeros((240, 320, 3), dtype=np.uint8)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    @abstractmethod
    def _actionSpace(self) -> gym.Space:
        """Returns the action space of the environment."""
        pass

    @abstractmethod
    def _observationSpace(self) -> gym.Space:
        """Returns the observation space of the environment."""
        pass

    @abstractmethod
    def _computeObs(self) -> np.ndarray:
        """Computes and returns the current observation."""
        pass

    @abstractmethod
    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """Pre-processes the action passed to `.step()`."""
        pass

    @abstractmethod
    def _computeReward(self) -> float:
        """Computes the current reward value."""
        pass

    @abstractmethod
    def _computeTerminated(self) -> bool:
        """Computes the current terminated state."""
        pass

    @abstractmethod
    def _computeTruncated(self) -> bool:
        """Computes the current truncated state."""
        pass

    @abstractmethod
    def _computeInfo(self) -> dict:
        """Computes the current info dict."""
        pass
