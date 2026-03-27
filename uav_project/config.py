"""
Configuration parameters for the UAV simulation and control.
"""
import numpy as np

# --- Simulation Settings ---
# Note: The actual simulation timestep is defined in the XML file (typically 0.0001s).
# This variable is for reference or if we want to override/check it.
SIM_TIMESTEP = 0.0001  
RENDER_FPS = 60

# --- UAV Physical Parameters ---
MASS = 0.4  # kg (Aligned with UAV.xml / Delta.xml UAV_body mass)
GRAVITY = 9.81  # m/s^2
ARM_LENGTH = 0.18  # m (Motor arm length) the distance from the axis to the motor, which means the total length is equal to sqrt(2) * ARM_LENGTH

#                x-aixs
#               ^
# (1)           |            (2)
#    \\\\       |       ////
#       \\\\    |    ////
# y-axis   \\\\ | ////
# <------------(+)-------------
#          //// | \\\\
#       ////    |    \\\\
#    ////       |       \\\\
# (4)           |            (3)

# prop1/prop3: anti_clockwise (CCW)
# prop2/prop4: clockwise (CW)

# --- Motor Parameters ---
# # Max thrust per motor (N)
# MAX_THRUST_PER_MOTOR = 6.5
# # Max torque per motor (Nm)
# MAX_TORQUE_PER_MOTOR = 0.15
# # Max motor speed (krpm)
MAX_MOTOR_SPEED_KRPM = 22.0

# Thrust coefficient (N / krpm^2) 
CT = 0.01343
# Torque coefficient (Nm / krpm^2)
CM = 0.00031



# --- Controller Frequencies (Hz) ---
FREQ_DELTA = 100
FREQ_POSITION = 50
FREQ_VELOCITY = 50
FREQ_ATTITUDE = 250
FREQ_ANGLE_RATE = 1000

# --- PID Gains ---
# Position Control (P-only)
POS_KP = np.array([2.0, 2.0, 2.0])

# Velocity Control (PID)
VEL_KP = np.array([2.0, 2.0, 3.0])
VEL_KI = np.array([0.5, 0.5, 1.0])
VEL_KD = np.array([0.05, 0.05, 0.05])

# Attitude Control (P-only, Quaternion based)
ATT_KP = np.array([10.0, 10.0, 6.0])

# Angular Rate Control (PID)
RATE_KP = np.array([0.15, 0.15, 0.05])
RATE_KI = np.array([0.05, 0.05, 0.02])
RATE_KD = np.array([0.005, 0.005, 0.001])

# --- Output Limits ---
# Can be added here if needed, currently handled in logic

# --- Reinforcement Learning Global Settings ---
RL_CONTROL_FREQ = 100       # RL agent decision frequency (Hz). If physics is 10000Hz, agent decides every 100 physics steps.

# --- Reinforcement Learning Task Configurations ---
# Dictionary to store hyper-parameters and config for each specific training task.
# This centralizes all tuning parameters so they don't have to be hardcoded in train_*.py files.
TRAINING_CONFIGS = {
    "uav_hover": {
        "num_envs": 32,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "episode_duration_sec": 10.0,
        "eval_freq_sec": 20.0,
        "total_train_sec": 100000.0, # e.g. 100,000s * 100Hz = 10,000,000 steps
        "reward_threshold": np.inf,  # Disable early stopping
        "model_save_dir": "models",
        "log_save_dir": "logs",
        "model_name": "ppo_hover_final"
    },
    "uav_track": {
        "num_envs": 32,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "episode_duration_sec": 20.0,
        "eval_freq_sec": 40.0,
        "total_train_sec": 150000.0, # Tracking is harder, needs more time
        "reward_threshold": np.inf,
        "model_save_dir": "track_models",
        "log_save_dir": "track_logs",
        "model_name": "ppo_track_final"
    },
    "delta_hover": {
        "num_envs": 32,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "episode_duration_sec": 10.0,
        "eval_freq_sec": 20.0,
        "total_train_sec": 100000.0,
        "reward_threshold": np.inf,
        "model_save_dir": "delta_hover_models",
        "log_save_dir": "delta_hover_logs",
        "model_name": "ppo_delta_hover_final"
    },
    "delta_track": {
        "num_envs": 32,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "episode_duration_sec": 20.0,
        "eval_freq_sec": 40.0,
        "total_train_sec": 150000.0,
        "reward_threshold": np.inf,
        "model_save_dir": "delta_track_models",
        "log_save_dir": "delta_track_logs",
        "model_name": "ppo_delta_track_final"
    },
    "disturbance_hover": {
        "num_envs": 32,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "episode_duration_sec": 15.0, # Needs longer duration to experience full disturbance cycle
        "eval_freq_sec": 30.0,
        "total_train_sec": 250000.0, # Much harder task, requires massive data to generalize over disturbance domains
        "reward_threshold": np.inf,
        "model_save_dir": "disturbance_hover_models",
        "log_save_dir": "disturbance_hover_logs",
        "model_name": "ppo_disturbance_hover_final"
    }
}
