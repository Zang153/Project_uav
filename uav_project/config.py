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
MASS = 1.27  # kg
GRAVITY = 9.81  # m/s^2
ARM_LENGTH = 0.18  # m (Motor arm length)

# --- Motor Parameters ---
# Thrust coefficient (N / krpm^2)
CT = 0.01343
# Drag coefficient (Nm / krpm^2)
CD = 3.099e-4
# Max thrust per motor (N)
MAX_THRUST_PER_MOTOR = 6.5
# Max torque per motor (Nm)
MAX_TORQUE_PER_MOTOR = 0.15
# Max motor speed (krpm)
MAX_MOTOR_SPEED_KRPM = 22.0

# --- Controller Frequencies (Hz) ---
FREQ_DELTA = 100
FREQ_POSITION = 50
FREQ_VELOCITY = 50
FREQ_ATTITUDE = 250
FREQ_ANGLE_RATE = 1000

# --- PID Gains ---
# Position Control (P-only)
POS_KP = np.array([8.0, 8.0, 12.0])

# Velocity Control (PID)
VEL_KP = np.array([4.0, 4.0, 6.0])
VEL_KI = np.array([0.0, 0.0, 0.0])
VEL_KD = np.array([0.05, 0.05, 0.05])

# Attitude Control (P-only, Quaternion based)
ATT_KP = np.array([9.0, 9.0, 12.0])

# Angular Rate Control (PID)
RATE_KP = np.array([3.0, 3.0, 3.0])
RATE_KI = np.array([100.0, 100.0, 100.0])
RATE_KD = np.array([0.00005, 0.00005, 0.00005])

# --- Output Limits ---
# Can be added here if needed, currently handled in logic
