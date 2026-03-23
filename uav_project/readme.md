# UAV Project - Delta Sim

This folder contains the main codebase for the UAV Delta Robot simulation project. It is the foundation for ongoing research.

## Environment Setup

This project uses **Conda** for environment management. Follow the steps below to set up the environment on your machine.

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.
- Git installed.

### Installation Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Zang153/Project_uav.git
    cd Project_uav
    ```

2.  **Create Conda Environment**

    We use an environment named `mujoco-sim` for this project. The `environment.yml` uses Conda to install most dependencies, but explicitly uses `pip` to install PyTorch with CUDA 13.0 support for RTX 4090 compatibility.
    
    You can create it directly from the `environment.yml` file:

    ```bash
    cd .. # Ensure you are in the root directory of Project_uav
    conda env create -f environment.yml
    ```

3.  **Activate Environment**

    ```bash
    conda activate mujoco-sim
    ```

    > **Note for Windows PowerShell Users:**
    > If you encounter errors activating the environment, you may need to initialize conda for PowerShell and set the execution policy:
    > ```powershell
    > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    > conda init powershell
    > ```
    > Restart your terminal after running these commands.

4.  **Verify Dependencies**

    Verify that the core dependencies, including `torch` and `mujoco`, are installed correctly:

    ```bash
    python -c "import matplotlib, scipy, numpy, quaternion, mujoco, torch; print('All dependencies installed successfully!')"
    ```

### Running the Simulation

To run the main simulation script:

```bash
cd uav_project
python main.py
```

## Dependencies
Core libraries:
- `torch>=2.0.0` (For tensor-based kinematics and control loops)
- `mujoco>=3.1.0` (Physics engine)
- `matplotlib>=3.7.0` (For plotting simulation results)
- `numpy>=1.24.0`
- `stable-baselines3[extra]>=2.2.1` (For reinforcement learning training and evaluation)
- `mujoco-python-viewer>=0.1.4` (For rendering the 3D simulation environment)

## Project Structure

- `controllers/`: Control algorithms (PID, Cascade, etc.)
- `meshes/`: STL files and XML models for the robot.
- `models/`: Robot model definitions and mixer logic.
- `simulation/`: MuJoCo simulator interface.
- `rl_envs/`: Reinforcement Learning Gymnasium environments (Hover, Track).
- `utils/`: Helper functions, kinematics, and logging.
- `main.py`: Entry point for the traditional PID simulation.
- `train_delta_*.py`: Training scripts for reinforcement learning tasks.
- `enjoy_delta_*.py`: Testing/Visualization scripts for trained RL models.

## Reinforcement Learning (RL) Workflows

This project uses `Stable-Baselines3` to train policies for the UAV with the Delta arm.

### 1. Training a Policy

There are currently two main RL tasks:
- **Hover Task**: Train the UAV to hover stably with the Delta arm attached.
  ```bash
  python train_delta_hover_rl.py
  ```
- **Track Task**: Train the UAV to hover while the Delta arm tracks a circular trajectory.
  ```bash
  python train_delta_track_rl.py
  ```
*Note: Training scripts use `SubprocVecEnv` with 32 parallel environments by default to maximize CPU utilization. The training duration is defined by `RL_EPISODE_DURATION` in `config.py`.*

### 2. Evaluating and Visualizing (Enjoy)

Once a model is trained (or during training), you can visualize the best model's performance:
- **Hover Task**:
  ```bash
  python enjoy_delta_hover_rl.py
  ```
- **Track Task**:
  ```bash
  python enjoy_delta_track_rl.py
  ```
*These scripts will open a 3D MuJoCo viewer (if X11/display is available) and print real-time state and tracking errors to the terminal.*

### 3. Monitoring Training Progress (TensorBoard)

During or after training, you can monitor the reward curves, episode lengths, and other metrics using TensorBoard:

```bash
# For the Hover task
tensorboard --logdir=rl_results/delta_hover_logs --port=6006

# For the Track task
tensorboard --logdir=rl_results/delta_track_logs --port=6006
```
Then, open your browser and navigate to `http://localhost:6006/`.

## Troubleshooting

- **ModuleNotFoundError**: Ensure you have activated the correct environment (`conda activate mujoco-sim`) and that all dependencies installed correctly.
- **No plot output**: If running via SSH without X11 forwarding, Matplotlib uses the `Agg` backend to save the plot as `simulation_results.png` instead of displaying it. This is configured in `utils/logger.py`.
- **Type errors**: If modifying PID gains, ensure they are converted to PyTorch tensors to maintain compatibility with the vectorized controllers (see `controllers/pid.py`).
- **CUDA errors**: PyTorch is installed with CUDA 13.0 support. Make sure your NVIDIA drivers (e.g., RTX 4090) are updated to support CUDA 13.0.
