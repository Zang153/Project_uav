import os
import sys
import time
import numpy as np

# 确保能正确导入当前项目的包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from uav_project.rl_envs.HoverMujocoAviary import HoverMujocoAviary
from uav_project.utils.logger import Logger

def main():
    # 1. 找到我们训练好的最佳模型
    # 使用绝对路径或相对于项目根目录的路径，以避免工作目录带来的问题
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "rl_results", "models", "best_model.zip")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] 找不到模型文件: {model_path}")
        print("请先运行 train_rl.py 训练模型！")
        return

    print(f"[INFO] 正在加载模型: {model_path}")
    # 2. 加载模型
    model = PPO.load(model_path)

    # 3. 测试配置
    NUM_TESTS = 3
    TEST_DURATION = 15.0 
    
    print(f"[INFO] 初始化测试环境，共进行 {NUM_TESTS} 次测试，每次 {TEST_DURATION} 秒...")

    env = HoverMujocoAviary(render_mode="human")
    
    # 获取控制频率，计算总步数和渲染间隔
    # 假设 HoverMujocoAviary 内部有 control_freq 属性，如果没有默认用 100
    control_freq = getattr(env, "control_freq", 100)
    test_steps = int(TEST_DURATION * control_freq)
    TARGET_FPS = 30
    render_interval = max(1, int(control_freq / TARGET_FPS)) 

    # 4. 运行多次测试循环
    for test_idx in range(1, NUM_TESTS + 1):
        print(f"\n==================================================")
        print(f"[INFO] 开始测试 {test_idx}/{NUM_TESTS}")
        print(f"==================================================")
        
        obs, info = env.reset()
        logger = Logger()
        
        episode_start_time = time.time()
        steps_in_current_episode = 0
        
        for i in range(test_steps):
            # 使用模型预测动作 (确定性策略)
            action, _states = model.predict(obs, deterministic=True)
            
            # 将动作输入到环境中
            obs, reward, terminated, truncated, info = env.step(action)
            steps_in_current_episode += 1
            sim_time = steps_in_current_episode / control_freq
            
            # --- 数据记录 (Logging) ---
            current_pos = info.get("current_pos", env.data.qpos[:3].copy())
            current_vel = env.data.qvel[:3].copy()
            current_quat = env.data.qpos[3:7].copy()
            current_ang_vel = env.data.qvel[3:6].copy()
            target_pos = env.TARGET_POS if hasattr(env, "TARGET_POS") else np.array([0, 0, 1.0])
            
            logger.log(
                time=sim_time,
                position=current_pos,
                velocity=current_vel,
                attitude_quat=current_quat,
                angle_rate=current_ang_vel,
                target_pos=target_pos,
                target_vel=np.zeros(3),
                target_att_quat=np.array([1, 0, 0, 0]),
                target_rate=np.zeros(3),
                motor_thrusts=np.zeros(4), # 占位
                mixer_outputs=np.zeros(6), # 占位
                delta_des_pos=np.zeros(3),
                delta_actual_pos=np.zeros(3)
            )
            
            # --- 终端输出 ---
            if steps_in_current_episode % 10 == 0:
                dist_err = np.linalg.norm(target_pos - current_pos)
                print(f"\r[测试 {test_idx} | 步数 {steps_in_current_episode:4d}] "
                      f"位置: ({current_pos[0]: 5.2f}, {current_pos[1]: 5.2f}, {current_pos[2]: 5.2f}) | "
                      f"误差: {dist_err:5.3f}m ", end="")
            
            # --- 渲染画面 (优化渲染频率，丢弃强制 sleep) ---
            if i % render_interval == 0:
                env.render()

            # 如果提前结束
            if terminated or truncated:
                print(f"\n[INFO] 触发终止条件，提前结束。")
                break
                
        # 单次测试结束统计
        episode_end_time = time.time()
        real_time_elapsed = episode_end_time - episode_start_time
        sim_time_elapsed = steps_in_current_episode / control_freq
        
        print(f"\n[INFO] 测试 {test_idx} 结束.")
        print(f"       - 仿真耗时: {sim_time_elapsed:.2f}s")
        print(f"       - 真实耗时: {real_time_elapsed:.2f}s")
        
        # 保存图表
        plot_path = os.path.join(project_dir, f'rl_hover_test_{test_idx}.png')
        logger.plot_results(save_path=plot_path)
        print(f"[INFO] 性能曲线已保存至 {plot_path}")

    env.close()
    print("\n[INFO] 所有测试完成.")

if __name__ == "__main__":
    main()
