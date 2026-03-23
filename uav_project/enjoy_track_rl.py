import os
import sys
import time

# 确保能正确导入当前项目的包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from uav_project.rl_envs.TrackCircularMujocoAviary import TrackCircularMujocoAviary

def main():
    # 1. 找到我们训练好的最佳模型
    # 使用绝对路径或相对于项目根目录的路径，以避免工作目录带来的问题
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "uav_project", "rl_results", "track_models", "best_model.zip")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] 找不到模型文件: {model_path}")
        print("请先运行 train_track_rl.py 训练模型！")
        return

    print(f"[INFO] 正在加载模型: {model_path}")
    # 2. 加载模型
    model = PPO.load(model_path)

    # 3. 创建测试环境 (这里我们需要开启渲染)
    # 注意：在实例化时传入 render_mode="human" 才能开启画面
    print("[INFO] 初始化测试环境...")
    env = TrackCircularMujocoAviary(render_mode="human")

    # 4. 测试循环
    obs, info = env.reset()
    
    print("[INFO] 开始测试...")
    for i in range(1000): # 运行 1000 步看看效果
        # 使用模型预测动作
        # deterministic=True 表示使用确定性策略（直接输出最大概率的动作，不随机采样）
        action, _states = model.predict(obs, deterministic=True)
        
        # 将动作输入到环境中
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染画面 (如果你在 BaseMujocoAviary 中实现了 viewer)
        env.render()
        
        # 为了让人眼能看清，稍微睡一下 (对应 30Hz 控制频率)
        time.sleep(1/30.0)

        # 如果无人机飞出界了，或者到达目标了，重置环境
        if terminated or truncated:
            print(f"[INFO] 回合结束 (步数: {i}). 重置环境...")
            obs, info = env.reset()

    env.close()
    print("[INFO] 测试结束.")

if __name__ == "__main__":
    main()
