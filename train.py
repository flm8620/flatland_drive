# train.py
# 最小化2D自动驾驶RL框架示例
# 依赖：
#   conda install -c conda-forge python=3.8 numpy opencv gym stable-baselines3 torch
#   （确保docker内有ffmpeg以支持视频写入）

import os
import cv2
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --------------------
# 地图生成器
# --------------------
class MapGenerator:
    def __init__(self, width=256, height=256, n_obstacles=5):
        self.width = width
        self.height = height
        self.n_obstacles = n_obstacles

    def generate(self):
        # 返回障碍列表及目标点
        obstacles = []
        for _ in range(self.n_obstacles):
            # 随机添加矩形障碍
            w, h = np.random.randint(20, 60, size=2)
            x = np.random.randint(0, self.width - w)
            y = np.random.randint(0, self.height - h)
            obstacles.append((x, y, w, h))
        # 目标点随机
        target = (np.random.randint(20, self.width-20), np.random.randint(20, self.height-20))
        return obstacles, target

# --------------------
# 2D渲染器
# --------------------
class SimpleRenderer:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height

    def render(self, agent_pos, obstacles, target):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # 绘制障碍物（红色）
        for (x, y, w, h) in obstacles:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), -1)
        # 绘制目标点（绿色小圆）
        cv2.circle(img, target, 5, (0, 255, 0), -1)
        # 绘制智能体（蓝色圆）
        pos = (int(agent_pos[0]), int(agent_pos[1]))
        cv2.circle(img, pos, 8, (255, 0, 0), -1)
        return img

# --------------------
# Gym环境定义
# --------------------
class DrivingEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self):
        super(DrivingEnv, self).__init__()
        # 地图参数
        self.map_w, self.map_h = 256, 256
        # 加速度动作：二维连续
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 观测：顶视图RGB图像
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.map_h, self.map_w, 3), dtype=np.uint8
        )
        # 组件初始化
        self.map_gen = MapGenerator(self.map_w, self.map_h, n_obstacles=8)
        self.renderer = SimpleRenderer(self.map_w, self.map_h)
        # 状态
        self.agent_pos = None
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.obstacles = None
        self.target = None
        self.step_count = 0
        self.max_steps = 500

    def reset(self):
        self.obstacles, self.target = self.map_gen.generate()
        # 智能体初始在地图中心
        self.agent_pos = np.array([self.map_w/2, self.map_h/2], dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        obs = self.renderer.render(self.agent_pos, self.obstacles, self.target)
        return obs

    def step(self, action):
        # action为[-1,1]范围内的加速度向量
        accel = np.clip(action, -1.0, 1.0) * 2.0  # 最大加速度2.0 px/frame^2
        # 更新速度和位置
        self.agent_vel += accel
        # 限制速度
        speed = np.linalg.norm(self.agent_vel)
        if speed > 5.0:
            self.agent_vel = self.agent_vel / speed * 5.0
        self.agent_pos += self.agent_vel
        self.step_count += 1

        # 计算奖励
        # 距离目标减少量奖励
        prev_dist = np.linalg.norm((self.agent_pos - self.agent_vel) - np.array(self.target))
        cur_dist = np.linalg.norm(self.agent_pos - np.array(self.target))
        reward = prev_dist - cur_dist
        # 碰撞检测
        done = False
        for (x, y, w, h) in self.obstacles:
            if x <= self.agent_pos[0] <= x+w and y <= self.agent_pos[1] <= y+h:
                reward -= 5.0
                done = True
                break
        # 达到目标
        if cur_dist < 10.0:
            reward += 10.0
            done = True
        # 超时
        if self.step_count >= self.max_steps:
            done = True

        obs = self.renderer.render(self.agent_pos, self.obstacles, self.target)
        info = {}
        return obs, float(reward), done, info

    def render(self, mode="rgb_array"):
        # 返回当前帧图像
        return self.renderer.render(self.agent_pos, self.obstacles, self.target)

# --------------------
# 主训练脚本
# --------------------
def main():
    # 创建多环境
    env = DummyVecEnv([lambda: DrivingEnv() for _ in range(4)])
    # 定义模型
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs",
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
    )
    # 训练
    total_timesteps = 100_000  # 初步测试100k步
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_driving")

    # 录制视频：运行几个测试episode
    os.makedirs("videos", exist_ok=True)
    video_path = os.path.join("videos", "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (256, 256))
    test_env = DrivingEnv()
    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = test_env.step(action)
        frame = test_env.render()
        out.write(frame)
    out.release()
    print(f"Saved test video to {video_path}")

if __name__ == "__main__":
    main()
