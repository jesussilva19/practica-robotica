import gymnasium as gym
from main import RoboboEnv

env = RoboboEnv()
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
