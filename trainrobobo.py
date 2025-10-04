from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
from main import RoboboEnv  # importa tu entorno

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = RoboboEnv()
env = Monitor(env, log_dir)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    device="cpu"  # PPO MLP va mejor en CPU
)

model.learn(total_timesteps=20000)
model.save(log_dir + "ppo_robobo")
print("✅ Entrenamiento finalizado")
env.close()


