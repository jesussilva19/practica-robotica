import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os


from prueba import RoboboEnv  


log_dir = "./robobo_logs/"
os.makedirs(log_dir, exist_ok=True)

# Crear entorno y monitorizar
env = RoboboEnv()
env = Monitor(env, log_dir)

# Definir modelo PPO
model = PPO(
    policy="MlpPolicy",   # red neuronal multi-layer perceptron
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

# Callback para evaluación durante entrenamiento
eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=5000,   
    deterministic=True,
    render=False,
)

# Entrenar el modelo
TIMESTEPS = 50000 
model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)

# Guardar modelo entrenado
model.save(log_dir + "ppo_robobo")
print("✅ Modelo entrenado y guardado en", log_dir)
