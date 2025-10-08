import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime
from main3 import RoboboEnv

# Configuración de directorios
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./robobo_logs/{timestamp}/"
models_dir = f"{log_dir}models/"
tensorboard_dir = f"{log_dir}tensorboard/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

print(f"Directorio de logs: {log_dir}")

# Crear entorno y monitorizar
base_env = RoboboEnv()
env = Monitor(base_env, log_dir)

# Configuración del modelo PPO con hiperparámetros optimizados
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=128,          # Aumentado para mejor exploración
    batch_size=64,
    n_epochs=10,
    gamma=0.99,            # Factor de descuento
    gae_lambda=0.95,       # Ventaja generalizada
    clip_range=0.2,        # Clipping para estabilidad
    ent_coef=0.01,         # Coeficiente de entropía para exploración
    vf_coef=0.5,           # Coeficiente de value function
    max_grad_norm=0.5,     # Gradient clipping
    verbose=1,
    tensorboard_log=tensorboard_dir,
)

# Callback para guardar checkpoints periódicos
checkpoint_callback = CheckpointCallback(
    save_freq=3605,  # Guardar cada 500 pasos
    save_path=models_dir,
    name_prefix="ppo_robobo_checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# Callback para evaluación durante entrenamiento
eval_callback = EvalCallback(
    env,
    best_model_save_path=models_dir,
    log_path=log_dir,
    eval_freq=500,         # Evaluar cada 500 pasos
    n_eval_episodes=5,     # Evaluar con 5 episodios
    deterministic=True,
    render=False,
)

# Entrenar el modelo
TOTAL_TIMESTEPS = 2500  # Aumentado significativamente

print("\nIniciando entrenamiento...")
print(f"Total de timesteps: {TOTAL_TIMESTEPS}")
print(f"Pasos por episodio: {base_env.max_steps}")
print(f"Episodios aproximados: {TOTAL_TIMESTEPS // base_env.max_steps}")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        log_interval=10,
    )
    
    # Guardar modelo final
    final_model_path = f"{models_dir}ppo_robobo_final"
    model.save(final_model_path)
    print(f"\nModelo entrenado y guardado en {final_model_path}")
    
    # Guardar también en formato .zip
    model.save(f"{log_dir}ppo_robobo_final.zip")
    
    print("\nInformación del entrenamiento:")
    print(f"   - Logs: {log_dir}")
    print(f"   - Modelos: {models_dir}")
    print(f"   - TensorBoard: tensorboard --logdir={tensorboard_dir}")
    
except KeyboardInterrupt:
    print("\nEntrenamiento interrumpido por el usuario")
    model.save(f"{models_dir}ppo_robobo_interrupted")
    print(f"💾 Modelo guardado en {models_dir}ppo_robobo_interrupted")
    
except Exception as e:
    print(f"\nError durante el entrenamiento: {e}")
    model.save(f"{models_dir}ppo_robobo_error")
    print(f"Modelo guardado en {models_dir}ppo_robobo_error")
    
finally:
    env.close()
    print("\nEntorno cerrado")