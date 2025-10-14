from main import RoboboEnv  # si tu clase está en main.py
import time

env = RoboboEnv()

# Resetear entorno
obs, info = env.reset()
print("Estado inicial:", obs)

done = False
while not done:
    # elegir una acción aleatoria
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Acción={action}, Obs={obs}, Recompensa={reward:.2f}")
    env.render()
    print(terminated, truncated)
    done = terminated or truncated
    time.sleep(1)  # para no saturar el simulador

env.close()
