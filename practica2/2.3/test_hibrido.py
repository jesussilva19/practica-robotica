"""
Test combinado AE -> AR con cambio dinámico según la distancia al objetivo.
"""

import os
import pickle
import time
import numpy as np
import neat
from stable_baselines3 import PPO

from main_neat_avoid_23 import RoboboNEATAvoidEnv23  # entorno AE
from main_ppo import RoboboEnv # entorno AR (ajusta si está en otro archivo)

# ============================================================
# CONFIG
# ============================================================

GENOME_PATH = "C:\\Users\\jesus\\Desktop\\practica-robotica\\practica2\\2.3\\neat_logs_2.3.2\\definitivo\\models\\best_genome_extracted.pkl"
CONFIG_PATH = "practica2/2.3/config-feedforwardmod"
PPO_MODEL_PATH = "practica2/2.3/best_model.zip"

NUM_EPISODES = 3
MAX_STEPS = 300
DIST_THRESHOLD = 1000.0  # 🔧 Cambia este valor a base de pruebas
RENDER = False


# ============================================================
# FUNCIONES
# ============================================================

def load_neat_genome(path):
    with open(path, "rb") as f:
        genome = pickle.load(f)
    print(f"✅ Genoma NEAT cargado desde {path}")
    return genome


def load_ppo_model(path):
    model = PPO.load(path)
    print(f"✅ Modelo PPO cargado desde {path}")
    return model


def compute_distance(sim):

    blob_loc = sim.getObjectLocation('CYLINDERBALL')
    robobo_loc = sim.getRobotLocation(0)

    blob_loc = blob_loc['position']
    robobo_loc = robobo_loc['position']
    
    # Calcular distancia al objetivo
    distance = np.sqrt(
        (blob_loc['x'] - robobo_loc['x'])**2 +
        (blob_loc['y'] - robobo_loc['y'])**2 +
        (blob_loc['z'] - robobo_loc['z'])**2
    )
    return distance



def run_hybrid_episode(genome, neat_config, ppo_model, max_steps=50, render=False):
    """
    Ejecuta un episodio híbrido:
    - Empieza con AE (NEAT) usando RoboboNEATAvoidEnv23
    - Cuando la distancia < DIST_THRESHOLD → pausa 3s → cambia a AR (PPO) usando RoboboEnv
    """
    neat_net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    env = RoboboNEATAvoidEnv23(max_steps=max_steps)
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    mode = "AE"  # modo inicial
    done = False

    print("\n🚀 Iniciando episodio híbrido (AE → AR)")
    print(f"🔧 Umbral de cambio de modo: {DIST_THRESHOLD:.1f}")

    try:
        while not done and steps < max_steps:
            # Calcular distancia al objetivo
            dist = compute_distance(env.sim)

            # Cambio de modo si se cumple condición
            if mode == "AE" and dist < DIST_THRESHOLD:
                mode = "AR"
                print(f"\n🔁 Cambio de control AE ➜ AR (distancia = {dist:.1f})")
                print("⏸️ Cambiando al entorno PPO (sin reiniciar simulación)...")
                
                # Guardar referencias de la simulación actual
                current_sim = env.sim
                current_robobo = env.robobo
                
                # Cerrar entorno NEAT (sin cerrar conexiones)
                try:
                    # No cerrar las conexiones, solo limpiar el entorno
                    pass
                except:
                    pass
                
                # Crear entorno PPO usando las mismas conexiones
                env = RoboboEnv(max_steps=max_steps - steps)
                
                # IMPORTANTE: Reutilizar las conexiones existentes (no hacer reset)
                env.sim = current_sim
                env.robobo = current_robobo
                env.steps = steps  # Mantener contador de pasos
                
                # Obtener estado actual (sin reiniciar posición)
                obs = env._get_state()
                
                print("✅ Entorno PPO (RoboboEnv) activado - posición mantenida")
                print("⏸️ Pausa de 3 segundos...")
                time.sleep(3)
                print("▶️ Reanudando con PPO (AR)\n")

            # --- Acción según modo actual ---
            if mode == "AE":
                output = neat_net.activate(obs)
                action = int(np.argmax(output))
            else:
                action, _ = ppo_model.predict(obs, deterministic=True)

            # Ejecutar paso en entorno
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

            if render:
                env.render()

            # Depuración cada cierto tiempo
            if steps % 20 == 0:
                print(f"[Paso {steps}] Modo={mode} | Dist={dist:.1f} | Reward acum={total_reward:.1f}")

        if terminated:
            print("✅ Éxito - Objetivo alcanzado")
        elif truncated:
            print("⏱️ Fin de episodio (máx. pasos alcanzado)")
        else:
            print("❌ Terminado anticipadamente")

    finally:
        env.close()

    print(f"🎯 Recompensa total episodio: {total_reward:.2f}")
    return total_reward


# ============================================================
# MAIN
# ============================================================

def main():
    # Cargar modelos
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    genome = load_neat_genome(GENOME_PATH)
    ppo_model = load_ppo_model(PPO_MODEL_PATH)

    rewards = []
    for ep in range(NUM_EPISODES):
        print("\n" + "="*60)
        print(f"🎬 EPISODIO {ep+1}/{NUM_EPISODES}")
        print("="*60)
        total = run_hybrid_episode(
            genome, neat_config, ppo_model,
            max_steps=MAX_STEPS, render=RENDER
        )
        rewards.append(total)

    print("\n" + "="*60)
    print("📊 RESULTADOS FINALES")
    print("="*60)
    print(f"Recompensas: {rewards}")
    print(f"Promedio: {np.mean(rewards):.2f}")
    print(f"Máximo: {np.max(rewards):.2f}")
    print(f"Mínimo: {np.min(rewards):.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
