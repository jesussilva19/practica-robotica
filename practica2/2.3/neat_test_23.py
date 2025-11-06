"""
Test automático del mejor genoma (Práctica 2.3 NEAT - Robobo)
Ejecuta directamente sin argumentos.
"""

import os
import pickle
import numpy as np
import neat
from main_neat_avoid_23 import RoboboNEATAvoidEnv23


# ============================================================
# CONFIGURACIÓN AUTOMÁTICA
# ============================================================

GENOME_PATH = "C:\\Users\\jesus\\Desktop\\practica-robotica\\practica2\\2.3\\neat_logs_2.3.2\\20251106_185011\\models\\best_genome_extracted.pkl"
CONFIG_PATH = "practica2/2.3/config-feedforwardmod"
NUM_EPISODES = 3
MAX_STEPS = 200
RENDER = False  # pon True si quieres ver el render del entorno


# ============================================================
# FUNCIONES DE CARGA Y PRUEBA
# ============================================================

def load_genome(genome_path):
    """Carga el genoma entrenado desde un archivo .pkl"""
    if not os.path.exists(genome_path):
        raise FileNotFoundError(f"No se encontró el genoma en: {genome_path}")
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    print(f"✅ Genoma cargado: {genome_path}")
    return genome


def run_episodes(genome, config, num_episodes=3, max_steps=30, render=False):
    """Ejecuta varios episodios en el entorno RoboboNEATAvoidEnv23"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_rewards = []
    successes = 0

    for ep in range(num_episodes):
        print(f"\n--- Episodio {ep + 1}/{num_episodes} ---")
        env = RoboboNEATAvoidEnv23(max_steps=max_steps)
        obs, _ = env.reset()

        total = 0.0
        steps = 0
        terminated = False
        truncated = False

        try:
            done = False
            while not done and steps < max_steps:
                output = net.activate(obs)
                action = int(np.argmax(output))
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                steps += 1

                if render:
                    env.render()

                done = terminated or truncated

        finally:
            env.close()

        total_rewards.append(total)
        if terminated and not truncated:
            successes += 1
            print("✅ ÉXITO - Objetivo alcanzado")
        elif truncated:
            print("⏱️ TRUNCADO - Límite de pasos alcanzado")
        else:
            print("❌ FALLÓ - No alcanzó el objetivo")

        print(f"   Recompensa total: {total:.2f}")
        print(f"   Pasos: {steps}")

    return successes, total_rewards


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🔍 TEST AUTOMÁTICO DEL GENOMA ENTRENADO (PRÁCTICA 2.3)")
    print("=" * 60)

    # 1️⃣ Cargar configuración NEAT
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {CONFIG_PATH}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    # 2️⃣ Cargar genoma entrenado
    genome = load_genome(GENOME_PATH)
    if getattr(genome, "fitness", None) is not None:
        print(f"ℹ️ Fitness guardado: {genome.fitness:.2f}")

    # 3️⃣ Ejecutar pruebas
    successes, rewards = run_episodes(
        genome,
        config,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        render=RENDER
    )

    # 4️⃣ Resumen final
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DEL TEST")
    print("=" * 60)
    print(f"Éxitos: {successes}/{NUM_EPISODES}")
    print(f"Reward promedio: {np.mean(rewards):.2f}")
    print(f"Reward máximo: {np.max(rewards):.2f}")
    print(f"Reward mínimo: {np.min(rewards):.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
