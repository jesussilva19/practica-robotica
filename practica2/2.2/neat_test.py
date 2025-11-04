"""
Script SIMPLE para probar modelos NEAT entrenados
Uso básico: python neat_test.py path/al/genoma.pkl
"""

import neat
import pickle
import gzip
import numpy as np
from main_neat import RoboboNEATEnv


def load_genome(genome_path):
    """
    Carga un genoma desde archivo pickle o checkpoint comprimido.
    """
    # Intentar cargar como pickle normal (best_genome.pkl)
    try:
        with open(genome_path, 'rb') as f:
            genome = pickle.load(f)
        print("Genoma cargado (pickle)")
        return genome
 
            
    except Exception as e:
        print(f"Error cargando archivo: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        raise


def test_genome_simple(genome_path, num_episodes=3):
    """
    Prueba un genoma de forma sencilla.
    """
    print(f"\n{'='*60}")
    print(f"PROBANDO GENOMA")
    print(f"{'='*60}\n")
    
    # 1. Cargar configuración NEAT
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        './practica2/2.2/config-feedforward'
    )
    
    # 2. Cargar genoma
    genome = load_genome(genome_path)
    
    if hasattr(genome, 'fitness') and genome.fitness is not None:
        print(f"Fitness: {genome.fitness:.2f}")
    else:
        print(f" Fitness no disponible")
    
    # 3. Crear red neuronal
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 4. Probar en episodios
    total_rewards = []
    successes = 0
    
    for episode in range(num_episodes):
        print(f"\n--- Episodio {episode + 1}/{num_episodes} ---")
        
        # Crear entorno
        env = RoboboNEATEnv(max_steps=75)
        
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        # Ejecutar episodio
        while not done and steps < 200:
            # Red decide acción
            output = net.activate(obs)
            action = np.argmax(output)
            
            # Ejecutar
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        # Guardar resultados
        total_rewards.append(total_reward)
        if terminated:
            successes += 1
            print("ÉXITO - Objetivo alcanzado")
        else:
            print("FALLO - No alcanzó objetivo")
        
        print(f"   Reward: {total_reward:.2f}")
        print(f"   Steps: {steps}")
        
        env.close()
    
    # 5. Resumen final
    print(f"\n{'='*60}")
    print(f"RESUMEN")
    print(f"{'='*60}")
    print(f"Éxitos: {successes}/{num_episodes}")
    print(f"Reward promedio: {np.mean(total_rewards):.2f}")
    print(f"Reward máximo: {max(total_rewards):.2f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python neat_test.py <ruta_al_genoma.pkl>")
        print("\nEjemplo:")
        print("  python neat_test.py neat_logs_2.1/ultimo/models/best_genome.pkl")
        sys.exit(1)
    
    genome_path = sys.argv[1]
    
    # Número de episodios (opcional)
    num_episodes = 3
    if len(sys.argv) > 2:
        try:
            num_episodes = int(sys.argv[2])
        except:
            print("Número de episodios inválido, usando 3")
    
    test_genome_simple(genome_path, num_episodes)
