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
        print("✅ Genoma cargado (pickle)")
        return genome
    except:
        pass
    
    # Intentar cargar como checkpoint comprimido de NEAT
    try:
        with gzip.open(genome_path, 'rb') as f:
            # Los checkpoints pueden tener diferente número de objetos
            # Intentamos cargar todos los que haya
            objects = []
            try:
                while True:
                    obj = pickle.load(f)
                    objects.append(obj)
            except EOFError:
                pass  # Normal, llegamos al final del archivo
            
            print(f"✅ Checkpoint cargado ({len(objects)} objetos)")
            
            # Debug: mostrar tipo de cada objeto
            for i, obj in enumerate(objects):
                print(f"   Objeto {i}: {type(obj).__name__}")
            
            # El checkpoint puede ser un objeto Checkpointer que contiene todo
            if len(objects) == 1:
                checkpoint_obj = objects[0]
                
                # Si es un objeto con atributos, intentar acceder a population
                if hasattr(checkpoint_obj, 'population'):
                    population = checkpoint_obj.population
                    generation = getattr(checkpoint_obj, 'generation', 'desconocida')
                    print(f"   Generación: {generation}")
                    print(f"   Población: {len(population)} genomas")
                # Si el objeto es directamente un genoma
                elif hasattr(checkpoint_obj, 'fitness'):
                    print("   Es un genoma individual")
                    return checkpoint_obj
                # Si es una tupla (generation, population, species, rndstate)
                elif isinstance(checkpoint_obj, tuple):
                    print(f"   Es una tupla con {len(checkpoint_obj)} elementos")
                    # Buscar el diccionario de población
                    for elem in checkpoint_obj:
                        if isinstance(elem, dict):
                            population = elem
                            break
                    else:
                        raise ValueError(f"No se encontró población. Tupla: {[type(e).__name__ for e in checkpoint_obj]}")
                else:
                    raise ValueError(f"Formato de checkpoint desconocido: {type(checkpoint_obj).__name__}")
            else:
                # Múltiples objetos: buscar generación y población
                population = None
                generation = None
                
                for i, obj in enumerate(objects):
                    if isinstance(obj, int):
                        generation = obj
                        print(f"   Generación: {generation}")
                    elif isinstance(obj, dict) and population is None:
                        population = obj
                        print(f"   Población: {len(obj)} genomas")
                
                if population is None:
                    raise ValueError("No se encontró población en los objetos múltiples")
            
            # Obtener el mejor genoma
            genome = max(population.values(), key=lambda g: g.fitness if g.fitness else -float('inf'))
            return genome
            
    except Exception as e:
        print(f"❌ Error cargando archivo: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        raise


def test_genome_simple(genome_path, num_episodes=3):
    """
    Prueba un genoma de forma sencilla.
    """
    print(f"\n{'='*60}")
    print(f"🧪 PROBANDO GENOMA")
    print(f"{'='*60}\n")
    
    # 1. Cargar configuración NEAT
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        './practica2/config-feedforward'
    )
    
    # 2. Cargar genoma (maneja .pkl y checkpoints comprimidos)
    genome = load_genome(genome_path)
    
    if hasattr(genome, 'fitness') and genome.fitness is not None:
        print(f"📊 Fitness: {genome.fitness:.2f}")
    else:
        print(f"⚠️  Fitness no disponible")
    
    # 3. Crear red neuronal
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 4. Probar en episodios
    total_rewards = []
    successes = 0
    
    for episode in range(num_episodes):
        print(f"\n--- Episodio {episode + 1}/{num_episodes} ---")
        
        # Crear entorno
        env = RoboboNEATEnv(max_steps=200)
        
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
            print("✅ ÉXITO - Objetivo alcanzado")
        else:
            print("❌ FALLO - No alcanzó objetivo")
        
        print(f"   Reward: {total_reward:.2f}")
        print(f"   Steps: {steps}")
        
        env.close()
    
    # 5. Resumen final
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN")
    print(f"{'='*60}")
    print(f"Éxitos: {successes}/{num_episodes}")
    print(f"Reward promedio: {np.mean(total_rewards):.2f}")
    print(f"Reward máximo: {max(total_rewards):.2f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("❌ Uso: python neat_test.py <ruta_al_genoma.pkl>")
        print("\nEjemplo:")
        print("  python neat_test.py neat_logs_2.1/20251101_153518/models/best_genome.pkl")
        sys.exit(1)
    
    genome_path = sys.argv[1]
    
    # Número de episodios (opcional)
    num_episodes = 3
    if len(sys.argv) > 2:
        try:
            num_episodes = int(sys.argv[2])
        except:
            print("⚠️ Número de episodios inválido, usando 3")
    
    test_genome_simple(genome_path, num_episodes)
