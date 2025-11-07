import neat
import pickle
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from main_neat import RoboboNEATEnv

# Configuración de directorios
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"practica2/2.2/neat_logs_2.2/{timestamp}/"
models_dir = f"{log_dir}models/"
graphs_dir = f"{log_dir}graphs/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

print(f"Directorio de logs: {log_dir}")

# Variables globales para seguimiento
best_genome_ever = None
best_fitness_ever = -float('inf')

def eval_genome(genome, config):
    """
    Evalúa un genoma individual ejecutándolo en el entorno.
    """
    # Crear red neuronal desde el genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Crear entorno con más pasos para permitir llegar al objetivo
    env = RoboboNEATEnv(max_steps=30)  # Aumentado de 30 a 50
    
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    
    try:
        while not done and steps < env.max_steps:
            # La red toma las observaciones y produce salidas
            output = net.activate(obs)
            
            # Elegir la acción con mayor activación
            action = np.argmax(output)
            
            # Ejecutar acción
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            done = terminated or truncated
            steps += 1
            
    except Exception as e:
        print(f"Error evaluando genoma: {e}")
        total_reward = -100  # Penalización por error
    
    finally:
        env.close()
    
    return total_reward


def eval_genomes(genomes, config):
    """
    Evalúa todos los genomas de una generación.
    Esta función es requerida por NEAT.
    """
    global best_genome_ever, best_fitness_ever
    
    # Evaluar cada genoma
    for i, (genome_id, genome) in enumerate(genomes, 1):
        fitness = eval_genome(genome, config)
        genome.fitness = fitness
        
        # Mostrar fitness de cada genoma
        print(f"Genoma {i}/{len(genomes)} (ID: {genome_id}): Fitness = {fitness:.2f}")
        
        # Actualizar mejor genoma
        if fitness > best_fitness_ever:
            best_fitness_ever = fitness
            best_genome_ever = genome
            print(f"🏆 ¡Nuevo mejor fitness: {fitness:.2f}!")


def run_neat(config_file, generations=30):
    """
    Ejecuta el algoritmo NEAT.
    """
    # Cargar configuración
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    
    # Crear población
    p = neat.Population(config)
    
    # Añadir reportes
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f'{models_dir}neat-checkpoint-'))
    
    # Ejecutar evolución
    print("\nIniciando evolución con NEAT...")
    print(f"Generaciones: {generations}")
    print(f"Tamaño población: {config.pop_size}")
    
    winner = p.run(eval_genomes, generations)
    
    # Guardar mejor genoma
    with open(f'{models_dir}best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    print(f"\nEvolución completada!")
    print(f"Mejor fitness alcanzado: {winner.fitness:.2f}")
    
    # Guardar estadísticas
    with open(f'{log_dir}stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    # Generar gráficas
    plot_stats(stats, winner, config)
    
    return winner, config, stats


def plot_stats(stats, winner, config):
    """
    Genera las gráficas requeridas para la memoria.
    """
    # 1. Gráfica de aprendizaje (fitness a lo largo de generaciones)
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'b-', label='Mejor Fitness', linewidth=2)
    plt.plot(generation, avg_fitness, 'r--', label='Fitness Promedio', linewidth=2)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolución del Fitness - Práctica 2.1', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}aprendizaje.png', dpi=300)
    print(f"Gráfica de aprendizaje guardada")
    plt.close()
    
    # 3. Gráfica de especies
    try:
        from neat.graphs import plot_species
        plot_species(stats, filename=f'{graphs_dir}especies.png')
        print(f"Gráfica de especies guardada")
    except:
        print("No se pudo generar gráfica de especies")


if __name__ == '__main__':
    # Archivo de configuración
    config_path = 'practica2/2.2/config-feedforward'

    
    if not os.path.exists(config_path):
        print(f"Error: No se encuentra el archivo {config_path}")
        exit(1)
    
    try:
        # Ejecutar NEAT
        winner, config, stats = run_neat(config_path, generations=75)
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"Resultados guardados en: {log_dir}")
        print(f"Mejor genoma: {models_dir}best_genome.pkl")
        print(f"Gráficas en: {graphs_dir}")
        
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario")
        if best_genome_ever is not None:
            with open(f'{models_dir}interrupted_genome.pkl', 'wb') as f:
                pickle.dump(best_genome_ever, f)
            print(f"Mejor genoma guardado en: {models_dir}interrupted_genome.pkl")
    
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()