# neat_train_23.py
import neat
import pickle
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from main_neat_avoid_232 import RoboboNEATAvoidEnv23

# === Configuración de directorios ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"practica2/2.3/neat_logs_2.3.2/{timestamp}/"
models_dir = f"{log_dir}models/"
graphs_dir = f"{log_dir}graphs/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

print(f"📂 Directorio de logs: {log_dir}")

# === Evaluación de un genoma individual ===
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = RoboboNEATAvoidEnv23(max_steps=45)
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    
    # Trackear diversidad de acciones
    action_counts = np.zeros(8)
    last_action = -1
    repeated_actions = 0

    try:
        done = False
        while not done and steps < env.max_steps:
            output = net.activate(obs)
            
            # Añadir pequeño ruido para fomentar exploración
            output = np.array(output) + np.random.normal(0, 0.1, len(output))
            
            action = int(np.argmax(output))
            action_counts[action] += 1
            
            # Penalizar repetición excesiva de la misma acción
            if action == last_action:
                repeated_actions += 1
                if repeated_actions > 5:
                    total -= 2.0  # Penalización fuerte
            else:
                repeated_actions = 0
            
            last_action = action
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            steps += 1
            
            # Terminar anticipadamente si el fitness es muy malo
            if total < -185:
                print(f"  Individuo eliminado por fitness muy bajo ({total:.2f})")
                env.close()
                return -200.0  # Fitness de penalización
            
        # Bonificación por usar variedad de acciones
        unique_actions = np.count_nonzero(action_counts)
        diversity_bonus = unique_actions * 3.0
        total += diversity_bonus
        
        # Penalización severa si usa solo 1-2 acciones
        if unique_actions <= 2:
            total -= 30.0
            
        print(f"   ✅ Fitness: {total:.2f} | Acciones únicas: {unique_actions}/8")
            
    except Exception as e:
        print(f"❌ Error evaluando genoma: {e}")
        total = -100.0
    finally:
        env.close()

    return float(total)

# === Evaluación de todos los genomas de una generación ===
def eval_genomes(genomes, config):
    """
    Evalúa todos los genomas de una generación.
    Imprime el fitness de cada individuo justo tras su evaluación.
    """
    for i, (gid, genome) in enumerate(genomes, start=1):
        print(f"\n🚀 Evaluando individuo {i}/{len(genomes)} (ID {gid}) ...")
        genome.fitness = eval_genome(genome, config)
        print(f"✅ Resultado final individuo {gid}: Fitness = {genome.fitness:.2f}")
        print("-" * 60)

# === Gráfica de evolución ===
def plot_stats(stats):
    gens = range(len(stats.most_fit_genomes))
    best = [g.fitness for g in stats.most_fit_genomes]
    avg = stats.get_fitness_mean()

    plt.figure(figsize=(10, 6))
    plt.plot(gens, best, label='Mejor Fitness', linewidth=2)
    plt.plot(gens, avg, '--', label='Fitness Promedio', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del Fitness - Práctica 2.3')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}aprendizaje.png", dpi=300)
    plt.close()
    print("📈 Gráfica de aprendizaje guardada")

# === Ejecución principal del algoritmo NEAT ===
def run(config_file, generations=10, previous_best_genome_path=None):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)
    
    # Si hay un genoma previo, añadirlo a la población inicial
    if previous_best_genome_path and os.path.exists(previous_best_genome_path):
        print(f"📥 Cargando mejor genoma anterior desde: {previous_best_genome_path}")
        with open(previous_best_genome_path, "rb") as f:
            best_genome = pickle.load(f)
        
        # Añadir el mejor genoma a la población inicial
        # Reemplazar un individuo aleatorio con el mejor anterior
        genome_id = list(p.population.keys())[0]
        p.population[genome_id] = best_genome
        print(f"✅ Mejor genoma anterior añadido a la población inicial (ID: {genome_id})")
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f"{models_dir}neat-checkpoint-"))

    print("\n🧠 Iniciando evolución con NEAT (Práctica 2.3)...")
    print(f"Generaciones: {generations}")
    print(f"Tamaño población inicial: {config.pop_size}")
    print(f"Elitism: {config.reproduction_config.elitism}")
    print(f"Especies elitism: {config.stagnation_config.species_elitism}")

    winner = p.run(eval_genomes, generations)

    with open(f"{models_dir}best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\n✅ Evolución completada!")
    print(f"🏆 Mejor fitness alcanzado: {winner.fitness:.2f}")

    with open(f"{log_dir}stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    plot_stats(stats)
    return winner, config, stats

# === MAIN ===
if __name__ == "__main__":
    # Ajusta si tu config tiene otra ruta/nombre
    config_path = "practica2/2.3/config-feedforwardmod"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encuentra config: {config_path}")

    # OPCIONAL: Ruta al mejor genoma de una ejecución anterior
    # Descomenta y ajusta la ruta si quieres usar un genoma previo
    previous_best = "practica2/2.3/neat_logs_2.3.2/20251106_160516/models/best_genome_extracted.pkl"
    # previous_best = "practica2/2.3/neat_logs_2.3.2/20251106_123456/models/best_genome.pkl"
    
    run(config_path, generations=10, previous_best_genome_path=previous_best)
