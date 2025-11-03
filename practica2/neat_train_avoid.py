import neat
import pickle
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from main_neat_avoid import RoboboNEATAvoidEnv

# Logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./neat_logs_2.2/{timestamp}/"
models_dir = f"{log_dir}models/"
graphs_dir = f"{log_dir}graphs/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

print(f"Directorio de logs: {log_dir}")

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = RoboboNEATAvoidEnv(max_steps=200)  # misma duración que 2.1
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    try:
        done = False
        while not done and steps < env.max_steps:
            output = net.activate(obs)
            action = int(np.argmax(output))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            steps += 1
    except Exception as e:
        print(f"Error evaluando genoma: {e}")
        total = -100.0
    finally:
        env.close()
    return float(total)

def eval_genomes(genomes, config):
    for gid, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def plot_stats(stats):
    gens = range(len(stats.most_fit_genomes))
    best = [g.fitness for g in stats.most_fit_genomes]
    avg = stats.get_fitness_mean()
    plt.figure(figsize=(10, 6))
    plt.plot(gens, best, label='Mejor Fitness', linewidth=2)
    plt.plot(gens, avg, '--', label='Fitness Promedio', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del Fitness - Práctica 2.2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}aprendizaje.png", dpi=300)
    plt.close()
    print("📈 Gráfica de aprendizaje guardada")

def run(config_file, generations=30):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f"{models_dir}neat-checkpoint-"))

    print("\n🚀 Iniciando evolución con NEAT (2.2)...")
    print(f"Generaciones: {generations}")
    print(f"Tamaño población: {config.pop_size}")

    winner = p.run(eval_genomes, generations)

    with open(f"{models_dir}best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\n✅ Evolución completada!")
    print(f"🏆 Mejor fitness alcanzado: {winner.fitness:.2f}")

    with open(f"{log_dir}stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    plot_stats(stats)
    return winner, config, stats

if __name__ == "__main__":
    # Usa el MISMO config-feedforward.txt (5 in, 6 out, sin threshold corto)
    config_path = "practica2/2.1/config-feedforward"  # asegúrate de la extensión .txt
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró config: {config_path}")
    run(config_path, generations=10)
