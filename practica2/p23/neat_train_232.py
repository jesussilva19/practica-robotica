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
log_dir = f"practica2/p23/neat_logs_2.3/{timestamp}/"
models_dir = f"{log_dir}models/"
graphs_dir = f"{log_dir}graphs/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

print(f" Directorio de logs: {log_dir}")

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
            
        print(f"Fitness: {total:.2f} | Acciones únicas: {unique_actions}/8")
            
    except Exception as e:
        print(f"Error evaluando genoma: {e}")
        total = -100.0
    finally:
        try:
            env.close()
        except:
            pass  # Si falla el close, continuar

    # Asegurar que siempre retorna un float válido
    if total is None or np.isnan(total) or np.isinf(total):
        total = -200.0
    
    return float(total)

# === Evaluación de todos los genomas de una generación ===
def eval_genomes(genomes, config):
    """
    Evalúa todos los genomas de una generación.
    Imprime el fitness de cada individuo justo tras su evaluación.
    """
    for i, (gid, genome) in enumerate(genomes, start=1):
        print(f"\nEvaluando individuo {i}/{len(genomes)} (ID {gid}) ...")
        try:
            genome.fitness = eval_genome(genome, config)
            # Asegurar que el fitness no sea None
            if genome.fitness is None:
                genome.fitness = -200.0
                print(f"Fitness None detectado, asignado: {genome.fitness:.2f}")
            else:
                print(f"Resultado final individuo {gid}: Fitness = {genome.fitness:.2f}")
        except Exception as e:
            print(f"Error crítico evaluando genoma {gid}: {e}")
            genome.fitness = -200.0  # Asignar fitness muy bajo en caso de error
        print("-" * 60)
    
    # VERIFICACIÓN FINAL: Asegurar que TODOS los genomas tienen fitness válido
    for gid, genome in genomes: 
        if genome.fitness is None:
            print(f" ALERTA: Genoma {gid} aún tiene fitness None. Asignando -200.0")
            genome.fitness = -200.0

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
    print(" Gráfica de aprendizaje guardada")

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
    
    loaded_genome_id = None  # Para trackear el genoma cargado
    
    # Si hay un genoma previo, añadirlo a la población inicial
    if previous_best_genome_path and os.path.exists(previous_best_genome_path):
        print(f"📥 Cargando mejor genoma anterior desde: {previous_best_genome_path}")
        try:
            with open(previous_best_genome_path, "rb") as f:
                best_genome = pickle.load(f)
            
            # Obtener un ID válido para el nuevo genoma
            genome_id = max(p.population.keys()) + 1
            loaded_genome_id = genome_id  # Guardar para identificarlo después
            
            # Crear una copia del genoma con el nuevo ID
            new_genome = config.genome_type(genome_id)
            new_genome.configure_crossover(best_genome, best_genome, config.genome_config)
            new_genome.nodes = best_genome.nodes
            new_genome.connections = best_genome.connections
            
            # IMPORTANTE: Evaluar inmediatamente para evitar fitness None
            print(f"\n{'='*60}")
            print(f"EVALUANDO GENOMA CARGADO (ID: {genome_id})...")
            print(f"{'='*60}")
            new_genome.fitness = eval_genome(new_genome, config)
            print(f"{'='*60}")
            print(f"GENOMA ANTERIOR EVALUADO: Fitness = {new_genome.fitness:.2f}")
            print(f"{'='*60}\n")
            
            # Añadir a la población (no reemplazar)
            p.population[genome_id] = new_genome
            
            print(f"Mejor genoma anterior añadido a la población inicial")
            print(f"   Tamaño población actual: {len(p.population)} genomas\n")
        except Exception as e:
            print(f"Error al cargar genoma anterior: {e}")
            print("Continuando sin genoma previo...")
            import traceback
            traceback.print_exc()
    

    print("\nEvaluando población inicial completa...")
    initial_genomes = list(p.population.items())
    for gid, genome in initial_genomes:
        if genome.fitness is None:
            # Identificar si es el genoma cargado
            if gid == loaded_genome_id:
                print(f"Genoma {gid} (CARGADO) ya evaluado - saltando")
                continue
            genome.fitness = eval_genome(genome, config)
            print(f"Genoma {gid} evaluado: Fitness = {genome.fitness:.2f}")
    print("Población inicial completamente evaluada\n")
    
    if loaded_genome_id is not None:
        print("Re-especiando población con genoma cargado...")
        p.species.speciate(config, p.population, p.generation)
        print("Especiación actualizada\n")
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f"{models_dir}neat-checkpoint-"))

    print("\nIniciando evolución con NEAT (Práctica 2.3)...")
    print(f"Generaciones: {generations}")
    print(f"Tamaño población inicial: {config.pop_size}")
    print(f"Elitism: {config.reproduction_config.elitism}")
    print(f"Especies elitism: {config.stagnation_config.species_elitism}")

    winner = p.run(eval_genomes, generations)

    with open(f"{models_dir}best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nEvolución completada!")
    print(f"Mejor fitness alcanzado: {winner.fitness:.2f}")

    with open(f"{log_dir}stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    plot_stats(stats)
    return winner, config, stats

# === MAIN ===
if __name__ == "__main__":
  
    config_path = "practica2/p23/config-feedforwardmod"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encuentra config: {config_path}")

    previous_best = "practica2/p23/neat_logs_2.3.2/20251106_160516/models/best_genome_extracted.pkl"
    
    run(config_path, generations=10, previous_best_genome_path=previous_best)