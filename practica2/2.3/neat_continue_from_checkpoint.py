# neat_continue_from_checkpoint.py
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
        print(f"\n🚀 Evaluando individuo {i}/{len(genomes)} (ID {gid}) ...")
        try:
            genome.fitness = eval_genome(genome, config)
            # Asegurar que el fitness no sea None
            if genome.fitness is None:
                genome.fitness = -200.0
                print(f"⚠️ Fitness None detectado, asignado: {genome.fitness:.2f}")
            else:
                print(f"✅ Resultado final individuo {gid}: Fitness = {genome.fitness:.2f}")
        except Exception as e:
            print(f"❌ Error crítico evaluando genoma {gid}: {e}")
            genome.fitness = -200.0  # Asignar fitness muy bajo en caso de error
        print("-" * 60)
    
    # VERIFICACIÓN FINAL: Asegurar que TODOS los genomas tienen fitness válido
    for gid, genome in genomes:
        if genome.fitness is None:
            print(f"⚠️ ALERTA: Genoma {gid} aún tiene fitness None. Asignando -200.0")
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
    plt.title('Evolución del Fitness - Práctica 2.3 (Continuación)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}aprendizaje.png", dpi=300)
    plt.close()
    print("📈 Gráfica de aprendizaje guardada")

# === Ejecución principal del algoritmo NEAT ===
def run(config_file, generations=10, checkpoint_path=None):
    """
    Ejecuta el entrenamiento con NEAT.
    
    Args:
        config_file: Ruta al archivo de configuración NEAT
        generations: Número de generaciones adicionales a entrenar
        checkpoint_path: Ruta a un checkpoint previo para continuar entrenamiento
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    
    # Cargar desde checkpoint o crear población nueva
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n{'='*70}")
        print(f"📥 RESTAURANDO DESDE CHECKPOINT")
        print(f"{'='*70}")
        print(f"Archivo: {checkpoint_path}")
        
        try:
            p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
            
            print(f"✅ Checkpoint cargado exitosamente")
            print(f"   📊 Generación inicial: {p.generation}")
            print(f"   👥 Tamaño población: {len(p.population)}")
            print(f"   🧬 Número de especies: {len(p.species.species)}")
            
            # Mostrar estadísticas del mejor genoma actual
            best_genome = None
            best_fitness = float('-inf')
            for gid, genome in p.population.items():
                if genome.fitness is not None and genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome
            
            if best_genome:
                print(f"   🏆 Mejor fitness actual: {best_fitness:.2f}")
                print(f"   🧠 Tamaño red mejor: ({len(best_genome.nodes)} nodos, {len(best_genome.connections)} conexiones)")
            
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"❌ Error al cargar checkpoint: {e}")
            print("Creando nueva población desde cero...")
            import traceback
            traceback.print_exc()
            p = neat.Population(config)
            
    else:
        if checkpoint_path:
            print(f"⚠️ Checkpoint no encontrado: {checkpoint_path}")
        print(f"\n🆕 Creando nueva población desde cero...")
        p = neat.Population(config)
    
    # Añadir reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f"{models_dir}neat-checkpoint-"))

    print("\n🧠 Iniciando evolución con NEAT (Práctica 2.3)...")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📍 Continuando desde generación: {p.generation}")
        print(f"➕ Generaciones adicionales: {generations}")
        print(f"🎯 Generación final esperada: {p.generation + generations}")
    else:
        print(f"🆕 Entrenamiento nuevo desde generación 0")
        print(f"📊 Generaciones totales: {generations}")
    
    print(f"👥 Tamaño población: {config.pop_size}")
    print(f"🏅 Elitism: {config.reproduction_config.elitism}")
    print(f"🧬 Especies elitism: {config.stagnation_config.species_elitism}")
    print(f"{'='*70}\n")
    
    # Ejecutar evolución
    winner = p.run(eval_genomes, generations)

    # Guardar mejor genoma
    with open(f"{models_dir}best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\n" + "="*70)
    print("🎉 EVOLUCIÓN COMPLETADA")
    print("="*70)
    print(f"🏆 Mejor fitness alcanzado: {winner.fitness:.2f}")
    print(f"📊 Generación final: {p.generation}")
    print(f"🧠 Tamaño red ganadora:")
    print(f"   - Nodos: {len(winner.nodes)}")
    print(f"   - Conexiones: {len(winner.connections)}")
    print(f"📁 Mejor genoma guardado en: {models_dir}best_genome.pkl")
    print("="*70 + "\n")

    # Guardar estadísticas
    with open(f"{log_dir}stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    plot_stats(stats)
    
    return winner, config, stats


# === MAIN ===
if __name__ == "__main__":
    config_path = "practica2/2.3/config-feedforwardmod"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encuentra config: {config_path}")
    
    # ================================================================
    # CONFIGURACIÓN: Elige una de las siguientes opciones
    # ================================================================
    
    # OPCIÓN 1: Continuar desde un checkpoint específico (RECOMENDADO)
    checkpoint = "practica2/2.3/neat_logs_2.3.2/20251106_160516/models/neat-checkpoint-1"
    
    # OPCIÓN 2: Buscar el checkpoint más reciente automáticamente
    # checkpoint = "practica2/2.3/neat_logs_2.3.2/20251106_160516/models/neat-checkpoint-5"
    
    # OPCIÓN 3: Empezar desde cero (nueva población)
    # checkpoint = None
    
    # ================================================================
    # EJECUTAR ENTRENAMIENTO
    # ================================================================
    run(config_path, generations=10, checkpoint_path=checkpoint)
