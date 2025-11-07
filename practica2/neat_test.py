"""
Módulo reutilizable para probar modelos NEAT entrenados.
Puede ser importado desde test1.py, test2.py, etc.
"""

import neat
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys


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


def plot_trajectories(trajectories, genome_path, goal_position=None, 
                     background_image=None, xlim=(-1500, 1500), ylim=(-1500, 1500),
                     output_folder=None):
    """
    Grafica las trayectorias del robot en un mapa 2D.
    
    Args:
        trajectories: Lista de diccionarios con 'x', 'y', 'success'
        genome_path: Ruta del genoma (para nombre del archivo)
        goal_position: Tupla (x, z) con la posición del objetivo
        background_image: Ruta a imagen de fondo (opcional)
        xlim: Tupla (min, max) para límites del eje X
        ylim: Tupla (min, max) para límites del eje Y
        output_folder: Carpeta donde guardar el gráfico
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Establecer límites fijos
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Cargar imagen de fondo si se proporciona
    if background_image is not None and os.path.exists(background_image):
        img = plt.imread(background_image)
        ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
                 aspect='auto', alpha=0.6, zorder=0)
        print(f"Imagen de fondo cargada: {background_image}")
    else:
        # Fondo por defecto (gris claro)
        ax.add_patch(Rectangle((xlim[0], ylim[0]), 
                               xlim[1]-xlim[0], ylim[1]-ylim[0],
                               facecolor='#f0f0f0', zorder=0))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    # Graficar cada trayectoria
    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label = f"Episodio {i+1}"
        
        if traj['success']:
            label += " ✓"
            linestyle = '-'
            alpha = 0.8
        else:
            label += " ✗"
            linestyle = '--'
            alpha = 0.5
        
        # Clipear trayectoria a los límites
        x_clipped = np.clip(traj['x'], xlim[0], xlim[1])
        y_clipped = np.clip(traj['y'], ylim[0], ylim[1])
        
        # Línea de trayectoria
        ax.plot(x_clipped, y_clipped, 
                color=color, 
                linestyle=linestyle,
                linewidth=2,
                alpha=alpha,
                label=label)
        
        # Punto inicial (círculo verde)
        ax.plot(x_clipped[0], y_clipped[0], 
                marker='o', 
                color='green', 
                markersize=12,
                markeredgecolor='darkgreen',
                markeredgewidth=2,
                label='Inicio' if i == 0 else '')
        
        # Punto final (estrella)
        end_color = 'gold' if traj['success'] else 'gray'
        ax.plot(x_clipped[-1], y_clipped[-1], 
                marker='*', 
                color=end_color, 
                markersize=20,
                markeredgecolor='black',
                markeredgewidth=1.5,
                label='Fin' if i == 0 else '')
        
        # Marcadores a lo largo de la trayectoria (cada N pasos)
        step_interval = max(1, len(x_clipped) // 10)
        for j in range(0, len(x_clipped), step_interval):
            ax.plot(x_clipped[j], y_clipped[j], 
                    marker=marker, 
                    color=color, 
                    markersize=4,
                    alpha=0.3)
    
    # Marcar el objetivo (cilindro rojo)
    if goal_position is not None:
        goal_x, goal_z = goal_position
        ax.plot(goal_x, goal_z, 
                marker='D',  # Diamante
                color='red', 
                markersize=20,
                markeredgecolor='darkred',
                markeredgewidth=3,
                label='Objetivo (Cilindro Rojo)',
                zorder=100)  # Asegurar que esté encima
        
        # Círculo de área del objetivo
        circle = plt.Circle((goal_x, goal_z), 150, 
                           color='red', 
                           fill=False, 
                           linestyle='--', 
                           linewidth=2, 
                           alpha=0.5,
                           label='Área objetivo')
        ax.add_patch(circle)
    
    # Configuración del gráfico
    ax.set_xlabel('Posición X (mm)', fontsize=12)
    ax.set_ylabel('Posición Z (mm)', fontsize=12)
    ax.set_title('Trayectorias del Robot Robobo', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    # Guardar
    if output_folder is None:
        output_folder = os.path.dirname(genome_path)
    
    output_name = os.path.basename(genome_path).replace('.pkl', '_trayectorias.png')
    output_path = os.path.join(output_folder, output_name)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMapa de trayectorias guardado en: {output_path}")
    plt.close()
    
    # También mostrar estadísticas de distancias
    print("\nEstadísticas de Distancias:")
    for i, traj in enumerate(trajectories):
        distance = calculate_path_length(traj['x'], traj['y'])
        print(f"   Episodio {i+1}: {distance:.2f} m")


def calculate_path_length(x_coords, y_coords):
    """
    Calcula la longitud total del camino recorrido.
    """
    total_distance = 0.0
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        total_distance += np.sqrt(dx**2 + dy**2)
    return total_distance


def test_genome_simple(genome_path, config_path, env_class, num_episodes, 
                       output_folder, background_image, 
                       xlim, ylim):
    """
    Prueba un genoma de forma sencilla.
    
    Args:
        genome_path: Ruta al archivo del genoma (.pkl)
        config_path: Ruta al archivo de configuración NEAT
        env_class: Clase del entorno (RoboboNEATEnv importada del main_neat correspondiente)
        num_episodes: Número de episodios de prueba
        output_folder: Carpeta donde guardar las gráficas (si None, usa el directorio del genoma)
        background_image: Ruta a imagen de fondo para las trayectorias
        xlim: Límites del eje X para el gráfico
        ylim: Límites del eje Y para el gráfico
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
        config_path
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
    all_trajectories = []  # Guardar trayectorias para graficar
    
    for episode in range(num_episodes):
        print(f"\n--- Episodio {episode + 1}/{num_episodes} ---")
        
        # Crear entorno con más steps para test
        env = env_class(max_steps=50)  # Más tiempo en test
        
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        # Registrar trayectoria
        trajectory = {
            'x': [],
            'y': [],
            'success': False
        }
        
        # Ejecutar episodio
        while not done and steps < env.max_steps:
            # Registrar posición actual del robot
            robot_location = env.sim.getRobotLocation(0)
            trajectory['x'].append(robot_location['position']['x'])
            trajectory['y'].append(robot_location['position']['z'])
            
            # Red decide acción
            output = net.activate(obs)
            action = np.argmax(output)
            
            # Ejecutar
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        # Guardar última posición
        robot_location = env.sim.getRobotLocation(0)
        trajectory['x'].append(robot_location['position']['x'])
        trajectory['y'].append(robot_location['position']['z'])
        trajectory['success'] = terminated
        
        all_trajectories.append(trajectory)
        
        # Obtener posición del objetivo (solo en el primer episodio)
        if episode == 0:
            try:
                goal_location = env.sim.getObjectLocation('CYLINDERMIDBALL')
                goal_position = (goal_location['position']['x'], goal_location['position']['z'])
            except:
                goal_location = env.sim.getObjectLocation('CYLINDERBALL')
                goal_position = (goal_location['position']['x'], goal_location['position']['z'])

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
    
    # 6. Graficar trayectorias
    if output_folder is None:
        output_folder = os.path.dirname(genome_path)
    
    plot_trajectories(all_trajectories, genome_path, goal_position,
                     background_image=background_image,
                     xlim=xlim,
                     ylim=ylim,
                     output_folder=output_folder)
    
    return all_trajectories


if __name__ == '__main__':
    # Este bloque solo se ejecuta si se llama directamente al script
    # (no cuando se importa como módulo)
    print("Este módulo está diseñado para ser importado desde test1.py o test2.py")
    print("\nEjemplo de uso desde test1.py:")
    print("  from neat_test import test_genome_simple")
    print("  from main_neat import RoboboNEATEnv")
    print("  test_genome_simple(genome_path, config_path, RoboboNEATEnv)")
    
    if len(sys.argv) >= 2:
        print("\nSi quieres ejecutar directamente, usa test1.py o test2.py")

