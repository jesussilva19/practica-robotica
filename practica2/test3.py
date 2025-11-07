"""
Test combinado AE -> AR con cambio dinámico según la distancia al objetivo.
"""

import os
import pickle
import time
import numpy as np
import neat
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from p23.main_neat_avoid_232 import RoboboNEATAvoidEnv23  # entorno AE
from p23.main_ppo import RoboboEnv # entorno AR

# ============================================================
# CONFIG
# ============================================================

GENOME_PATH = os.path.join(os.path.dirname(__file__), 'p23', 'neat', 'models', 'best_genome_extracted.pkl')

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'p23', 'config-feedforward')
PPO_MODEL_PATH = os.path.join(os.path.dirname(__file__),'p23', 'best_model.zip')

NUM_EPISODES = 3
MAX_STEPS = 300
DIST_THRESHOLD = 1000.0  # Cambia este valor a base de pruebas
RENDER = False

output_path = os.path.join(os.path.dirname(__file__), 'test_results_2.3')
background_image = os.path.join(os.path.dirname(__file__), 'test_results_2.3', 'entorno3.png')  # Puedes especificar una ruta aquí si tienes imagen


# ============================================================
# FUNCIONES
# ============================================================

def plot_hybrid_trajectories(all_trajectories, goal_position=None, 
                             background_image=None, xlim=(-1500, 1500), ylim=(-1500, 1500)):
    """
    Grafica las trayectorias híbridas mostrando los cambios de modo AE→AR.
    
    Args:
        all_trajectories: Lista de diccionarios con trayectorias por episodio
        goal_position: Tupla (x, z) con la posición del objetivo
        background_image: Ruta a imagen de fondo (opcional)
        xlim: Tupla (min, max) para límites del eje X
        ylim: Tupla (min, max) para límites del eje Y
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Establecer límites fijos
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Cargar imagen de fondo si se proporciona
    if background_image is not None and os.path.exists(background_image):
        img = plt.imread(background_image)
        ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
                 aspect='auto', alpha=0.6, zorder=0)
        print(f"✓ Imagen de fondo cargada: {background_image}")
    else:
        # Fondo por defecto (gris claro)
        ax.add_patch(Rectangle((xlim[0], ylim[0]), 
                               xlim[1]-xlim[0], ylim[1]-ylim[0],
                               facecolor='#f0f0f0', zorder=0))
    
    episode_colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Graficar cada episodio
    for ep_idx, traj_data in enumerate(all_trajectories):
        color = episode_colors[ep_idx % len(episode_colors)]
        
        # Información del episodio
        ae_trajectory = traj_data['ae_trajectory']
        ar_trajectory = traj_data.get('ar_trajectory', None)
        success = traj_data['success']
        switch_point = traj_data.get('switch_point', None)
        
        label_base = f"Episodio {ep_idx+1}"
        if success:
            label_base += " ✓"
        else:
            label_base += " ✗"
        
        # --- Graficar fase AE (NEAT) ---
        if ae_trajectory['x']:
            x_ae = np.clip(ae_trajectory['x'], xlim[0], xlim[1])
            y_ae = np.clip(ae_trajectory['y'], ylim[0], ylim[1])
            
            ax.plot(x_ae, y_ae,
                   color=color,
                   linestyle='-',
                   linewidth=2.5,
                   alpha=0.8,
                   label=f"{label_base} (AE)")
            
            # Punto inicial
            if ep_idx == 0:
                ax.plot(x_ae[0], y_ae[0],
                       marker='o',
                       color='green',
                       markersize=14,
                       markeredgecolor='darkgreen',
                       markeredgewidth=2,
                       label='🟢 Inicio',
                       zorder=50)
            else:
                ax.plot(x_ae[0], y_ae[0],
                       marker='o',
                       color='green',
                       markersize=14,
                       markeredgecolor='darkgreen',
                       markeredgewidth=2,
                       zorder=50)
        
        # --- Graficar fase AR (PPO) ---
        if ar_trajectory and ar_trajectory['x']:
            x_ar = np.clip(ar_trajectory['x'], xlim[0], xlim[1])
            y_ar = np.clip(ar_trajectory['y'], ylim[0], ylim[1])
            
            ax.plot(x_ar, y_ar,
                   color=color,
                   linestyle='--',  # Línea punteada para diferenciar
                   linewidth=2.5,
                   alpha=0.8,
                   label=f"{label_base} (AR)")
            
            # Punto de cambio AE→AR
            if switch_point:
                if ep_idx == 0:
                    ax.plot(switch_point[0], switch_point[1],
                           marker='X',
                           color='orange',
                           markersize=16,
                           markeredgecolor='darkorange',
                           markeredgewidth=2,
                           label='Cambio AE→AR',
                           zorder=60)
                else:
                    ax.plot(switch_point[0], switch_point[1],
                           marker='X',
                           color='orange',
                           markersize=16,
                           markeredgecolor='darkorange',
                           markeredgewidth=2,
                           zorder=60)
            
            # Punto final
            end_color = 'gold' if success else 'gray'
            if ep_idx == 0:
                ax.plot(x_ar[-1], y_ar[-1],
                       marker='*',
                       color=end_color,
                       markersize=22,
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       label='⭐ Fin',
                       zorder=55)
            else:
                ax.plot(x_ar[-1], y_ar[-1],
                       marker='*',
                       color=end_color,
                       markersize=22,
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       zorder=55)
        else:
            # Si no hay fase AR, marcar el final de AE
            if ae_trajectory['x']:
                end_color = 'gold' if success else 'gray'
                ax.plot(x_ae[-1], y_ae[-1],
                       marker='*',
                       color=end_color,
                       markersize=22,
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       zorder=55)
    
    # Marcar el objetivo
    if goal_position is not None:
        goal_x, goal_z = goal_position
        ax.plot(goal_x, goal_z,
               marker='D',
               color='red',
               markersize=22,
               markeredgecolor='darkred',
               markeredgewidth=3,
               label='Objetivo',
               zorder=100)
        
        # Círculo de área del objetivo
        circle = plt.Circle((goal_x, goal_z), 150,
                           color='red',
                           fill=False,
                           linestyle=':',
                           linewidth=2.5,
                           alpha=0.6,
                           label='Área objetivo')
        ax.add_patch(circle)
    
    # Configuración del gráfico
    ax.set_xlabel('Posición X (mm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Posición Z (mm)', fontsize=13, fontweight='bold')
    ax.set_title('Trayectorias Híbridas (AE→AR) del Robot Robobo', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')
    
    # Guardar
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMapa de trayectorias híbridas guardado en: {output_path}")
    plt.close()
    
    # Estadísticas
    print("\nEstadísticas de Trayectorias:")
    for i, traj_data in enumerate(all_trajectories):
        ae_dist = calculate_path_length(traj_data['ae_trajectory']['x'], 
                                        traj_data['ae_trajectory']['y'])
        ar_dist = 0.0
        if traj_data.get('ar_trajectory') and traj_data['ar_trajectory']['x']:
            ar_dist = calculate_path_length(traj_data['ar_trajectory']['x'],
                                           traj_data['ar_trajectory']['y'])
        total_dist = ae_dist + ar_dist
        print(f"   Episodio {i+1}:")
        print(f"      - Fase AE: {ae_dist:.2f} mm")
        print(f"      - Fase AR: {ar_dist:.2f} mm")
        print(f"      - Total: {total_dist:.2f} mm")


def calculate_path_length(x_coords, y_coords):
    """
    Calcula la longitud total del camino recorrido.
    """
    if not x_coords or len(x_coords) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        total_distance += np.sqrt(dx**2 + dy**2)
    return total_distance


def load_neat_genome(path):
    with open(path, "rb") as f:
        genome = pickle.load(f)
    print(f"Genoma NEAT cargado desde {path}")
    return genome


def load_ppo_model(path):
    model = PPO.load(path)
    print(f"Modelo PPO cargado desde {path}")
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
    
    Retorna: dict con información del episodio y trayectorias
    """
    neat_net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    env = RoboboNEATAvoidEnv23(max_steps=max_steps)
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    mode = "AE"  # modo inicial
    done = False
    
    # Registrar trayectorias
    ae_trajectory = {'x': [], 'y': []}
    ar_trajectory = {'x': [], 'y': []}
    switch_point = None
    goal_position = None

    print("\nIniciando episodio híbrido (AE → AR)")
    print(f"Umbral de cambio de modo: {DIST_THRESHOLD:.1f}")

    try:
        # Obtener posición del objetivo (solo una vez)
        goal_loc = env.sim.getObjectLocation('CYLINDERBALL')
        if goal_loc:
            goal_position = (goal_loc['position']['x'], goal_loc['position']['z'])
        
        while not done and steps < max_steps:
            # Registrar posición actual
            robot_loc = env.sim.getRobotLocation(0)
            if mode == "AE":
                ae_trajectory['x'].append(robot_loc['position']['x'])
                ae_trajectory['y'].append(robot_loc['position']['z'])
            else:
                ar_trajectory['x'].append(robot_loc['position']['x'])
                ar_trajectory['y'].append(robot_loc['position']['z'])
            
            # Calcular distancia al objetivo
            dist = compute_distance(env.sim)

            # Cambio de modo si se cumple condición
            if mode == "AE" and dist < DIST_THRESHOLD:
                mode = "AR"
                
                # Guardar punto de cambio
                switch_point = (robot_loc['position']['x'], robot_loc['position']['z'])
                
                print(f"\nCambio de control AE ➜ AR (distancia = {dist:.1f})")
                print("⏸Cambiando al entorno PPO (sin reiniciar simulación)...")
                
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
                
                print("Entorno PPO (RoboboEnv) activado - posición mantenida")
                print("Pausa de 3 segundos...")
                time.sleep(3)
                print("Reanudando con PPO (AR)\n")

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
            print("Éxito - Objetivo alcanzado")
        elif truncated:
            print("Fin de episodio (máx. pasos alcanzado)")
        else:
            print("Terminado anticipadamente")

    finally:
        # Registrar última posición
        try:
            robot_loc = env.sim.getRobotLocation(0)
            if mode == "AE":
                ae_trajectory['x'].append(robot_loc['position']['x'])
                ae_trajectory['y'].append(robot_loc['position']['z'])
            else:
                ar_trajectory['x'].append(robot_loc['position']['x'])
                ar_trajectory['y'].append(robot_loc['position']['z'])
        except:
            pass
        
        env.close()

    print(f"Recompensa total episodio: {total_reward:.2f}")
    
    # Retornar información completa del episodio
    return {
        'total_reward': total_reward,
        'success': terminated,
        'steps': steps,
        'ae_trajectory': ae_trajectory,
        'ar_trajectory': ar_trajectory if ar_trajectory['x'] else None,
        'switch_point': switch_point,
        'goal_position': goal_position
    }


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
    all_trajectories = []
    goal_position = None
    
    for ep in range(NUM_EPISODES):
        print("\n" + "="*60)
        print(f"EPISODIO {ep+1}/{NUM_EPISODES}")
        print("="*60)
        
        episode_data = run_hybrid_episode(
            genome, neat_config, ppo_model,
            max_steps=MAX_STEPS, render=RENDER
        )
        
        rewards.append(episode_data['total_reward'])
        all_trajectories.append(episode_data)
        
        # Guardar goal_position del primer episodio
        if ep == 0 and episode_data['goal_position']:
            goal_position = episode_data['goal_position']

    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"Recompensas: {rewards}")
    print(f"Promedio: {np.mean(rewards):.2f}")
    print(f"Máximo: {np.max(rewards):.2f}")
    print(f"Mínimo: {np.min(rewards):.2f}")
    print("="*60)
    
    # Generar gráfica de trayectorias
    print("\nGenerando mapa de trayectorias híbridas...")
    plot_hybrid_trajectories(
        all_trajectories,
        goal_position=goal_position,
        background_image=background_image,
        xlim=(-1500, 1500),
        ylim=(-1500, 1500)
    )
    print("Proceso completado")


if __name__ == "__main__":
    main()
