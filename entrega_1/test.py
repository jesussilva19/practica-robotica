"""
Script para probar el modelo PPO entrenado con el robot Robobo.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from main import RoboboEnv
import sys

def test_model(model_path, n_episodes=3, render=True):
    """
    Prueba un modelo entrenado en el entorno Robobo.
    
    Args:
        model_path: Ruta al modelo guardado
        n_episodes: Número de episodios a ejecutar
        render: Si mostrar información durante la ejecución
    """
    print(f"Cargando modelo desde: {model_path}")
    
    try:
        # Cargar modelo
        model = PPO.load(model_path)
        
        # Crear entorno
        env = RoboboEnv()
        
        print(f"\nEjecutando {n_episodes} episodios de prueba...\n")
        
        total_rewards = []
        success_count = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            print(f"{'='*50}")
            print(f"Episodio {episode + 1}/{n_episodes}")
            print(f"{'='*50}")
            
            while not done:
                # Predecir acción usando el modelo
                action, _states = model.predict(obs, deterministic=True)
                
                # Ejecutar acción
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if render:
                    env.render()
                
                done = terminated or truncated
                
                if terminated:
                    print("Episodio completado exitosamente")
                    success_count += 1
                elif truncated:
                    print("Episodio truncado (tiempo máximo)")
            
            total_rewards.append(episode_reward)
            
            print(f"\nResultados del episodio:")
            print(f"  - Pasos: {step_count}")
            print(f"  - Recompensa total: {episode_reward:.2f}")
            print()
        
        # Estadísticas finales
        print(f"\n{'='*50}")
        print("ESTADÍSTICAS FINALES")
        print(f"{'='*50}")
        print(f"Episodios completados: {n_episodes}")
        print(f"Éxitos: {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
        print(f"Recompensa promedio: {sum(total_rewards)/len(total_rewards):.2f}")
        print(f"Recompensa máxima: {max(total_rewards):.2f}")
        print(f"Recompensa mínima: {min(total_rewards):.2f}")
        
        env.close()
        
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {model_path}")
        print("Verifica que la ruta sea correcta y que el archivo exista")
    except Exception as e:
        print(f"Error al cargar o ejecutar el modelo: {e}")

def main():
    """Función principal."""
    # Cambiar esta ruta al modelo que se quiera probar
    model_path = "C:\\Users\\jesus\\Desktop\\practica-robotica\\robobo_logs\\finalultimo4\\ppo_robobo_final.zip"
    
    # Si se proporciona ruta como argumento
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Número de episodios de prueba
    n_episodes = 5
    
    test_model(model_path, n_episodes=n_episodes, render=True)

if __name__ == "__main__":
    main()