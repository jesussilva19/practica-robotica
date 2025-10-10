"""
Script para probar un modelo PPO entrenado con el Robobo y visualizar la trayectoria en 2D.
"""

import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from main5 import RoboboEnv   # importa tu entorno


def plot_trajectory(env, episode_num):
    if not env.trajectory:
        print("⚠️ No hay posiciones registradas")
        return

    xs, ys = zip(*env.trajectory)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o", label="Trayectoria Robobo")
    plt.scatter(xs[0], ys[0], c="green", s=100, label="Inicio")
    plt.scatter(xs[-1], ys[-1], c="red", s=100, label="Fin")

    if env.cylinder_positions:
        cx, cy = zip(*env.cylinder_positions)
        plt.scatter(cx, cy, c="blue", marker="x", label="Cilindro")

    plt.title(f"Episodio {episode_num} - Trayectoria")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_model(model_path, n_episodes=3, render=True):
    print(f"Cargando modelo desde: {model_path}")
    model = PPO.load(model_path)
    env = RoboboEnv()

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if render:
                env.render()

        print(f"\nEpisodio {episode+1} finalizado. Pasos: {steps}, Recompensa: {total_reward:.2f}")
        plot_trajectory(env, episode+1)

    env.close()


def main():
    model_path = "C:\\Users\\jesus\\Desktop\\practica-robotica\\robobo_logs\\finalultimo4\\ppo_robobo_final.zip"

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    test_model(model_path, n_episodes=3, render=True)


if __name__ == "__main__":
    main()
