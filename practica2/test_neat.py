import pickle
import neat
import time
from main_neat import RoboboNEATEnv

def run_best_genome(genome_path, config_path, n_episodes=3, render=True):
    # 1. cargar config (tiene que ser la misma con la que entrenaste)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # 2. cargar mejor genoma
    with open(genome_path, "rb") as f:
        best_genome = pickle.load(f)

    # 3. crear red a partir del genoma
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # 4. crear entorno
    env = RoboboNEATEnv(max_steps=150)

    for ep in range(n_episodes):
        print(f"\n========== EPISODIO {ep+1}/{n_episodes} ==========")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < env.max_steps:
            # 5. pasar observación por la red
            output = net.activate(obs)

            # 6. elegir acción (la de mayor activación)
            action = max(range(len(output)), key=lambda i: output[i])

            # 7. ejecutar acción en el entorno
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()

            done = terminated or truncated

            # opcional: para que se vea en el simulador
            time.sleep(0.05)

        print(f"➡️ Episodio terminado en {steps} pasos. Recompensa: {total_reward:.2f}")

    env.close()
    print("✅ Test finalizado")


if __name__ == "__main__":
    # RUTAS: cámbialas según la fecha de tu log
    genome_path = "./neat_logs_2.1/20251102_000801/models/best_genome.pkl"
    config_path = "./practica2/config-feedforward.txt"   # o donde lo tengas

    run_best_genome(genome_path, config_path, n_episodes=3, render=True)
