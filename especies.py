import pickle
import matplotlib.pyplot as plt

# Ruta al archivo stats.pkl generado tras el entrenamiento
STATS_PATH = "C:\\Users\\jesus\\Desktop\\practica-robotica\\practica2\\p23\\neat_logs_2.3.2\\definitivo\\stats.pkl"
OUTPUT_PATH = "C:\\Users\\jesus\\Desktop\\practica-robotica\\practica2\\p23\\neat_logs_2.3.2\\definitivo\\graphs\\species_post.png"

# Cargar las estadísticas
with open(STATS_PATH, "rb") as f:
    stats = pickle.load(f)

# Extraer tamaños de especies por generación
species_sizes = stats.get_species_sizes()
num_generations = len(species_sizes)
num_species = max(len(gen) for gen in species_sizes)

# Crear la gráfica
plt.figure(figsize=(10, 6))
for i in range(num_species):
    sizes = [gen[i] if i < len(gen) else 0 for gen in species_sizes]
    plt.plot(range(num_generations), sizes, label=f"Especie {i+1}")

plt.xlabel("Generación")
plt.ylabel("Número de individuos")
plt.title("Evolución de las especies (post-entrenamiento)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"📊 Gráfica de especies generada en: {OUTPUT_PATH}")
