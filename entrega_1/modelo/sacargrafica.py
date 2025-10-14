import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV (omite la primera línea con el diccionario JSON)
df = pd.read_csv("C:\\Users\\jesus\\Desktop\\practica-robotica\\robobo_logs\\finalultimo4\\monitor.csv", comment="#")

# Número de episodios = índice
df["episode"] = range(1, len(df)+1)

# Gráfica de recompensa por episodio
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["r"], marker="o", label="Recompensa")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total (r)")
plt.title("Evolución de la recompensa por episodio")
plt.legend()
plt.grid(True)
plt.show()

# Gráfica de longitud por episodio
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["l"], marker="o", color="orange", label="Longitud")
plt.xlabel("Episodio")
plt.ylabel("Longitud del episodio (steps)")
plt.title("Evolución de la longitud de episodios")
plt.legend()
plt.grid(True)
plt.show()
