# neat_train_232_continue.py - Continuar desde el mejor genoma anterior
import neat
import pickle
import os
import sys

# Importar la función run del archivo principal
from neat_train_232 import run

if __name__ == "__main__":
    config_path = "practica2/2.3/config-feedforwardmod"
    
    # Especifica la ruta al mejor genoma anterior
    if len(sys.argv) > 1:
        previous_best = sys.argv[1]
    else:
        # Buscar automáticamente el más reciente
        logs_dir = "practica2/2.3/neat_logs_2.3.2"
        subdirs = sorted([d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))], reverse=True)
        
        if subdirs:
            most_recent = subdirs[0]
            previous_best = f"{logs_dir}/{most_recent}/models/best_genome.pkl"
            print(f"🔍 Usando el genoma más reciente: {previous_best}")
        else:
            print("❌ No se encontró ningún genoma anterior")
            previous_best = None
    
    if previous_best and os.path.exists(previous_best):
        print(f"✅ Continuando desde: {previous_best}")
        run(config_path, generations=12, previous_best_genome_path=previous_best)
    else:
        print(f"⚠️ No se encontró el archivo: {previous_best}")
        print("Iniciando entrenamiento desde cero...")
        run(config_path, generations=12)
