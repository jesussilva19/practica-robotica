"""
Script de prueba para modelos NEAT de la práctica 2.
Uso: python test.py
"""

import sys
import os

# Importar módulo de pruebas y entorno
import neat_test
from p22.main_neat import RoboboNEATEnv

genome_path = os.path.join(os.path.dirname(__file__), 'p22', 'neat', 'models', 'best_genome.pkl')
num_episodes = 3
config_path = os.path.join(os.path.dirname(__file__), 'p22', 'config-feedforward')
output_folder = os.path.join(os.path.dirname(__file__), 'test_results_2.2')
background_image = os.path.join(os.path.dirname(__file__), 'test_results_2.2', 'entorno2.png')  # Puedes especificar una ruta aquí si tienes imagen


def main():
    """
    Función principal para probar genomas de la práctica 2.2
    """
    # Crear carpeta de resultados si no existe
    os.makedirs(output_folder, exist_ok=True)
        
    print(f"\n{'='*70}")
    print(f"  PRUEBA DE GENOMA - PRÁCTICA 2.2")
    print(f"{'='*70}")
    print(f"Genoma: {genome_path}")
    print(f"Config: {config_path}")
    print(f"Episodios: {num_episodes}")
    print(f"Resultados: {output_folder}")
    print(f"{'='*70}\n")
    
    neat_test.test_genome_simple(
        genome_path, 
        config_path, 
        RoboboNEATEnv, 
        num_episodes, 
        output_folder, 
        background_image, 
        (-1500, 1500), 
        (-1500, 1500))


    print(f"Test completado. Resultados guardados en: {output_folder}")


if __name__ == '__main__':
    main()
