"""
Script de prueba para modelos NEAT de la práctica 2.1
Uso: python test1.py <ruta_al_genoma.pkl> [num_episodios]

Ejemplo:
  python test1.py 2.1/neat_logs_2.1/ultimo/models/best_genome.pkl 5
"""

import sys
import os


# Importar módulo de pruebas y entorno
import neat_test
from p21.main_neat import RoboboNEATEnv

genome_path = os.path.join(os.path.dirname(__file__), 'p21', 'neat_logs_2.1', 'ultimo', 'models', 'best_genome.pkl')
num_episodes = 3
config_path = os.path.join(os.path.dirname(__file__), 'p21', 'config-feedforward')
output_folder = os.path.join(os.path.dirname(__file__), 'test_results_2.1')
background_image = os.path.join(os.path.dirname(__file__), 'p21', 'entorno1.png')  # Puedes especificar una ruta aquí si tienes imagen


def main():
    """
    Función principal para probar genomas de la práctica 2.1
    """
    # Crear carpeta de resultados si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Opcional: imagen de fondo del escenario 2.1
    
    print(f"\n{'='*70}")
    print(f"  PRUEBA DE GENOMA - PRÁCTICA 2.1")
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
        (-1000, 1000), 
        (-1000, 1000))


    print(f"Test completado. Resultados guardados en: {output_folder}")


if __name__ == '__main__':
    main()
