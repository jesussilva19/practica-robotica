"""
Script para extraer el mejor genoma de un checkpoint de NEAT.
"""
import neat
import pickle
import sys
import os

def extract_best_genome(checkpoint_path, output_path=None):
    """
    Extrae el mejor genoma de un checkpoint de NEAT.
    
    Args:
        checkpoint_path: Ruta al archivo de checkpoint
        output_path: Ruta donde guardar el mejor genoma (opcional)
    """
    print(f"📂 Cargando checkpoint: {checkpoint_path}")
    
    try:
        # Intentar cargar el checkpoint (puede estar comprimido con gzip)
        import gzip
        
        # Primero intentar con gzip
        try:
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print("✅ Checkpoint descomprimido (gzip)")
        except:
            # Si falla, intentar sin comprimir
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print("✅ Checkpoint cargado (sin comprimir)")
        
        # El checkpoint contiene: generation, config, population, species_set, rndstate
        generation = checkpoint[0]
        config = checkpoint[1]
        population = checkpoint[2]
        
        print(f"✅ Checkpoint cargado exitosamente")
        print(f"   Generación: {generation}")
        print(f"   Individuos en población: {len(population)}")
        
        # Encontrar el mejor genoma
        best_genome_id = None
        best_fitness = float('-inf')
        
        for genome_id, genome in population.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome_id = genome_id
                best_genome = genome
        
        if best_genome_id is None:
            print("❌ No se encontró ningún genoma con fitness válido")
            return None
        
        print(f"\n🏆 Mejor genoma encontrado:")
        print(f"   ID: {best_genome_id}")
        print(f"   Fitness: {best_fitness:.2f}")
        
        # Guardar el mejor genoma
        if output_path is None:
            # Crear nombre automático en el mismo directorio
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_path = os.path.join(checkpoint_dir, "best_genome_extracted.pkl")
        
        with open(output_path, 'wb') as f:
            pickle.dump(best_genome, f)
        
        print(f"\n💾 Mejor genoma guardado en: {output_path}")
        return best_genome, best_fitness
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {checkpoint_path}")
        return None
    except Exception as e:
        print(f"❌ Error al procesar el checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Usar ruta por defecto o desde argumentos
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Ruta por defecto
        checkpoint_path = "practica2/2.3/neat_logs_2.3.2/20251106_160516/models/neat-checkpoint-1"
    
    # Ruta de salida opcional
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*60)
    print("EXTRACTOR DE MEJOR GENOMA DESDE CHECKPOINT")
    print("="*60)
    
    result = extract_best_genome(checkpoint_path, output_path)
    
    if result:
        print("\n✅ Proceso completado exitosamente")
    else:
        print("\n❌ No se pudo extraer el genoma")
