# draw_network.py
import pickle
import neat
from visualize import draw_net  

# Rutas: ajusta a tu caso
CONFIG_PATH = "practica2/p21/config-feedforward"
GENOME_PATH = "C://Users//jesus/Desktop/practica-robotica/practica2/p21/neat_logs_2.1/ultimo/models/best_genome.pkl"


# 1) Cargar config NEAT
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH
)


with open(GENOME_PATH, "rb") as f:
    genome = pickle.load(f)


node_names = {
   
    -1: "blob_x", -2: "blob_size", -3: "IR_FC", -4: "IR_FL", -5: "IR_FR",
    #
     0: "Avanzar", 1: "GirarIzq_L", 2: "GirarDer_L",
     3: "GirarIzq_F", 4: "GirarDer_F", 5: "Giro180",
     
}

# 4) Dibujar (genera 'red_neat.svg' en el directorio actual)
draw_net(
    config=config,
    genome=genome,
    view=False,                 # True para abrir tras generar
    filename="red_neat",        # sin extensión; se usará .svg por defecto
    node_names=node_names,
    show_disabled=True,         # muestra conexiones deshabilitadas en punteado
    prune_unused=False,          # elimina nodos que no afectan a la salida
    fmt="png"                   # "svg" (nítido) o "png"
)
print("✅ Red guardada como red_neat.png")
