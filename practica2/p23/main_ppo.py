import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim

class RoboboEnv(gym.Env):
    """
    Entorno de Gymnasium para robot Robobo que debe encontrar y acercarse a un objetivo rojo.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=50, host="localhost"):
        super(RoboboEnv, self).__init__()
        
        # Conexión con el robot
        self.robobo = Robobo(host)
        self.sim = RoboboSim(host) 
        self.robobo.connect()
        self.sim.connect()
        
        # Configuración inicial de la cámara
        self.robobo.moveTiltTo(200, 50)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        # Espacios de observación y acción
        self.observation_space = spaces.Discrete(14)  # 14 estados según posición del objetivo
        self.action_space = spaces.Discrete(6)       # 6 acciones de movimiento

        # Variables de estado
        self.state = None
        self.steps = 0
        self.max_steps = max_steps
        
        # Constantes para detección
        self.OBSTACLE_THRESHOLD_FRONT = 30
        self.OBSTACLE_THRESHOLD_SIDE = 300
        self.BLOB_SIZE_MIN = 2
        self.BLOB_SIZE_GOAL = 10
        self.CENTER_MIN = 45
        self.CENTER_MAX = 55
        
        # Posiciones del pan para búsqueda
        self.pan_positions = [0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60, -75,-90]

    def reset(self, *, seed=None):
        """Reinicia el entorno y retorna el estado inicial."""
        super().reset(seed=seed)
        self.steps = 0
        
        # Reiniciar simulación
        self.sim.resetSimulation()  
        self.robobo.wait(1.0)
        
        # Reconfigurar cámara
        self.robobo.moveTiltTo(200, 50)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)
        
        self.state = self._get_state()
        return self.state, {}


    def _is_at_goal(self):
        """
        Verifica si el robot ha alcanzado el objetivo.
        Retorna True si está en el objetivo, False en caso contrario.
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        distancia = self.robobo.readIRSensor(IR.FrontC)
        
        # Condiciones para considerar que llegó al objetivo:
        # 1. El blob es lo suficientemente grande
        # 2. La distancia es muy cercana
        at_goal = (blob.size > self.BLOB_SIZE_GOAL and  
                   distancia > self.OBSTACLE_THRESHOLD_FRONT)
        
        return at_goal
    
    def _avoid_obstacle(self):
        """
        Detecta obstáculos y realiza maniobra de evasión si es necesario.
        Retorna True si evadió un obstáculo, False en caso contrario.
        """
        # PRIMERO: Verificar si estamos en el objetivo
        if self._is_at_goal():
            print("En el objetivo - NO evadir")
            return False
        
        # Leer sensores IR
        front_c = self.robobo.readIRSensor(IR.FrontC)
        front_l = self.robobo.readIRSensor(IR.FrontL)
        front_r = self.robobo.readIRSensor(IR.FrontR)
        
        # Verificar si vemos el objetivo (pero no estamos en él)
        blob = self.robobo.readColorBlob(BlobColor.RED)
        veo_objetivo = blob.size > self.BLOB_SIZE_MIN
        
        # Si vemos el objetivo grande, asumimos que es el obstáculo detectado
        # y NO evadimos (queremos acercarnos)
        if veo_objetivo and blob.size > 5:
            print("Objetivo visible y centrado - Acercándose")
            return False
        
        # Si hay obstáculo y NO vemos bien el objetivo, evadir
        has_obstacle = (front_c > self.OBSTACLE_THRESHOLD_FRONT or 
                       front_l > self.OBSTACLE_THRESHOLD_SIDE or 
                       front_r > self.OBSTACLE_THRESHOLD_SIDE)
        
        if has_obstacle:
            print("Obstáculo detectado - Evadiendo")
            self.robobo.moveWheelsByTime(-20, -20, 1)  # Retroceder
            self.robobo.moveWheelsByTime(30, -30, 1)   # Girar
            self.robobo.wait(0.5)
            return True
        
        return False
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Acciones:
        0: Avanzar recto
        1: Girar izquierda (leve)
        2: Girar derecha (leve)
        3: Girar izquierda (fuerte)
        4: Girar derecha (fuerte)
        5: Girar 180 grados
        """
        self.steps += 1
        
        # Verificar y evitar obstáculos antes de la acción
        evaded = self._avoid_obstacle()
        
        # Ejecutar acción solo si no se evadió obstáculo
        if not evaded:
            if action == 0:  # Avanzar
                self.robobo.moveWheelsByTime(5, 5, 2)  
            elif action == 1:  # Girar izquierda leve
                self.robobo.moveWheelsByTime(0, 5, 2)
            elif action == 2:  # Girar derecha leve
                self.robobo.moveWheelsByTime(5, 0, 2)
            elif action == 3:  # Girar izquierda fuerte
                self.robobo.moveWheelsByTime(0, 5, 4)
            elif action == 4:  # Girar derecha fuerte
                self.robobo.moveWheelsByTime(5, 0, 4)
            elif action == 5:  # Giro 180°
                self.robobo.moveWheelsByTime(10, -10, 3)

        # Obtener nuevo estado
        self.state = self._get_state()

        reward = 0
        if self.state == 0: reward = 3*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [1,7]: reward = 1.5*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [2,8]: reward = 1*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [3,9]: reward = 0.6*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [4,10]: reward = 0.4*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [5,11]: reward = 0.2*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [6,12]: reward = 0.1*self.robobo.readColorBlob(BlobColor.RED).size
        else: reward = -5

        # Verificar si alcanzó el objetivo
        terminated = self._is_at_goal()
        
        if terminated:
            print(" ¡OBJETIVO ALCANZADO! ")
            reward += 200  # Gran recompensa por completar el objetivo
        
        # Verificar condiciones de terminación por tiempo
        truncated = self.steps >= self.max_steps
        
        # Obtener información para logging
        blob = self.robobo.readColorBlob(BlobColor.RED)
        distancia = self.robobo.readIRSensor(IR.FrontC)
        
        # Logging
        print(f"\n--- Step {self.steps} ---")
        print(f"Acción: {action}")
        print(f"Estado: {self.state}")
        print(f"Distancia IR: {distancia}")
        print(f"Tamaño Blob: {blob.size}")
        print(f"Blob Pos X: {blob.posx}")
        print(f"Recompensa: {reward:.2f}")
        
        if truncated:
            print("** Tiempo máximo alcanzado - Episodio truncado")

        return self.state, reward, terminated, truncated, {}



    def _get_state(self):
        """
        Determina el estado actual basado en la posición del objetivo rojo.
        
        Estados:
        0: Objetivo centrado (45-55)
        1: Objetivo centro-izquierda (25-45)
        2: Objetivo centro-derecha (55-75)
        3: Objetivo extremo izquierda (1-25)
        4: Objetivo extremo derecha (75+)
        5: Objetivo no visible
        """
        # Buscar objetivo moviendo la cámara pan
        for i, ang in enumerate(self.pan_positions):
            self.robobo.movePanTo(ang, 100, True)
            blobs = self.robobo.readColorBlob(BlobColor.RED)
            
            if blobs.size > 0:
                print(f"blobs.size: {blobs.size}")
                # Retornar a posición central después de encontrar
            
                return i
        
        # Si no se encuentra en ninguna posición
        return len(self.pan_positions)

    def render(self):
        """Muestra información del estado actual."""
        state_names = {
            0: "Centrado",
            1: "Centro-Izquierda", 
            2: "Centro-Derecha",
            3: "Extremo Izquierda",
            4: "Extremo Derecha",
            5: "No Visible"
        }
        print(f"Estado actual: {state_names.get(self.state, 'Desconocido')} ({self.state})")

    def close(self):
        """Cierra las conexiones con el robot."""
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
            print(" Conexiones cerradas correctamente")
        except Exception as e:
            print(f"Error al cerrar conexiones: {e}")