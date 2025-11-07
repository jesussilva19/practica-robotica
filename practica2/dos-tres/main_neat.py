import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim

class RoboboNEATEnv(gym.Env):
    """
    Entorno de Gymnasium para NEAT - Práctica 2.1
    El robot debe encontrar y acercarse al cilindro rojo inmóvil.
    Escenario: cylinder
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=200, host="localhost"):
        super(RoboboNEATEnv, self).__init__()
        
        # Conexión con el robot
        self.robobo = Robobo(host)
        self.sim = RoboboSim(host) 
        self.robobo.connect()
        self.sim.connect()
        
        # Configuración inicial de la cámara
        self.robobo.moveTiltTo(200, 70)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        # NEAT necesita entradas continuas (no estados discretos)
        # Entradas: [blob_x, blob_size, ir_front_c, ir_front_l, ir_front_r]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 6 acciones discretas
        self.action_space = spaces.Discrete(6)

        # Variables de estado
        self.state = None
        self.steps = 0
        self.max_steps = max_steps
        
        # Constantes para detección (valores SIN normalizar - usados en comparaciones internas)
        # Nota: El estado de la red neuronal SÍ está normalizado, pero estos thresholds
        # se comparan directamente con los valores del simulador
        self.OBSTACLE_THRESHOLD_FRONT = 40    # IR < 40 indica obstáculo cercano
        self.OBSTACLE_THRESHOLD_SIDE = 300    # IR lateral
        self.BLOB_SIZE_MIN = 2                # Tamaño mínimo para considerar que ve el blob
        self.BLOB_SIZE_GOAL = 300             # Tamaño para considerar objetivo alcanzado
        self.GOAL_DISTANCE_THRESHOLD = 40     # Distancia IR para confirmar llegada

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

    def _get_state(self):
        """
        Obtiene el estado como un array continuo para NEAT.
        Retorna: [blob_x_normalizado, blob_size_normalizado, ir_front_c, ir_front_l, ir_front_r]
        """
        # Leer blob rojo
        blob = self.robobo.readColorBlob(BlobColor.RED)
        
        # Leer sensores IR
        ir_front_c = self.robobo.readIRSensor(IR.FrontC)
        ir_front_l = self.robobo.readIRSensor(IR.FrontL)
        ir_front_r = self.robobo.readIRSensor(IR.FrontR)
        
        # Normalizar posición X del blob (0-100)
        blob_x = blob.posx if blob.size > 0 else 50.0  # Centro si no hay blob
        
        # Tamaño del blob
        blob_size = min(blob.size, 500.0)  # Limitar a 500
        

        # Normalizar todo a [0, 1]
        state = np.array([
            blob_x / 100.0,           # 0-100 -> 0-1
            blob_size / 500.0,         # 0-500 -> 0-1
            ir_front_c / 1000.0,       # 0-1000 -> 0-1
            ir_front_l / 1000.0,
            ir_front_r / 1000.0
        ], dtype=np.float32)
        
        return state

    def _is_at_goal(self):
        """
        Verifica si el robot ha alcanzado el objetivo.
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        distancia = self.robobo.readIRSensor(IR.FrontC)
        
        # Objetivo alcanzado si el blob es grande y está cerca
        at_goal = (blob.size > self.BLOB_SIZE_GOAL and distancia > self.GOAL_DISTANCE_THRESHOLD)

        
        return at_goal

    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Acciones:
        0: Avanzar recto
        1: Girar izquierda (leve)
        2: Girar derecha (leve)
        3: Girar izquierda (fuerte)
        4: Girar derecha (fuerte)
        5: Giro 180°
        """
        self.steps += 1
        # Ejecutar acción
        if action == 0:  # Avanzar
            self.robobo.moveWheelsByTime(20, 20, 1)  
        elif action == 1:  # Girar izquierda leve
            self.robobo.moveWheelsByTime(10, 20, 1)
        elif action == 2:  # Girar derecha leve
            self.robobo.moveWheelsByTime(20, 10, 1)
        elif action == 3:  # Girar izquierda fuerte
            self.robobo.moveWheelsByTime(0, 20, 1)
        elif action == 4:  # Girar derecha fuerte
            self.robobo.moveWheelsByTime(20, 0, 1)
        elif action == 5:  # Giro 180°
            self.robobo.moveWheelsByTime(20, -20, 1)

        # Obtener nuevo estado
        self.state = self._get_state()
        
        # Calcular recompensa
        reward = self._calculate_reward()

        # Verificar si alcanzó el objetivo
        terminated = self._is_at_goal()
        
        if terminated:
            print("¡OBJETIVO ALCANZADO!")
            reward += 100  # Bonus por completar (reducido de 500 para mejor gradiente)
        
        # Verificar condiciones de terminación por tiempo
        truncated = self.steps >= self.max_steps
        
        if truncated:
            print(f"Tiempo máximo alcanzado ({self.max_steps} steps)")

        return self.state, reward, terminated, truncated, {}

    def _calculate_reward(self):
        """
        Función de fitness/recompensa para NEAT.
        Premia acercarse al objetivo y mantenerlo centrado.
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_front = self.robobo.readIRSensor(IR.FrontC)
        blob_loc = self.sim.getObjectLocation('CYLINDERBALL')
        robobo_loc = self.sim.getRobotLocation(0)

        blob_loc = blob_loc['position']
        robobo_loc = robobo_loc['position']
        
        # Calcular distancia euclidiana 3D al objetivo
        distance = np.sqrt(
            (blob_loc['x'] - robobo_loc['x'])**2 +
            (blob_loc['y'] - robobo_loc['y'])**2 +
            (blob_loc['z'] - robobo_loc['z'])**2
        )

        reward = 0.0
        
        # Recompensa base inversamente proporcional a la distancia (más agresiva)
        max_distance = 2000.0
        distance_reward = 25.0 * (1.0 - min(distance / max_distance, 1.0))
        reward += distance_reward
        
        # BONUS EXTRA por estar MUY cerca (< 400 unidades)
        if distance < 400:
            proximity_bonus = (400 - distance) / 20.0  # Hasta +20 puntos
            reward += proximity_bonus
        
        # BONUS MEGA por estar SÚPER cerca (< 250 unidades)
        if distance < 250:
            mega_bonus = (250 - distance) / 10.0  # Hasta +25 puntos adicionales
            reward += mega_bonus
        
        # Recompensa por ver el blob (crítico para orientación)
        if blob.size > self.BLOB_SIZE_MIN:
            reward += 8.0  # Aumentado de 5.0
            
            # Recompensa progresiva por tamaño (proximidad visual)
            size_reward = min(blob.size / 25.0, 15.0)  # Máximo 15 puntos (antes 10)
            reward += size_reward
            
            # Recompensa crítica por centrar el blob (navegación correcta)
            center_error = abs(blob.posx - 50.0)
            if center_error < 10:
                reward += 10.0  # Muy centrado (antes 8.0)
            elif center_error < 20:
                reward += 5.0   # Bastante centrado (antes 4.0)
            elif center_error < 30:
                reward += 2.5   # Algo centrado (antes 2.0)
            else:
                # Penalización progresiva por descentrado
                reward -= center_error / 25.0
            
        else:
            # Penalización moderada si no ve el objetivo
            reward -= 5.0
        
        # Penalización por obstáculos (evitar colisiones)
        if ir_front > self.OBSTACLE_THRESHOLD_FRONT and blob.size < self.BLOB_SIZE_GOAL:
            reward -= 10.0  # Aumentado de 8.0
        
        # Penalización mínima por cada paso
        reward -= 0.1
        
        return reward

    def render(self):
        """Muestra información del estado actual."""
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_front = self.robobo.readIRSensor(IR.FrontC)
        
        print(f"\n--- Step {self.steps}/{self.max_steps} ---")
        print(f"Blob X: {blob.posx:.1f}, Size: {blob.size:.1f}")
        print(f"IR Front: {ir_front:.1f}")
        print(f"Estado: {self.state}")

    def close(self):
        """Cierra las conexiones con el robot."""
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
        except Exception as e:
            print(f"Error al cerrar conexiones: {e}")
