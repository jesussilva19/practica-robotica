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
        self.robobo.moveTiltTo(200, 50)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        # NEAT necesita entradas continuas (no estados discretos)
        # Entradas: [blob_x, blob_size, ir_front_c, ir_front_l, ir_front_r]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([100.0, 100.0, 1000.0, 1000.0, 1000.0]),
            dtype=np.float32
        )
        
        # 6 acciones discretas
        self.action_space = spaces.Discrete(6)

        # Variables de estado
        self.state = None
        self.steps = 0
        self.max_steps = max_steps
        
        # Constantes para detección
        self.OBSTACLE_THRESHOLD_FRONT = 30
        self.OBSTACLE_THRESHOLD_SIDE = 300
        self.BLOB_SIZE_MIN = 2
        self.BLOB_SIZE_GOAL = 15  # Tamaño para considerar objetivo alcanzado
        self.GOAL_DISTANCE_THRESHOLD = 50  # Distancia IR para objetivo

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
        blob_size = min(blob.size, 100.0)  # Limitar a 100
        
        state = np.array([
            blob_x,
            blob_size, 
            ir_front_c,
            ir_front_l,
            ir_front_r
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
            self.robobo.moveWheelsByTime(10, 10, 1)  
        elif action == 1:  # Girar izquierda leve
            self.robobo.moveWheelsByTime(5, 10, 1)
        elif action == 2:  # Girar derecha leve
            self.robobo.moveWheelsByTime(10, 5, 1)
        elif action == 3:  # Girar izquierda fuerte
            self.robobo.moveWheelsByTime(0, 10, 1)
        elif action == 4:  # Girar derecha fuerte
            self.robobo.moveWheelsByTime(10, 0, 1)
        elif action == 5:  # Giro 180°
            self.robobo.moveWheelsByTime(10, -10, 2)

        # Obtener nuevo estado
        self.state = self._get_state()
        
        # Calcular recompensa
        reward = self._calculate_reward()

        # Verificar si alcanzó el objetivo
        terminated = self._is_at_goal()
        
        if terminated:
            print("🎯 ¡OBJETIVO ALCANZADO! 🎯")
            reward += 500  # Gran recompensa por completar el objetivo
        
        # Verificar condiciones de terminación por tiempo
        truncated = self.steps >= self.max_steps
        
        if truncated:
            print(f"⏱️ Tiempo máximo alcanzado ({self.max_steps} steps)")
            reward -= 50  # Penalización por no completar

        return self.state, reward, terminated, truncated, {}

    def _calculate_reward(self):
        """
        Función de fitness/recompensa para NEAT.
        Premia acercarse al objetivo y mantenerlo centrado.
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_front = self.robobo.readIRSensor(IR.FrontC)
        
        reward = 0.0
        
        # Recompensa por ver el blob (detectar el objetivo)
        if blob.size > self.BLOB_SIZE_MIN:
            reward += 1.0
            
            # Recompensa por tamaño del blob (más grande = más cerca)
            size_reward = min(blob.size / 20.0, 5.0)  # Máximo 5 puntos
            reward += size_reward
            
            # Recompensa por centrar el blob
            center_error = abs(blob.posx - 50.0)  # 50 es el centro
            if center_error < 10:
                reward += 3.0  # Muy centrado
            elif center_error < 20:
                reward += 1.5  # Bastante centrado
            elif center_error < 30:
                reward += 0.5  # Algo centrado
            
            # Penalización por estar descentrado
            reward -= center_error / 50.0
            
        else:
            # Penalización fuerte si no ve el objetivo
            reward -= 2.0
        
        # Penalización por estar muy cerca de obstáculos (excepto el objetivo)
        if ir_front < self.OBSTACLE_THRESHOLD_FRONT and blob.size < self.BLOB_SIZE_GOAL:
            reward -= 5.0
        
        # Pequeña penalización por cada paso (fomenta rapidez)
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
            print("✅ Conexiones cerradas correctamente")
        except Exception as e:
            print(f"❌ Error al cerrar conexiones: {e}")