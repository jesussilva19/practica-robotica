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
        self.observation_space = spaces.Discrete(14)  
        self.action_space = spaces.Discrete(6)       

        # Variables de estado
        self.state = None
        self.steps = 0
        self.max_steps = max_steps

        # Trayectorias
        self.trajectory = []          # posiciones del Robobo
        self.cylinder_positions = []  # posiciones del cilindro (opcional)
        
        # Constantes para detección
        self.OBSTACLE_THRESHOLD_FRONT = 30
        self.OBSTACLE_THRESHOLD_SIDE = 300
        self.BLOB_SIZE_MIN = 2
        self.BLOB_SIZE_GOAL = 10
        self.CENTER_MIN = 45
        self.CENTER_MAX = 55
        
        # Posiciones del pan para búsqueda
        self.pan_positions = [0, 15, 30, 45, 60, 75, 90,
                              -15, -30, -45, -60, -75, -90]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.trajectory = []
        self.cylinder_positions = []
        
        # Reiniciar simulación
        self.sim.resetSimulation()  
        self.robobo.wait(1.0)
        
        # Reconfigurar cámara
        self.robobo.moveTiltTo(200, 50)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)
        
        self.state = self._get_state()

        # Guardar posición inicial
        try:
            x, y = self.sim.getPosition()
            self.trajectory.append((x, y))
        except:
            print("⚠️ No se pudo obtener posición inicial del Robobo")

        try:
            cx, cy = self.sim.getCylinderPosition()
            self.cylinder_positions.append((cx, cy))
        except:
            pass

        return self.state, {}

    def _is_goal_centered(self, blob):
        return True

    def _is_at_goal(self):
        blob = self.robobo.readColorBlob(BlobColor.RED)
        distancia = self.robobo.readIRSensor(IR.FrontC)
        objetivo_centrado = self._is_goal_centered(blob)
        return (blob.size > self.BLOB_SIZE_GOAL and 
                objetivo_centrado and 
                distancia > self.OBSTACLE_THRESHOLD_FRONT)
    
    def _avoid_obstacle(self):
        if self._is_at_goal():
            return False
        
        front_c = self.robobo.readIRSensor(IR.FrontC)
        front_l = self.robobo.readIRSensor(IR.FrontL)
        front_r = self.robobo.readIRSensor(IR.FrontR)
        
        blob = self.robobo.readColorBlob(BlobColor.RED)
        veo_objetivo = blob.size > self.BLOB_SIZE_MIN
        objetivo_centrado = self._is_goal_centered(blob)
        
        if veo_objetivo and objetivo_centrado and blob.size > 5:
            return False
        
        has_obstacle = (front_c > self.OBSTACLE_THRESHOLD_FRONT or 
                        front_l > self.OBSTACLE_THRESHOLD_SIDE or 
                        front_r > self.OBSTACLE_THRESHOLD_SIDE)
        
        if has_obstacle:
            self.robobo.moveWheelsByTime(-20, -20, 1)  
            self.robobo.moveWheelsByTime(30, -30, 1)   
            self.robobo.wait(0.5)
            return True
        
        return False
    
    def step(self, action):
        self.steps += 1
        evaded = self._avoid_obstacle()
        
        if not evaded:
            if action == 0:  
                self.robobo.moveWheelsByTime(5, 5, 2)  
            elif action == 1:  
                self.robobo.moveWheelsByTime(0, 5, 2)
            elif action == 2:  
                self.robobo.moveWheelsByTime(5, 0, 2)
            elif action == 3:  
                self.robobo.moveWheelsByTime(0, 5, 4)
            elif action == 4:  
                self.robobo.moveWheelsByTime(5, 0, 4)
            elif action == 5:  
                self.robobo.moveWheelsByTime(10, -10, 3)

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

        terminated = self._is_at_goal()
        if terminated:
            reward += 200  
        
        truncated = self.steps >= self.max_steps

        # Guardar posición actual
        try:
            x, y = self.sim.getPosition()
            self.trajectory.append((x, y))
        except:
            pass

        try:
            cx, cy = self.sim.getCylinderPosition()
            self.cylinder_positions.append((cx, cy))
        except:
            pass

        return self.state, reward, terminated, truncated, {}

    def _get_state(self):
        for i, ang in enumerate(self.pan_positions):
            self.robobo.movePanTo(ang, 100, True)
            blobs = self.robobo.readColorBlob(BlobColor.RED)
            if blobs.size > 0:
                return i
        return len(self.pan_positions)

    def render(self):
        print(f"Estado actual: {self.state}")

    def close(self):
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
            print(" Conexiones cerradas correctamente")
        except Exception as e:
            print(f"Error al cerrar conexiones: {e}")
