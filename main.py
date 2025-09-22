import gymnasium as gym

import numpy as np



from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.BlobColor import BlobColor
from gymnasium import spaces
class RoboboEnv(gym.Env):
    """Entorno personalizado de Gymnasium para el robot Robobo"""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super(RoboboEnv, self).__init__()
        
        # Espacio de observaciones (distancia y ángulo)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Espacio de acciones (3 posibles movimientos)
        self.action_space = spaces.Discrete(3)
        
        self.state = None
        self.steps = 0
        self.max_steps = 200
    
    def reset(self, *, seed=None, options=None):
        """Reinicia el entorno"""
        super().reset(seed=seed)
        
        distancia = np.random.uniform(0.5, 1.0)
        angulo = np.random.uniform(-1.0, 1.0)
        self.state = np.array([distancia, angulo], dtype=np.float32)
        
        self.steps = 0
        
        # 🔑 Importante: devolver obs, info
        return self.state, {}
    
    def step(self, action):
        """Ejecuta una acción"""
        distancia, angulo = self.state
        self.steps += 1
        
        if action == 0:   # avanzar
            distancia -= 0.05
        elif action == 1: # girar izq
            angulo -= 0.1
        elif action == 2: # girar der
            angulo += 0.1
        
        distancia = np.clip(distancia, 0.0, 1.0)
        angulo = np.clip(angulo, -1.0, 1.0)
        self.state = np.array([distancia, angulo], dtype=np.float32)
        
        # Recompensa
        reward = (1 - distancia) * 2.0 - abs(angulo)
        
        # Condiciones de finalización
        terminated = False
        truncated = False
        if distancia <= 0.05:
            reward += 10.0
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True
        
        return self.state, reward, terminated, truncated, {}
    
    def render(self):
        print(f"Robot: distancia={self.state[0]:.2f}, angulo={self.state[1]:.2f}")
    
    def close(self):
        pass
