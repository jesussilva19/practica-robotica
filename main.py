import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.BlobColor import BlobColor
from gymnasium import spaces
from torch import seed

robobo = Robobo("localhost")
robobo.connect()
distancia = 1.0

class RoboboEnv(gym.Env):
    
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super(RoboboEnv, self).__init__()
        
 
        # Espacio de observaciones: 6 estados discretos
        self.observation_space = spaces.Discrete(6)

        
        # Espacio de acciones (3 posibles movimientos)
        self.action_space = spaces.Discrete(3)
        
        self.state = None
        self.steps = 0
        self.max_steps = 200
    
    def reset(self, *, seed=None, options=None):
        
        super().reset(seed=seed)
        
        self.state = self.observation_space.sample()
        
        self.steps = 0
        
        
        return self.state, {}
    
    

    def step(self, action):
        robobo.wait(0.5)
        
        
        self.steps += 1
        
        self.state = self.observation_space.sample()

        if action == 0:  # avanzar
            robobo.moveWheelsByTime(10, 10, 2)  
        elif action == 1:  # girar izquierda
            robobo.moveWheelsByTime(0, 10, 2)
        elif action == 2:  # girar derecha
            robobo.moveWheelsByTime(10, 0, 2)
        elif action == 3:  # retroceder
            robobo.moveWheelsByTime(0, -10, 4)
        elif action == 4:  # detenerse
            robobo.moveWheelsByTime(10, 0, 4)
        elif action == 5:  # girar 180 grados
            robobo.moveWheelsByTime(10, -10, 4)

        # Recompensa
        
        if self.state == 0:
            reward = 1
        elif self.state == 1:
            reward = 0.5        
        elif self.state == 2:
            reward = 0.5

        elif self.state == 3:
            reward = 0.2

        elif self.state == 4:
            reward = 0.2

        else:
            reward = 0

        print(f"Recompensa: {reward:.2f}")
        
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
        print(f"Estado actual: {self.state}")
    
    def close(self):
        pass
