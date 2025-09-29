import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor

class RoboboEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(RoboboEnv, self).__init__()
        self.robobo = Robobo("localhost")
        self.robobo.connect()

        self.observation_space = spaces.Discrete(6)   # 6 estados: pelota relativa
        self.action_space = spaces.Discrete(4)        # avanzar, izq, der, retro

        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # reiniciar sim
        #self.Sim.resetSimulation()
        
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        self.steps += 1
        if action == 0:  # avanzar
            self.robobo.moveWheelsByTime(10, 10, 2)  
        elif action == 1:  # girar izquierda
            self.robobo.moveWheelsByTime(0, 10, 2)
        elif action == 2:  # girar derecha
            self.robobo.moveWheelsByTime(10, 0, 2)
        elif action == 3:  
            self.robobo.moveWheelsByTime(0, 10, 4)
        elif action == 4:  
            self.robobo.moveWheelsByTime(10, 0, 4)
        elif action == 5:  
            self.robobo.moveWheelsByTime(10, -10, 4)
        # nuevo estado
        self.state = self._get_state()

        # recompensa
        reward = 0
        if self.state == 0: reward = 1
        elif self.state in [1,2,3,4]: reward = 0.5
        elif self.state in [3,4]: reward = 0.2
        else: reward = -0.2

        # comprobar distancia con IR
        distancia = self.robobo.readIRSensor(IR.FrontC)
        print("Distancia IR:", distancia)
         # si está muy cerca, penalizar
        terminated = distancia > 100
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def _get_state(self):
        self.robobo.setActiveBlobs(red=True, green=True, blue=False, custom=False)
        blobs = self.robobo.readColorBlob(BlobColor.RED)
        print("Blobs:", blobs)
        if blobs.size==0: return 5
        print("Pos X:", blobs.posx)
        x = blobs.posx
        if abs(x) < 0.1: return 0
        elif x < -0.5: return 2
        elif x < 0: return 1
        elif x > 0.5: return 4
        elif x > 0: return 3
        return 5

    def render(self):
        print(f"Estado: {self.state}")

    def close(self):
        self.robobo.disconnect()
