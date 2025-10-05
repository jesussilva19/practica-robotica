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
        self.robobo.moveTiltTo(115, 50)

        self.observation_space = spaces.Discrete(6)   # 6 estados: pelota relativa
        self.action_space = spaces.Discrete(6)        # avanzar, izq, der, retro

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


    def _avoid_obstacle(self):

        if (self.robobo.readIRSensor(IR.FrontC) > 100 or 
            self.robobo.readIRSensor(IR.FrontL) > 300 or 
            self.robobo.readIRSensor(IR.FrontR) > 300):
            
            # Maniobra de evasión: retroceder y girar
            self.robobo.moveWheelsByTime(-20, -20, 1)  # Retroceder
            self.robobo.moveWheelsByTime(30, -30, 1)   # Girar a la izquierda
            self.robobo.wait(0.5)                      # Esperar
            return True  
        return False  

    def step(self, action):
        self.steps += 1
        
        # Verificar obstáculos antes de ejecutar la acción
        if not self._avoid_obstacle():
            # Solo ejecutar la acción si no hay obstáculos
            if action == 0:  # avanzar
                self.robobo.moveWheelsByTime(10, 10, 2)  
            elif action == 1:  # girar izquierda
                self.robobo.moveWheelsByTime(0, 5, 2)
            elif action == 2:  # girar derecha
                self.robobo.moveWheelsByTime(5, 0, 2)
            elif action == 3:  
                self.robobo.moveWheelsByTime(0, 5, 4)
            elif action == 4:  
                self.robobo.moveWheelsByTime(5, 0, 4)
            elif action == 5:  
                self.robobo.moveWheelsByTime(10, -10, 4)
            
        # nuevo estado
        self.state = self._get_state()

        # recompensa
        reward = 0
        if self.state == 0: reward = 1*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [1,2]: reward = 0.5*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [3,4]: reward = 0.2*self.robobo.readColorBlob(BlobColor.RED).size
        else: reward = -0.2

        # comprobar distancia con IR
        distancia = self.robobo.readIRSensor(IR.FrontC)
        print("Distancia IR:", distancia)
        print("Tamaño Blob:", self.robobo.readColorBlob(BlobColor.RED).size)
        print("Recompensa:", reward)
        print("acción:", action)
         # si está muy cerca, penalizar
        terminated = distancia > 100 and self.robobo.readColorBlob(BlobColor.RED).size>10
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def _get_state(self):
        self.robobo.setActiveBlobs(red=True, green=True, blue=False, custom=False)
        blobs = self.robobo.readColorBlob(BlobColor.RED)
        print("Blobs:", blobs)
        if blobs.size==0: return 5
        print("Pos X:", blobs.posx)
        x = blobs.posx
        if 45 <= x <= 55: return 0
        elif 25 <= x < 45: return 1
        elif 55 < x <= 75: return 2
        elif 1<= x < 25: return 3
        elif 75 < x: return 4
        return 5


    def _get_state(self):
        
        posiciones = [0, 20, -20, 90, -90]  
        

        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        for i, ang in enumerate(posiciones):
            
            self.robobo.movePanTo(ang, 100, True)

           
            blobs = self.robobo.readColorBlob(BlobColor.RED)

            if blobs.size > 0:  
                return i  

        return len(posiciones)  

    def render(self):
        print(f"Estado: {self.state}")

    def close(self):
        self.robobo.disconnect()
