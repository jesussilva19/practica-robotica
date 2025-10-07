import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim

class RoboboEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(RoboboEnv, self).__init__()
        self.robobo = Robobo("localhost")
        self.sim = RoboboSim("localhost") 
        self.robobo.connect()
        self.sim.connect()
        
        self.robobo.moveTiltTo(120, 50)

        self.observation_space = spaces.Discrete(6)   # 6 estados: pelota relativa
        self.action_space = spaces.Discrete(6)        # avanzar, izq, der, retro

        self.state = None
        self.steps = 0
        self.max_steps = 30

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # reiniciar sim
        self.sim.resetSimulation()  
        self.robobo.wait(1.0)
        self.robobo.moveTiltTo(115, 50)
        self.state = self._get_state()
        return self.state, {}


    def _avoid_obstacle(self):
        # Leer sensores IR
        front_c = self.robobo.readIRSensor(IR.FrontC)
        front_l = self.robobo.readIRSensor(IR.FrontL)
        front_r = self.robobo.readIRSensor(IR.FrontR)
        
        # Verificar si vemos el blob rojo (el objetivo)
        blob = self.robobo.readColorBlob(BlobColor.RED)
        veo_objetivo = blob.size > 2  # Si el blob es visible y grande
        objetivo_centrado = 45 <= blob.posx <= 55 if blob.size > 0 else False
        
        # Si hay obstáculo PERO es el objetivo centrado, NO evadir
        if veo_objetivo and objetivo_centrado and front_c > 100:
            print(" Objetivo detectado - NO evadir")
            return False
        
        # Si hay obstáculo y NO es el objetivo, evadir
        if (front_c > 100 or front_l > 300 or front_r > 300):
            if not veo_objetivo:  # Solo evadir si NO vemos el objetivo
                print(" Obstáculo detectado - Evadiendo")
                self.robobo.moveWheelsByTime(-20, -20, 1)
                self.robobo.moveWheelsByTime(30, -30, 1)
                self.robobo.wait(0.5)
                return True
        
        return False
    
    def step(self, action):
        self.steps += 1
        
        # Verificar obstáculos antes de ejecutar la acción
        if not self._avoid_obstacle():
            # Solo ejecutar la acción si no hay obstáculos
            if action == 0:  # avanzar
                self.robobo.moveWheelsByTime(5, 5, 2)  
            elif action == 1:  # girar izquierda
                self.robobo.moveWheelsByTime(0, 5, 2)
            elif action == 2:  # girar derecha
                self.robobo.moveWheelsByTime(5, 0, 2)
            elif action == 3:  
                self.robobo.moveWheelsByTime(0, 5, 4)
            elif action == 4:  
                self.robobo.moveWheelsByTime(5, 0, 4)
            elif action == 5:  
                self.robobo.moveWheelsByTime(12, -12, 4)
            
        # nuevo estado
        self.state = self._get_state()

        # recompensa
        reward = 0
        if self.state == 0: reward = 1*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [1,2]: reward = 0.5*self.robobo.readColorBlob(BlobColor.RED).size
        elif self.state in [3,4]: reward = 0.2*self.robobo.readColorBlob(BlobColor.RED).size
        else: reward = -0.5

        # comprobar distancia con IR
        distancia = self.robobo.readIRSensor(IR.FrontC)
        blob_size = self.robobo.readColorBlob(BlobColor.RED).size
        objetivo_centrado = 45 <= self.robobo.readColorBlob(BlobColor.RED).posx <= 55
        print("\n--- Step", self.steps, "---")
        print("Distancia IR:", distancia)
        print("Tamaño Blob:", self.robobo.readColorBlob(BlobColor.RED).size)
        print("Recompensa:", reward)
        print("acción:", action)
        truncated = self.steps >= self.max_steps

        # Termina si está cerca Y ve el objetivo grande Y está centrado
        terminated = (distancia > 100 and blob_size > 10 and objetivo_centrado)

        return self.state, reward, terminated, truncated, {}


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
