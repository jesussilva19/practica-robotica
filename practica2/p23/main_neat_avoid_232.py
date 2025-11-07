# main_neat_avoid_23.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim

class RoboboNEATAvoidEnv23(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=200, host="localhost"):
        super().__init__()
        # Conexión
        self.robobo = Robobo(host)
        self.sim = RoboboSim(host)
        self.robobo.connect()
        self.sim.connect()

        # Cámara (como 2.1/2.2)
        self.robobo.moveTiltTo(200, 70)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

     
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 500.0, 1000.0, 1000.0, 1000.0], dtype=np.float32),
        )
        # 8 acciones discretas
        self.action_space = spaces.Discrete(8)

        # Estado/episodio
        self.state = None
        self.steps = 0
        self.max_steps = max_steps

  
        self.OBSTACLE_THRESHOLD_FRONT = 40  
        self.OBSTACLE_THRESHOLD_SIDE  = 300
        self.BLOB_SIZE_MIN            = 2
        self.BLOB_SIZE_GOAL           = 300
        self.GOAL_DISTANCE_THRESHOLD  = 40
        self.CENTER_X = 50.0

        # Trayectorias para plano 2D
        self.trajectory = []
        self.cylinder_positions = []

    
        self.cylinder_name = "cylinder"
        self.block_name    = "block"


    def _read_robot_xy(self):
      
        try:
            x, y = self.sim.getPosition()
            return float(x), float(y)
        except Exception:
            return None, None

    def _read_cylinder_xy(self):
        try:
            cx, cy = self.sim.getObjectPosition(self.cylinder_name)
            return float(cx), float(cy)
        except Exception:
            return None, None
    # --------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.trajectory = []
        self.cylinder_positions = []

    
        self.sim.resetSimulation()
        self.robobo.wait(1.0)

        # Cámara
        self.robobo.moveTiltTo(200, 50)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        # Guardar primeras posiciones
        x, y = self._read_robot_xy()
        if x is not None:
            self.trajectory.append((x, y))
        cx, cy = self._read_cylinder_xy()
        if cx is not None:
            self.cylinder_positions.append((cx, cy))

        self.state = self._get_state()
        return self.state, {}

    def _get_state(self):
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_c = self.robobo.readIRSensor(IR.FrontC)
        ir_l = self.robobo.readIRSensor(IR.FrontL)
        ir_r = self.robobo.readIRSensor(IR.FrontR)

        blob_x = blob.posx if blob.size > 0 else self.CENTER_X
        blob_size = min(blob.size, 500.0)

        state = np.array([blob_x, blob_size, ir_c, ir_l, ir_r], dtype=np.float32)
        return state

    def _is_at_goal(self):

        blob = self.robobo.readColorBlob(BlobColor.RED)
        distancia = self.robobo.readIRSensor(IR.FrontC)
        return (blob.size > self.BLOB_SIZE_GOAL) and (distancia > self.GOAL_DISTANCE_THRESHOLD)

    def step(self, action):
        self.steps += 1

        # Acciones 
        if action == 0:   # Avanzar recto
            self.robobo.movePanTo(0, 100, True)
            self.robobo.moveWheelsByTime(20, 20, 1)
            
        elif action == 1: # Girar izquierda (leve)
            self.robobo.movePanTo(0, 100, True)
            self.robobo.moveWheelsByTime(5, 10, 1)
            
        elif action == 2: # Girar derecha (leve)
            self.robobo.movePanTo(0, 100, True)
            self.robobo.moveWheelsByTime(10, 5, 1)
            
        elif action == 3: # Girar izquierda (fuerte)
            self.robobo.movePanTo(0, 100, True)
            self.robobo.moveWheelsByTime(0, 10, 1)
            
        elif action == 4: # Girar derecha (fuerte)
            self.robobo.movePanTo(0, 100, True)
            self.robobo.moveWheelsByTime(10, 0, 1)
            
        elif action == 5: # Giro 180°
            self.robobo.movePanTo(0, 100, True)
            self.robobo.moveWheelsByTime(10, -10, 1)
            
            


        elif action == 6: # Giro 180°
            self.robobo.movePanTo(45, 100, True)
            self.robobo.moveWheelsByTime(10, -10, 1)
            self.robobo.moveWheelsByTime(20, 20, 1)
            self.robobo.moveWheelsByTime(-10, 10, 1)
            self.robobo.moveWheelsByTime(20, 20, 1)
            

        elif action == 7: # Giro 180°
            self.robobo.movePanTo(-45, 100, True)
            self.robobo.moveWheelsByTime(-10, 10, 1)
            self.robobo.moveWheelsByTime(20, 20, 1)
            self.robobo.moveWheelsByTime(10, -10, 1)
            self.robobo.moveWheelsByTime(20, 20, 1)
            

        # Nuevo estado
        self.state = self._get_state()

        # Recompensa
        reward = self._calculate_reward()

        # Terminación
        terminated = self._is_at_goal()
        if terminated:
            print("¡OBJETIVO ALCANZADO!")
            reward += 500

        truncated = self.steps >= self.max_steps
        if truncated:
            print(f"⏱️ Tiempo máximo alcanzado ({self.max_steps} steps)")
            reward -= 50

        # Guardar posiciones para plano 2D
        x, y = self._read_robot_xy()
        if x is not None:
            self.trajectory.append((x, y))
        cx, cy = self._read_cylinder_xy()
        if cx is not None:
            self.cylinder_positions.append((cx, cy))

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
            
            # Calcular distancia al objetivo
            distance = np.sqrt(
                (blob_loc['x'] - robobo_loc['x'])**2 +
                (blob_loc['y'] - robobo_loc['y'])**2 +
                (blob_loc['z'] - robobo_loc['z'])**2
            )

            reward = 0.0
            # Recompensa por ver el blob (detectar el objetivo)
            if blob.size > self.BLOB_SIZE_MIN:
                reward += 10.0
                
                # Recompensa por tamaño del blob (más grande = más cerca)
                size_reward = min(blob.size / 50.0, 5.0)  # Máximo 5 puntos
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
                reward -= 1.0
            
            # Penalización por estar muy cerca de obstáculos
            if ir_front > self.OBSTACLE_THRESHOLD_FRONT:
                reward -= 0.05
            
            # Penalización muy pequeña por cada paso
            reward -= 0.05

            # Recompensa por acercarse al objetivo
            if distance > 0:
                reward += 500.0 / distance  # Más cerca = mayor recompensa



            
            return reward

    def render(self):
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_front = self.robobo.readIRSensor(IR.FrontC)
        print(f"[{self.steps}/{self.max_steps}] Blob(x={blob.posx:.1f}, size={blob.size:.1f}) IR={ir_front:.1f}")

    def close(self):
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
            print(" Conexiones cerradas correctamente")
        except Exception as e:
            print(f" Error al cerrar conexiones: {e}")
