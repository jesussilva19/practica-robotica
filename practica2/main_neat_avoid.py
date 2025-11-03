import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim

class RoboboNEATAvoidEnv(gym.Env):
    """
    Práctica 2.2 – Avoid the block (basado en parámetros 2.1)
    - Posición inicial fija exigida
    - Objetivo: acercarse al cilindro rojo evitando el bloque
    - Observación/acciones/recompensas con misma escala que 2.1
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=200, host="localhost"):
        super().__init__()
        # Conexión
        self.robobo = Robobo(host)
        self.sim = RoboboSim(host)
        self.robobo.connect()
        self.sim.connect()

        # Cámara (mismos ángulos 2.1)
        self.robobo.moveTiltTo(200, 70)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        # Observación continua NEAT (mismos rangos 2.1)
        # [blob_x (0..100), blob_size (0..500), IR C/L/R (0..1000)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 500.0, 1000.0, 1000.0, 1000.0], dtype=np.float32),
        )
        # 6 acciones discretas (igual 2.1)
        self.action_space = spaces.Discrete(6)

        # Estado/episodio
        self.state = None
        self.steps = 0
        self.max_steps = max_steps

        # Constantes (idénticas a 2.1)
        self.OBSTACLE_THRESHOLD_FRONT = 30
        self.OBSTACLE_THRESHOLD_SIDE = 300
        self.BLOB_SIZE_MIN = 2
        self.BLOB_SIZE_GOAL = 300
        self.GOAL_DISTANCE_THRESHOLD = 30  # (se mantiene aunque es estricta)
        self.CENTER_X = 50.0

        # Trayectorias para plano 2D
        self.trajectory = []
        self.cylinder_positions = []

        # Nombres en la sim (ajusta si difieren)
        self.cylinder_name = "cylinder"
        self.block_name = "block"



    def _read_robot_xy(self):
        try:
            x, y = self.sim.getPosition()  # algunas versiones devuelven (x, y) o (x, z)
            return float(x), float(y)
        except Exception:
            return None, None

    def _read_cylinder_xy(self):
        try:
            cx, cy = self.sim.getObjectPosition(self.cylinder_name)
            return float(cx), float(cy)
        except Exception:
            return None, None
    # ---------------------------------------------------------------
    def _place_initial_pose(self):
        """Posición inicial exigida por el enunciado."""
        try:
            
            
            self.sim.setRobotLocation(0, {'x': -1000.0, 'y': 39.0, 'z': -400.0})
            
            
        except Exception as e:
            print(f"⚠️ No se pudo fijar la pose inicial: {e}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.trajectory = []
        self.cylinder_positions = []

        self.sim.resetSimulation()
        self.robobo.wait(1.0)

        # 👉 POSICIÓN INICIAL OBLIGATORIA
        self._place_initial_pose()

        # Reconfig cámara igual que en 2.1
        self.robobo.moveTiltTo(200, 50)
        self.robobo.setActiveBlobs(red=True, green=False, blue=False, custom=False)

        # Guardar primera muestra de trayectoria
        x, y = self._read_robot_xy()
        if x is not None:
            self.trajectory.append((x, y))

        # (opcional) registrar posición del cilindro para el plano 2D
        cx, cy = self._read_cylinder_xy()
        if cx is not None:
            self.cylinder_positions.append((cx, cy))

        self.state = self._get_state()

        # Comprobación rápida por consola (útil para la memoria)
        print("📍 Pose inicial fijada a x=-1000.0, y=39.0, z=-400.0")

        return self.state, {}

    def _get_state(self):
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_c = self.robobo.readIRSensor(IR.FrontC)
        ir_l = self.robobo.readIRSensor(IR.FrontL)
        ir_r = self.robobo.readIRSensor(IR.FrontR)

        blob_x = blob.posx if blob.size > 0 else self.CENTER_X
        blob_size = min(blob.size, 500.0)

        state = np.array([blob_x, blob_size, ir_c, ir_l, ir_r], dtype=np.float32)
        # print(f"Estado: {state}")  # descomenta si quieres log
        return state

    def _is_at_goal(self):
        """
        Igual criterio que 2.1 (aunque es estricto).
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        distancia = self.robobo.readIRSensor(IR.FrontC)
        return (blob.size > self.BLOB_SIZE_GOAL) and (distancia > self.GOAL_DISTANCE_THRESHOLD)

    def step(self, action):
        self.steps += 1

        # Acciones (mismas que 2.1)
        if action == 0:   # Avanzar recto
            self.robobo.moveWheelsByTime(10, 10, 1)
        elif action == 1: # Girar izquierda (leve)
            self.robobo.moveWheelsByTime(5, 10, 1)
        elif action == 2: # Girar derecha (leve)
            self.robobo.moveWheelsByTime(10, 5, 1)
        elif action == 3: # Girar izquierda (fuerte)
            self.robobo.moveWheelsByTime(0, 10, 1)
        elif action == 4: # Girar derecha (fuerte)
            self.robobo.moveWheelsByTime(10, 0, 1)
        elif action == 5: # Giro 180°
            self.robobo.moveWheelsByTime(10, -10, 2)

        # Nuevo estado y recompensa
        self.state = self._get_state()
        reward = self._calculate_reward()

        # Terminación
        terminated = self._is_at_goal()
        if terminated:
            print("🎯 ¡OBJETIVO ALCANZADO! 🎯")
            reward += 500  # igual que 2.1

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
        Misma estructura que 2.1, con penalización adicional por cercanía al obstáculo
        (aprovechando IR Front si no se ve un blob grande).
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_front = self.robobo.readIRSensor(IR.FrontC)

        reward = 0.0

        # Ver el blob
        if blob.size > self.BLOB_SIZE_MIN:
            reward += 1.0

            # Tamaño (más grande = más cerca)
            reward += min(blob.size / 50.0, 5.0)  # mismo límite 5 pts (2.1)

            # Centrando
            center_error = abs(blob.posx - self.CENTER_X)
            if center_error < 10:
                reward += 3.0
            elif center_error < 20:
                reward += 1.5
            elif center_error < 30:
                reward += 0.5

            # Penalización por descentrado
            reward -= center_error / 50.0
        else:
            # No ver objetivo
            reward -= 3.0

        # Evitar bloque: si IR muy bajo y blob no es grande (no es el objetivo “ya”)
        if ir_front < self.OBSTACLE_THRESHOLD_FRONT and blob.size < self.BLOB_SIZE_GOAL:
            reward -= 5.0

        # Penalización por paso (fomenta rapidez)
        reward -= 0.1

        return float(reward)

    def render(self):
        blob = self.robobo.readColorBlob(BlobColor.RED)
        ir_front = self.robobo.readIRSensor(IR.FrontC)
        print(f"\n--- Step {self.steps}/{self.max_steps} ---")
        print(f"Blob X: {blob.posx:.1f}, Size: {blob.size:.1f}")
        print(f"IR Front: {ir_front:.1f}")
        print(f"Estado: {self.state}")

    def close(self):
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
            print("✅ Conexiones cerradas correctamente")
        except Exception as e:
            print(f"❌ Error al cerrar conexiones: {e}")
