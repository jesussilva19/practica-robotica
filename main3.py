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
        self.observation_space = spaces.Discrete(42)  # 6 estados según posición del objetivo
        self.action_space = spaces.Discrete(6)       # 6 acciones de movimiento

        # Variables de estado
        self.state = None
        self.steps = 0
        self.max_steps = max_steps

        # Constantes para detección de obstáculos
        self.OBSTACLE_THRESHOLD_FRONT = 30
        self.OBSTACLE_THRESHOLD_SIDE = 300
        self.OBSTACLE_DANGER_ZONE = 50  # Zona de peligro para penalización
        
        
        # Constantes para detección
        self.OBSTACLE_THRESHOLD_FRONT = 30
        self.OBSTACLE_THRESHOLD_SIDE = 300
        self.BLOB_SIZE_MIN = 2
        self.BLOB_SIZE_GOAL = 50
        self.BLOB_SIZE_CLOSE = 20  # Si el blob es grande, probablemente es el objetivo cerca
        self.CENTER_MIN = 45
        self.CENTER_MAX = 55

        # Categorías de distancia basadas en tamaño del blob
        self.DISTANCE_CLOSE = 30   # blob.size >= 30
        self.DISTANCE_MEDIUM = 20   # blob.size >= 20
        # DISTANCE_FAR: blob.size < 20

        # Posiciones del pan para búsqueda
        self.pan_positions = [0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60, -75,-90]

    def reset(self, *, seed=None, options=None):
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

    def _is_goal_centered(self, blob):
        """Verifica si el objetivo está centrado en la visión.
        if blob.size == 0:
            return False
        return self.CENTER_MIN <= blob.posx <= self.CENTER_MAX"""
        return True

    def _is_at_goal(self):
        """
        Verifica si el robot ha alcanzado el objetivo.
        Retorna True si está en el objetivo, False en caso contrario.
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        distancia = self.robobo.readIRSensor(IR.FrontC)
        objetivo_centrado = self._is_goal_centered(blob)
        
        # Condiciones para considerar que llegó al objetivo:
        # 1. El blob es lo suficientemente grande
        # 2. Está centrado en la visión
        # 3. La distancia es muy cercana
        at_goal = (blob.size > self.BLOB_SIZE_GOAL and 
                   objetivo_centrado and 
                   distancia > self.OBSTACLE_THRESHOLD_FRONT)
        
        return at_goal
    
    def _is_approaching_goal(self):
        """
        Determina si el robot se está acercando al OBJETIVO (no a un obstáculo).
        Retorna True si hay evidencia clara del objetivo rojo delante.
        """
        blob = self.robobo.readColorBlob(BlobColor.RED)
        
        # Criterios para identificar que es el OBJETIVO:
        # 1. El blob rojo es visible y de tamaño significativo
        # 2. Está relativamente centrado (o al menos visible)
        is_goal = (blob.size >= self.BLOB_SIZE_CLOSE and 
                   blob.posx > 20 and blob.posx < 80)
        
        if is_goal:
            print(f"🎯 Objetivo detectado delante (size: {blob.size}, posx: {blob.posx})")
        
        return is_goal
    
    def _check_collision_risk(self):
        """
        Verifica si hay riesgo de colisión con un OBSTÁCULO (no el objetivo).
        Retorna el nivel de peligro: 0 (seguro), 1 (precaución), 2 (peligro)
        """
        # Leer sensores IR
        front_c = self.robobo.readIRSensor(IR.FrontC)
        front_l = self.robobo.readIRSensor(IR.FrontL)
        front_r = self.robobo.readIRSensor(IR.FrontR)
        
        # IMPORTANTE: Primero verificar si es el objetivo
        if self._is_approaching_goal():
            print("✅ Acercándose al objetivo - Sin penalización")
            return 0  # No hay peligro, es el objetivo
        
        # Si NO es el objetivo y hay lecturas altas de IR = OBSTÁCULO
        max_ir = max(front_c, front_l, front_r)
        
        if max_ir > self.OBSTACLE_DANGER_ZONE:
            print(f"⚠️⚠️ PELIGRO: Obstáculo muy cerca (IR: {max_ir})")
            return 2  # Peligro alto
        elif max_ir > self.OBSTACLE_THRESHOLD_FRONT:
            print(f"⚠️ PRECAUCIÓN: Obstáculo detectado (IR: {max_ir})")
            return 1  # Precaución
        
        return 0  # Seguro
    
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Acciones:
        0: Avanzar recto
        1: Girar izquierda (leve)
        2: Girar derecha (leve)
        3: Girar izquierda (fuerte)
        4: Girar derecha (fuerte)
        5: Girar 180 grados
        """
        self.steps += 1
        

        if action == 0:  # Avanzar
            self.robobo.moveWheelsByTime(5, 5, 2)  
        elif action == 1:  # Girar izquierda leve
            self.robobo.moveWheelsByTime(0, 5, 2)
        elif action == 2:  # Girar derecha leve
            self.robobo.moveWheelsByTime(5, 0, 2)
        elif action == 3:  # Girar izquierda fuerte
            self.robobo.moveWheelsByTime(0, 5, 4)
        elif action == 4:  # Girar derecha fuerte
            self.robobo.moveWheelsByTime(5, 0, 4)
        elif action == 5:  # Giro 180°
            self.robobo.moveWheelsByTime(10, -10, 3)


        collision_risk = self._check_collision_risk()

        # Obtener nuevo estado
        self.state = self._get_state()
   
        # Extraer posición y distancia del estado
        position = self.state // 3
        distance_category = self.state % 3
        
        # Calcular recompensa basada en posición y distancia
        blob = self.robobo.readColorBlob(BlobColor.RED)
        
        # Recompensa base según posición (0 = centrado, mejor)
        if position == 0:  # Centrado
            position_reward = 3.0
        elif position in [1, 7]:  # Centro-izquierda/derecha
            position_reward = 1.5
        elif position in [2, 8]:
            position_reward = 1.0
        elif position in [3, 9]:
            position_reward = 0.6
        elif position in [4, 10]:
            position_reward = 0.4
        elif position in [5, 11]:
            position_reward = 0.2
        elif position in [6, 12]:
            position_reward = 0.1
        else:  # No visible
            position_reward = -5.0
        
        # Multiplicador según distancia (más cerca = mejor)
        if distance_category == 0:  # Cerca
            distance_multiplier = 3.0
        elif distance_category == 1:  # Media
            distance_multiplier = 1.5
        else:  # Lejos
            distance_multiplier = 0.5

                
        # Recompensa final combinada
        if position == 13:  # No visible
            reward = position_reward
        else:
            reward = position_reward * distance_multiplier
        
        # PENALIZACIÓN POR COLISIÓN (solo si NO es el objetivo)
        if collision_risk == 2:  # Peligro alto
            reward -= 15.0
            print("💥 PENALIZACIÓN ALTA: Demasiado cerca de obstáculo")
        elif collision_risk == 1:  # Precaución
            reward -= 5.0
            print("⚠️ PENALIZACIÓN MEDIA: Obstáculo cerca")
        
        # Verificar si alcanzó el objetivo
        terminated = self._is_at_goal()
        
        if terminated:
            print("✅ ¡OBJETIVO ALCANZADO! 🎉")
            reward += 200  # Gran recompensa por completar el objetivo
        
        # Verificar condiciones de terminación por tiempo
        truncated = self.steps >= self.max_steps
        
        # Obtener información para logging
        distancia_ir = self.robobo.readIRSensor(IR.FrontC)
        
        # Logging
        print(f"\n--- Step {self.steps} ---")
        print(f"Acción: {action}")
        print(f"Estado: {self.state} (Pos: {position}, Dist: {distance_category})")
        print(f"Distancia IR: {distancia_ir}")
        print(f"Tamaño Blob: {blob.size}")
        print(f"Blob Pos X: {blob.posx}")
        print(f"Riesgo Colisión: {collision_risk}")
        print(f"Recompensa: {reward:.2f}")
        
        if truncated:
            print("⏱️ Tiempo máximo alcanzado - Episodio truncado")

        return self.state, reward, terminated, truncated, {}



    """
    def _get_state(self):
      
        Determina el estado actual basado en la posición del objetivo rojo.
        
        Estados:
        0: Objetivo centrado (45-55)
        1: Objetivo centro-izquierda (25-45)
        2: Objetivo centro-derecha (55-75)
        3: Objetivo extremo izquierda (1-25)
        4: Objetivo extremo derecha (75+)
        5: Objetivo no visible
    
        # Buscar objetivo moviendo la cámara pan
        for i, ang in enumerate(self.pan_positions):
            self.robobo.movePanTo(ang, 100, True)
            blobs = self.robobo.readColorBlob(BlobColor.RED)
            
            if blobs.size > 0:
                print(f"blobs.size: {blobs.size}")
                # Retornar a posición central después de encontrar
            
                return i
        
        # Si no se encuentra en ninguna posición
        return len(self.pan_positions)
        """
    def _get_state(self):
        """
        Determina el estado actual basado en la posición del objetivo rojo y su distancia.
        
        Estados combinados: posición (14) × distancia (3) = 42 estados totales
        
        Posiciones (13 ángulos + no visible):
        0-12: Posiciones de pan desde 0° hasta -90°
        13: Objetivo no visible
        
        Distancias:
        0: Cerca (blob.size >= 10)
        1: Media (5 <= blob.size < 10)
        2: Lejos (blob.size < 5)
        
        Fórmula del estado: posición * 3 + categoría_distancia
        """
        # Buscar objetivo moviendo la cámara pan
        for i, ang in enumerate(self.pan_positions):
            self.robobo.movePanTo(ang, 100, True)
            blob = self.robobo.readColorBlob(BlobColor.RED)
            
            if blob.size > 0:
                print(f"🔴 Blob detectado - size: {blob.size}, posición: {i}, ángulo: {ang}°")
                
                # Categorizar distancia según tamaño del blob
                if blob.size >= self.DISTANCE_CLOSE:
                    distance_category = 0  # Cerca
                    distance_label = "Cerca"
                elif blob.size >= self.DISTANCE_MEDIUM:
                    distance_category = 1  # Media
                    distance_label = "Media"
                else:
                    distance_category = 2  # Lejos
                    distance_label = "Lejos"
                
                print(f"📏 Distancia: {distance_label} (categoría {distance_category})")
                
                # Calcular estado combinado
                state = i * 3 + distance_category
                return state
        
        # Si no se encuentra en ninguna posición: estados 39, 40, 41
        # Usamos estado 39 (posición 13 × 3 + 0) para "no visible"
        print("❌ Objetivo no visible")
        return len(self.pan_positions) * 3

    def render(self):
        """Muestra información del estado actual."""
        if self.state is None:
            print("Estado no inicializado")
            return
        
        position = self.state // 3
        distance_category = self.state % 3
        
        # Nombres de estados
        position_names = {
            0: "0°", 1: "15°", 2: "30°", 3: "45°", 
            4: "60°", 5: "75°", 6: "90°",
            7: "-15°", 8: "-30°", 9: "-45°",
            10: "-60°", 11: "-75°", 12: "-90°",
            13: "No Visible"
        }
        
        distance_names = {0: "Cerca", 1: "Media", 2: "Lejos"}
        
        print(f"Estado: {self.state} | "
              f"Posición: {position_names.get(position, 'Desconocido')} | "
              f"Distancia: {distance_names.get(distance_category, 'Desconocido')}")

    def close(self):
        """Cierra las conexiones con el robot."""
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
            print("✅ Conexiones cerradas correctamente")
        except Exception as e:
            print(f"❌ Error al cerrar conexiones: {e}")