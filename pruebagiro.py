import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim

robobo = Robobo('localhost')

robobo.connect()


robobo.moveWheelsByTime(10, -10, 3.5)
            


