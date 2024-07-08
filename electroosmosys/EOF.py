import espressomd
import numpy as np
import matplotlib.pyplot as plt


box_length=10
num_free_particles = 10

system = espressomd.System(box_l=[box_length, box_length, box_length])
system.time_step = 0.01
system.cell_system.skin = 0.4
system.periodicity = [False, False, True]


new_parts = system.part.add(pos=np.random.random((num_free_particles, 3)) * box_length)
