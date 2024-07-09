import espressomd
import numpy as np
import espressomd.io.writer.vtf as vtf
from tqdm import tqdm

box_length = 16
num_free_particles = 200
N_particles_per_wall_per_dim = 20

system = espressomd.System(box_l=[box_length, box_length, box_length])
system.time_step = 0.01
system.cell_system.skin = 0.4
system.periodicity = [True, True, True]

fp = open("trajectory.vtf", mode="w+t")

# Ensure free particles are well within the box to avoid entering ELC gap region
safe_margin = 2.0
new_parts = system.part.add(
    pos=np.random.uniform(low=safe_margin, high=box_length-safe_margin, size=(num_free_particles, 3)),
    q=[-1] * num_free_particles,
    m=[1] * num_free_particles,
    type=[0] * num_free_particles
)

# Add wall particles
for i in range(N_particles_per_wall_per_dim):
    for j in range(N_particles_per_wall_per_dim):
        system.part.add(
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,1.0],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )
        system.part.add(
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,box_length-1.0],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )

# Define non-bonded interactions
system.non_bonded_inter[0, 0].wca.set_params(epsilon=1, sigma=1)
system.non_bonded_inter[0, 1].wca.set_params(epsilon=1, sigma=1)



# Write initial configuration
vtf.writevsf(system, fp)
vtf.writevcf(system, fp)

print("Cooldown")
system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
system.integrator.run(1000)
vtf.writevcf(system, fp)

# Set up electrostatics with ELC
solver = espressomd.electrostatics.P3M(prefactor=0.5, accuracy=1e-3)
elc_gap = box_length * 0.1
elc = espressomd.electrostatics.ELC(actor=solver, gap_size=elc_gap, maxPWerror=1e-2)
system.electrostatics.solver = solver

print("Run")
system.integrator.set_vv()
for i in tqdm(range(100)):
    system.integrator.run(100)
    vtf.writevcf(system, fp)
fp.close()
