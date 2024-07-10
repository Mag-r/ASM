import espressomd
import numpy as np
import espressomd.io.writer.vtf as vtf
import espressomd.shapes
import espressomd.lb
import espressomd.observables as obs
from tqdm import tqdm


#unit defined by elementary charge,k_b*300K as Energy, nm as length, 1 as mass
box_length = 16
safe_margin = 1.0
elc_gap=5
bjerrum_length = 0.7095
num_free_particles = 200
N_particles_per_wall_per_dim = 20
fluid_density = 26.18
viscosity = 0.25
friction = 15.0
E_field = [25.0,0,0]

system = espressomd.System(box_l=[box_length, box_length, box_length+2*safe_margin+elc_gap])
system.time_step = 0.01
system.cell_system.skin = 0.4
system.periodicity = [True, True, True]

# Lattice-Boltzmann
lbf = espressomd.lb.LBFluidWalberla(agrid=1.0,density = fluid_density, kinematic_viscosity = viscosity, tau=0.01)
system.lb=lbf

fp = open("trajectory.vtf", mode="w+t")

# Ensure free particles are well within the box to avoid entering ELC gap region

new_parts = system.part.add(
    pos=np.random.uniform(low=2*safe_margin, high=box_length-safe_margin, size=(num_free_particles, 3)),
    q=[-1] * num_free_particles,
    m=[1] * num_free_particles,
    type=[0] * num_free_particles
)

# Add wall particles
for i in range(N_particles_per_wall_per_dim):
    for j in range(N_particles_per_wall_per_dim):
        system.part.add(
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,safe_margin],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )
        system.part.add(
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,box_length+2*safe_margin],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )

# Add Walls 
wall = espressomd.shapes.Wall(normal=[0, 0, 1], dist=0)
system.constraints.add(shape=wall, particle_type=1)
wall = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(box_length+elc_gap+safe_margin))
system.constraints.add(shape=wall, particle_type=1)

# Define non-bonded interactions
system.non_bonded_inter[0, 0].wca.set_params(epsilon=1, sigma=1)
system.non_bonded_inter[0, 1].wca.set_params(epsilon=1, sigma=1)

# Add external E-field
E_field = espressomd.constraints.LinearElectricPotential(E=E_field)
system.constraints.add(E_field)

# Write initial configuration
vtf.writevsf(system, fp)
vtf.writevcf(system, fp)

part_vel = obs.FluxDensityProfile(ids=range(num_free_particles), n_x_bins =1,n_y_bins=1,n_z_bins=100,min_x=0,max_x=box_length,min_y=0,max_y=box_length,min_z=safe_margin,max_z=box_length+2*safe_margin)

print("Cooldown")
system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
system.integrator.run(1000)
vtf.writevcf(system, fp)

# Set up electrostatics with ELC
solver = espressomd.electrostatics.P3M(prefactor=bjerrum_length, accuracy=1e-3)

elc = espressomd.electrostatics.ELC(actor=solver, gap_size=elc_gap, maxPWerror=1e-2)
system.electrostatics.solver = elc

print("Equilibrate")
system.integrator.set_vv()
system.thermostat.set_lb(LB_fluid=lbf, seed=42,gamma = friction)
system.integrator.run(1000)

print("Run")
list=[]
for i in tqdm(range(1000)):
    system.integrator.run(100)
    list.append(part_vel.calculate())
    vtf.writevcf(system, fp)
fp.close()
list=np.array(list)
np.save("flux_density.npy",list)