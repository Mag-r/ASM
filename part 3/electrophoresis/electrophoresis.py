import espressomd
import numpy as np
import espressomd.io.writer.vtf as vtf
import espressomd.shapes
import espressomd.lb
import espressomd.observables as obs
import espressomd.polymer
import espressomd.interactions
import espressomd.accumulators
from tqdm import tqdm

box_length = 16
bjerrum_length = 0.7095
num_free_particles = 500
N_particles_per_wall_per_dim = 20
fluid_density = 26.18
viscosity = 0.25
friction = 15.0
E_field = [25.0,0,0]
polymer_length=10


fp = open("electro_polymer.vtf", mode="w+t")

system = espressomd.System(box_l=[box_length, box_length, box_length])
system.time_step = 0.00001
system.cell_system.skin = 0.4
system.periodicity = [True, True, True]

lbf = espressomd.lb.LBFluidWalberlaGPU(agrid=1.0,density = fluid_density, kinematic_viscosity = viscosity, tau=0.01)
system.lb=lbf

# Add E-field
E_field = espressomd.constraints.LinearElectricPotential(E=E_field)
system.constraints.add(E_field)

# add polymer
fene =espressomd.interactions.FeneBond(k=10,d_r_max=2)
system.bonded_inter.add(fene)
polymer_positions = espressomd.polymer.linear_polymer_positions(n_polymers=1,beads_per_chain=polymer_length,bond_length=1.0,seed=42)
for positions in polymer_positions:
    monomers = system.part.add(pos=positions,q=[-1]*polymer_length,m=[1]*polymer_length,type=[0]*polymer_length)
    previous_part = None
    for part in monomers:
        if not previous_part is None:
            part.add_bond((fene, previous_part))
        previous_part = part

# Add Counterions
system.part.add(pos=np.random.uniform(low=0, high=box_length, size=(polymer_length, 3)),q=[1]*polymer_length,m=[1]*polymer_length,type=[1]*polymer_length)
# Write initial configuration
vtf.writevsf(system, fp)
vtf.writevcf(system, fp)

# Add observables
vel=espressomd.observables.ParticleVelocities(ids=range(polymer_length))
vel_com=espressomd.observables.ComVelocity(ids=range(polymer_length))
corr=espressomd.accumulators.Correlator(vel,vel_com,"scalar_product",1,100*system.time_step)
system.auto_update_accumulators.add(corr)

print("Cooldown")
system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
system.integrator.run(1000)
vtf.writevcf(system, fp)

# Set up electrostatics with ELC
solver = espressomd.electrostatics.P3M(prefactor=bjerrum_length, accuracy=1e-3)
system.electrostatics.solver = solver

part_vel_list=[]
com_vel_list=[]
for i in tqdm(range(10000)):
    system.integrator.run(100)
    vtf.writevcf(system, fp)
    part_vel_list.append(vel.evaluate(system))
    com_vel_list.append(vel_com.evaluate(system))
fp.close()
np.save("vel.npy",part_vel_list)
np.save("com_vel.npy",com_vel_list)