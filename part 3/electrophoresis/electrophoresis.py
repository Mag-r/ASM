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
import logging
import multiprocessing

box_length = 16
bjerrum_length = 0.7095
num_free_particles = 50
N_particles_per_wall_per_dim = 20
fluid_density = 26.18
viscosity = 0.25
friction = 15.0
E_field = [25.0,0,0]
logger=logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

def createElectrophoresisSystem(system, box_length, bjerrum_length, fluid_density, viscosity, E_field, polymer_length):
    system.time_step = 0.0001
    system.cell_system.skin = 0.4
    system.periodicity = [True, True, True]

    lbf = espressomd.lb.LBFluidWalberlaGPU(agrid=1.0,density = fluid_density, kinematic_viscosity = viscosity, tau=0.01)
    system.lb=lbf

# Add E-field
    E_field = espressomd.constraints.LinearElectricPotential(E=E_field)
    system.constraints.add(E_field)

# add polymer
    logging.info("Adding polymer with length %d", polymer_length)
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

    system.non_bonded_inter[0, 0].wca.set_params(epsilon=1, sigma=1)
    system.non_bonded_inter[0, 1].wca.set_params(epsilon=1, sigma=1)
# Write initial configuration
    fp = open(f"electro_polymer_{polymer_length}.vtf", mode="w+t")
    vtf.writevsf(system, fp)
    vtf.writevcf(system, fp)

# Add observables
    vel=espressomd.observables.ParticleVelocities(ids=range(system.part.highest_particle_id))
    vel_com=espressomd.observables.ComVelocity(ids=range(polymer_length))
    corr=espressomd.accumulators.Correlator(obs1=vel,obs2=vel_com,corr_operation="scalar_product",delta_N=1,tau_max=100*system.time_step,tau_lin=1)
    system.auto_update_accumulators.add(corr)

    print("Cooldown")
    system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
    system.integrator.run(1000)
    vtf.writevcf(system, fp)

# Set up electrostatics
    logging.info("Setting up electrostatics")
    solver = espressomd.electrostatics.P3M(prefactor=bjerrum_length, accuracy=1e-3)
    system.electrostatics.solver = solver
    return fp,system,vel,vel_com,corr

def simulateElectrophoresis(system, box_length, bjerrum_length, fluid_density, viscosity, E_field, polymer_length):
    fp, system, vel, vel_com, corr = createElectrophoresisSystem(system, box_length, bjerrum_length, fluid_density, viscosity, E_field, polymer_length)
    part_vel_list=[]
    com_vel_list=[]
    mu_list = []
    logging.info("Simulating electrophoresis with polymer length %d", polymer_length)
    for i in tqdm(range(10000)):
        system.integrator.run(100)
        vtf.writevcf(system, fp)
        part_vel_list.append(vel.calculate())
        com_vel_list.append(vel_com.calculate())
        mu_list.append(corr.result())
        if i % 1000 == 0:
            logging.info("save data for polymer length %d", polymer_length)


    fp.close()
    np.save(f"vel_{polymer_length}.npy",part_vel_list)
    np.save(f"com_vel_{polymer_length}.npy",com_vel_list)
    np.save(f"mu_{polymer_length}.npy",mu_list)
    logging.info("Simulation for polymer length %d completed", polymer_length)

if __name__ == "__main__":
    #parallel execute the simulation for different polymer lenghts
    polymer_lengths=np.arange(2,15)
    system = espressomd.System(box_l=[box_length, box_length, box_length])
    for polymer_length in polymer_lengths:
        simulateElectrophoresis(system,box_length, bjerrum_length, fluid_density, viscosity, E_field, polymer_length)
        system.part.clear()
        system.constraints.clear()
        system.bonded_inter.clear()
        system.electrostatics.clear()
        system.auto_update_accumulators.clear()

    # simulateElectrophoresis(box_length, bjerrum_length, fluid_density, viscosity, E_field, polymer_length)