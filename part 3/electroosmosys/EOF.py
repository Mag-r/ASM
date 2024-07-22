import espressomd
import numpy as np
import espressomd.io.writer.vtf as vtf
import espressomd.shapes
import espressomd.lb
import espressomd.observables as obs
import espressomd.visualization
from tqdm import tqdm
import logging
import signal
import os
import espressomd.checkpointing

#unit defined by elementary charge,k_b*300K as Energy, nm as length, 1 as mass
box_length = 16.0
elc_gap=5.0
bjerrum_length = 0.7095
num_free_particles = 128
N_particles_per_wall_per_dim = 8
fluid_density = 26.18
viscosity = 0.25
friction = 15.0
E_field = [25.0,0,0]
safe_margin=1.5

logger=logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

def generateInitialConfiguration(box_length, safe_margin, elc_gap, bjerrum_length, num_free_particles, N_particles_per_wall_per_dim, fluid_density, viscosity, friction, E_field):
    system = espressomd.System(box_l=[box_length, box_length, box_length+2*safe_margin+elc_gap])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    system.periodicity = [True, True, True]

# Lattice-Boltzmann
    lbf = espressomd.lb.LBFluidWalberla(agrid=1,density = fluid_density, kinematic_viscosity = viscosity, tau=0.01)



# Add free particles

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
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,safe_margin],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )
            system.part.add(
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,box_length-safe_margin],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )

# Add Walls 
    wall = espressomd.shapes.Wall(normal=[0, 0, 1], dist=safe_margin)
    system.constraints.add(shape=wall, particle_type=1)
    lbf.add_boundary_from_shape(shape=wall)
    wall = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(box_length-safe_margin))
    system.constraints.add(shape=wall, particle_type=1)
    lbf.add_boundary_from_shape(shape=wall)
    system.lb=lbf
# Define non-bonded interactions
    system.non_bonded_inter[0, 0].wca.set_params(epsilon=1, sigma=1)
    system.non_bonded_inter[0, 1].wca.set_params(epsilon=1, sigma=1)

# Add external E-field
    E_field = espressomd.constraints.LinearElectricPotential(E=E_field)
    system.constraints.add(E_field)

# Write initial configuration
    fp = open("trajectory.vtf", mode="w+t")
    vtf.writevsf(system, fp)
    vtf.writevcf(system, fp)

    # Set up observables
    # part_vel = obs.FluxDensityProfile(ids=range(num_free_particles), n_x_bins =1,n_y_bins=1,n_z_bins=1000,min_x=0,max_x=box_length,min_y=0,max_y=box_length,min_z=safe_margin,max_z=box_length+2*safe_margin)
    part_vel = obs.ParticleVelocities(ids=range(num_free_particles))
    pos=obs.ParticlePositions(ids=range(num_free_particles))
    fluid_vel = obs.LBVelocityProfile(ids=range(num_free_particles), n_x_bins =1,n_y_bins=1,n_z_bins=50,min_x=0,max_x=box_length,min_y=0,max_y=box_length,min_z=0,max_z=box_length,sampling_delta_x=1,sampling_delta_y=1,sampling_delta_z=1/10,allow_empty_bins=True,sampling_offset_z=0,sampling_offset_y=0,sampling_offset_x=0)
    # dens = obs.DensityProfile(ids=range(num_free_particles), n_x_bins =1,n_y_bins=1,n_z_bins=1000,min_x=0,max_x=box_length,min_y=0,max_y=box_length,min_z=safe_margin,max_z=box_length+2*safe_margin)

    logging.info("Remove overlaps")
    system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
    system.integrator.run(1000)
    vtf.writevcf(system, fp)


# Set up electrostatics with ELC
    logging.info("Set up electrostatics with ELC")
    solver = espressomd.electrostatics.P3M(prefactor=bjerrum_length, accuracy=1e-3)
    elc = espressomd.electrostatics.ELC(actor=solver, gap_size=elc_gap, maxPWerror=1e-4)
    system.electrostatics.solver = elc

    logging.info("Equilibrate system")
    system.integrator.set_vv()
    system.thermostat.set_lb(LB_fluid=lbf, seed=42,gamma = friction)
    system.integrator.run(3000)
    return system,fp,part_vel,fluid_vel,pos




def run_simulation(system,fp,part_vel,fluid_vel,pos):
    logging.info("Run simulation")
    part_vel_list=np.load("part_vel.npy",allow_pickle=True) if os.path.exists("part_vel.npy") else np.array([part_vel.calculate()])
    lb_vel=np.load("lb_vel.npy",allow_pickle=True) if os.path.exists("lb_vel.npy") else np.array([fluid_vel.calculate()])
    pos_list=np.load("pos.npy",allow_pickle=True) if os.path.exists("pos.npy") else np.array([pos.calculate()])
    for i in tqdm(range(20000)):
        system.integrator.run(500)
        part_vel_list=np.append(part_vel_list,[part_vel.calculate()],axis=0)
        lb_vel=np.append(lb_vel,[fluid_vel.calculate()],axis=0)
        pos_list=np.append(pos_list,[pos.calculate()],axis=0)
        vtf.writevcf(system, fp)
        if i %100==0:
            logging.info("Save data") 
            np.save("part_vel.npy",part_vel_list)
            np.save("pos.npy",pos_list)
            np.save("lb_vel.npy",lb_vel)
    fp.close()
    part_vel_list=np.array(part_vel_list)
    lb_vel=np.array(lb_vel)
    pos_list=np.array(pos_list)
    np.save("part_vel.npy",part_vel_list)
    np.save("pos.npy",pos_list)
    np.save("lb_vel.npy",lb_vel)


if __name__ == "__main__":
    system, fp, part_vel, fluid_vel, pos = generateInitialConfiguration(box_length, safe_margin, elc_gap, bjerrum_length, num_free_particles, N_particles_per_wall_per_dim, fluid_density, viscosity, friction, E_field)
    run_simulation(system,fp,part_vel,fluid_vel,pos)
