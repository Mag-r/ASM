import espressomd
import numpy as np
import espressomd.io.writer.vtf as vtf
import espressomd.shapes
import espressomd.lb
import espressomd.observables as obs
import espressomd.visualization
from tqdm import tqdm


#unit defined by elementary charge,k_b*300K as Energy, nm as length, 1 as mass
box_length = 16
safe_margin = 2.0
elc_gap=5.0
bjerrum_length = 0.7095
num_free_particles = 500
N_particles_per_wall_per_dim = 20
fluid_density = 26.18
viscosity = 0.25
friction = 15.0
E_field = [25.0,0,0]

def generateInitialConfiguration(box_length, safe_margin, elc_gap, bjerrum_length, num_free_particles, N_particles_per_wall_per_dim, fluid_density, viscosity, friction, E_field):
    system = espressomd.System(box_l=[box_length, box_length, box_length+2*safe_margin+elc_gap])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    system.periodicity = [True, True, True]

# Lattice-Boltzmann
    lbf = espressomd.lb.LBFluidWalberlaGPU(agrid=1.0,density = fluid_density, kinematic_viscosity = viscosity, tau=0.01)
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
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,0],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )
            system.part.add(
            pos=[ i * box_length / N_particles_per_wall_per_dim, j * box_length / N_particles_per_wall_per_dim,box_length],
            q=num_free_particles / (N_particles_per_wall_per_dim**2 * 2),
            m=1,
            type=1,
            fix=[True, True, True]
        )

# Add Walls 
    wall = espressomd.shapes.Wall(normal=[0, 0, 1], dist=0)
    system.constraints.add(shape=wall, particle_type=1)
    lbf.add_boundary_from_shape(shape=wall)
    wall = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(box_length))
    system.constraints.add(shape=wall, particle_type=1)
    lbf.add_boundary_from_shape(shape=wall)

# Define non-bonded interactions
    system.non_bonded_inter[0, 0].wca.set_params(epsilon=100, sigma=1)
    system.non_bonded_inter[0, 1].wca.set_params(epsilon=100, sigma=1.4)

# Add external E-field
    E_field = espressomd.constraints.LinearElectricPotential(E=E_field)
    system.constraints.add(E_field)

# Write initial configuration
    vtf.writevsf(system, fp)
    vtf.writevcf(system, fp)

    part_vel = obs.FluxDensityProfile(ids=range(num_free_particles), n_x_bins =1,n_y_bins=1,n_z_bins=1000,min_x=0,max_x=box_length,min_y=0,max_y=box_length,min_z=safe_margin,max_z=box_length+2*safe_margin)
    fluid_vel = obs.LBVelocityProfile(ids=range(num_free_particles), n_x_bins =1,n_y_bins=1,n_z_bins=1000,min_x=0,max_x=box_length,min_y=0,max_y=box_length,min_z=safe_margin,max_z=box_length+2*safe_margin,sampling_delta_x=5,sampling_delta_y=5,sampling_delta_z=5,allow_empty_bins=True,sampling_offset_z=0,sampling_offset_y=0,sampling_offset_x=0)
    dens = obs.DensityProfile(ids=range(num_free_particles), n_x_bins =1,n_y_bins=1,n_z_bins=1000,min_x=0,max_x=box_length,min_y=0,max_y=box_length,min_z=safe_margin,max_z=box_length+2*safe_margin)

    print("Cooldown")
    system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
    system.integrator.run(1000)
    vtf.writevcf(system, fp)

# Set up electrostatics with ELC
    solver = espressomd.electrostatics.P3M(prefactor=bjerrum_length, accuracy=1e-3)

    elc = espressomd.electrostatics.ELC(actor=solver, gap_size=elc_gap-2*safe_margin, maxPWerror=1e-2)
    system.electrostatics.solver = elc

    print("Equilibrate")
    system.integrator.set_vv()
    system.thermostat.set_lb(LB_fluid=lbf, seed=42,gamma = friction)
    system.integrator.run(3000)
    return system,fp,part_vel,fluid_vel,dens




def run_simulation(system,fp,part_vel,fluid_vel,dens):
    print("Run")
    part_vel_list=[]
    lb_vel=[]
    dens_list=[]
    for i in tqdm(range(10000)):
        try:
            system.integrator.run(100)
            part_vel_list.append(part_vel.calculate())
            lb_vel.append(fluid_vel.calculate())
            dens_list.append(dens.calculate())
            # print(system.analysis.linear_momentum(include_lbfluid=False))
            vtf.writevcf(system, fp)
        except:
            fp = open("fatal.vtf", mode="w+t")
            vtf.writevsf(system, fp)
            vtf.writevcf(system, fp)
            fp.close()
            raise ValueError("Fatal error")
        
    fp.close()
    part_vel_list=np.array(part_vel_list)
    lb_vel=np.array(lb_vel)
    dens_list=np.array(dens_list)
    np.save("part_vel.npy",part_vel_list)
    np.save("dens.npy",dens_list)
    np.save("lb_vel.npy",lb_vel)

if __name__ == "__main__":
    system, fp, part_vel, fluid_vel, dens = generateInitialConfiguration(box_length, safe_margin, elc_gap, bjerrum_length, num_free_particles, N_particles_per_wall_per_dim, fluid_density, viscosity, friction, E_field)
    run_simulation(system,fp,part_vel,fluid_vel,dens)
