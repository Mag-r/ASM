# %%
# Linalg support
import numpy as onp

# Jax imports
import jax
from jax import lax
from jax import jit, vmap, grad
import jax.numpy as np
from jax import random
from functools import partial
from tqdm import tqdm
# ML imports
import optax

# Jax-md imports
from jax_md import energy, space, simulate, quantity, minimize

# Plotting.
import matplotlib.pyplot as plt

# Tracking
from rich.progress import track
import pickle

# %%
# Load the model params for one functional

xc="pbe"

params = None
with open(f"/tikhome/mgern/Desktop/AdvancedSimMethods/ml-module/mlpes/params_{xc}.pkl", "rb") as f:
    params = pickle.load(f)
n_particles = [100, 1000,]  # Scale to required sizes
time_step = 0.01
n_steps = 10000  # Scale up for better statistics

# %%
def run_md(model_params, n_particles: int, temperature, time_step, n_steps):
    """
    Run molecular dynamics simulation.
    """
    # Parameters to correctly instantiate the simulation box.
    rho_argon = 1.40  # g/cm³
    mass_argon = 39.95  # g/mol
    avogadro = 6.022e23  # atoms/mol
    kb = 8.617e-5  # Boltzmann constant

    # Calculate number density in atoms/cm³
    n_argon = (rho_argon * avogadro) / mass_argon

    # Continuing from the previous calculation
    n_argon_cm3 = n_argon  # This is in atoms/cm³

    # Conversion factor from cm³ to Å³
    conversion_factor = 1e24  # (10^8)^3

    # Convert number density to atoms/Å³ for use in simulations
    n_argon_A3 = n_argon_cm3 / conversion_factor

    # Get the initial conditions of the simulation
    n_particles = 64  # Increase this to the required number of particles
    dimension = 3

    # Compute the box size
    box_size = quantity.box_size_at_number_density(n_particles, n_argon_A3, dimension)

    # Get displacement function
    displacement, shift = space.periodic(box_size)

    # Set initial positions of particles.
    key = random.PRNGKey(0)
    initial_positions = random.uniform(
        key, (n_particles, dimension), minval=0.0, maxval=box_size, dtype=np.float64
    )

    # Define a random graph neural network. This must be changed for the BP network.
    neighbor_fn, init_fn, energy_fn = energy.graph_network_neighbor_list(
        displacement, box_size, r_cutoff=4.0, dr_threshold=0.0, n_recurrences=2, mlp_sizes=(12, 12))

    # Neighbour list computation, should improve performance.
    neighbor = neighbor_fn.allocate(initial_positions, extra_capacity=6)

    print('Allocating space for at most {} edges'.format(neighbor.idx.shape[1]))

    baked_e_function = partial(energy_fn, model_params)

    # Energy minimization

    # Prepare the FIRE minimization
    fire_init, fire_apply = minimize.fire_descent(baked_e_function, shift)
    fire_apply = jit(fire_apply)
    fire_state = fire_init(initial_positions, neighbor=neighbor)

    # Perform the minimization
    minimisation_energy = []

    @jit
    def fire_sim(state, nbrs):
        def step(i, state_nbrs):
            state, nbrs = state_nbrs
            nbrs = nbrs.update(state.position)
            return fire_apply(state, neighbor=nbrs), nbrs
        return lax.fori_loop(0, 25, step, (state, nbrs))

    # Adjust the minimisation steps until it converges
    for i in tqdm(range(1000)):
        fire_state, neighbor = fire_sim(fire_state, neighbor)
        minimisation_energy += [baked_e_function(fire_state.position, neighbor=neighbor)]

    # Run the simulation

    init_fn, apply_fn = simulate.nve(baked_e_function, shift, time_step)
    state = init_fn(key, fire_state.position, kT = kb * temperature,neighbor=neighbor)

    # Add your analysis code from the previous exercise here.
    def compute_potential_energy(state, neighbor_):
        """
        Compute the potential energy of the system.

        Parameters
        ----------
            The current state of the simulation.
        """
        # Implement the function
        potential_energy = energy_fn(state.position, neighbor_)
        return potential_energy

    def compute_kinetic_energy(state):
        """
        Compute the kinetic energy of the system.

        Parameters
        ----------
        state 
            The current state of the simulation.
        """
        # Implement the function
        kinetic_energy = quantity.kinetic_energy(momentum=state.momentum)
        return kinetic_energy
    
    @jit
    def prod_sim(state, nbrs):
        def step(i, state_nbrs):
            state, nbrs = state_nbrs
            nbrs = nbrs.update(state.position)
            return apply_fn(state, neighbor=nbrs), nbrs
        return lax.fori_loop(0, 25, step, (state, nbrs))
    

    potential_energy = []
    kinetic_energy = []
    trajectory = []
    for step in tqdm(range(n_steps)):
        state, neighbor = prod_sim(state, neighbor)  # Update the state
        
        # Record at certain intervals
        if step % 50 == 0:
            # Compute the potential and kinetic energy
            potential_energy.append(compute_potential_energy(state, neighbor))
            kinetic_energy.append(compute_kinetic_energy(state))

            # Add some trajectory recording here
            trajectory.append(state.position)

    return trajectory, potential_energy, kinetic_energy

# %%
# Run the simulations

# Compute RDFS

# Plot the results
trajectory,pot, kin =run_md(params, n_particles[0], 85, time_step, n_steps) 
print("Done")
np.save(f'trajectory_{xc}.npy', trajectory)




