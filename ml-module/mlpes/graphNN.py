# %%
# Linalg support
import numpy as onp

# Jax imports
import jax
from jax import lax
from jax import jit, vmap, grad
import jax.numpy as np
from jax import random

# ML imports
import optax

# Jax-md imports
from jax_md import energy, space, simulate, quantity

# Plotting.
import matplotlib.pyplot as plt

from tqdm import tqdm
import pickle

# %%
# Load up your saved data here.
positions=0
with open("/tikhome/mgern/Desktop/AdvancedSimMethods/ml-module/configuration-space-exploration/selected_configurations.pkl",'rb') as f:
     # Load one data frame here
    data = pickle.load(f)
    positions=np.array(data)
xc="LDA"
print(f"Using {xc} functional.")
energies = np.load(f"/tikhome/mgern/Desktop/AdvancedSimMethods/ml-module/ab-initio-calculations/bulk_energy_{xc}.npy")
forces = np.load(f"/tikhome/mgern/Desktop/AdvancedSimMethods/ml-module/ab-initio-calculations/bulk_forces_{xc}.npy")

# %%
# Split the data into training and testing sets.
n_train_points = 2300
train_indices = onp.random.choice(np.arange(energies.shape[0]), n_train_points, replace=False)
test_indices = np.array([i for i in np.arange(energies.shape[0]) if i not in train_indices])

train_positions = positions[train_indices,:,:]
train_energies = energies[train_indices]
train_forces = forces[train_indices]
test_positions = positions[test_indices]
test_energies = energies[test_indices]
test_forces = forces[test_indices]

# %%
# Normalize the energies.
energy_mean = np.mean(train_energies)
energy_std = np.std(train_energies)

train_energies = (train_energies - energy_mean) / energy_std
test_energies = (test_energies - energy_mean) / energy_std

# %%
box_size = 14.474693  # The size of the simulation region, adjust if necessary.
displacement, shift = space.periodic(box_size)

# %%
# Define the graph network.
neighbor_fn, init_fn, energy_fn = energy.graph_network_neighbor_list(
    displacement, box_size, r_cutoff=4.0, dr_threshold=0.0, n_recurrences=2, mlp_sizes=(12, 12))

# Neighbour list computation, should improve performance.

neighbor = neighbor_fn.allocate(train_positions[0], extra_capacity=6)

print('Allocating space for at most {} edges'.format(neighbor.idx.shape[1]))

# %%
@jit
def train_energy_fn(params, R):
  _neighbor = neighbor.update(R)
  return energy_fn(params, R, _neighbor)

# Vectorize over states, not parameters.
vectorized_energy_fn = vmap(train_energy_fn, (None, 0))

grad_fn = grad(train_energy_fn, argnums=1)
force_fn = lambda params, R, **kwargs: -grad_fn(params, R)
vectorized_force_fn = vmap(force_fn, (None, 0))

# %%
# Initialize the neural network parameters
key = random.PRNGKey(0)
params = init_fn(key, train_positions[0], neighbor)

#try to read params from file
try:
  with open(f"params_{xc}.pkl", "rb") as f:
    params = pickle.load(f)
    print("Parameters loaded.")
except:
  print("No parameters found, using random initialization.")

# %%
# Look at the priors over the data before training. What do you see?

predicted_energies = vmap(train_energy_fn, (None, 0))(params, train_positions)
predicted_forces = vectorized_force_fn(params, test_positions)

fig, ax = plt.subplots(1, 2, figsize=(8, 8))

# Energy priors
ax[0].plot(train_energies, predicted_energies, 'o',label="predicted energies")
ax[0].plot(train_energies, train_energies, 'k--', label="true energies")
ax[0].set_xlabel("Energy")
ax[0].set_ylabel("Energy")
ax[0].set_title("Energy priors")
ax[0].legend()
# Force priors
ax[1].plot(test_forces.flatten(), predicted_forces.flatten(), 'o',label="predicted forces")
ax[1].plot(test_forces.flatten(), test_forces.flatten(), 'k--', label="true forces")
ax[1].set_xlabel("force")
ax[1].set_ylabel("force")
ax[1].set_title("Force priors")
ax[1].legend()
# plt.show()
plt.savefig(f"priors_{xc}.png")
print("Priors saved.")
# %%
# Define the loss functions.
@jit
def energy_loss(params, R, energy_targets):
  return np.mean((vectorized_energy_fn(params, R) - energy_targets) ** 2)

@jit
def force_loss(params, R, force_targets):
  dforces = vectorized_force_fn(params, R) - force_targets
  return np.mean(np.sum(dforces ** 2, axis=(1, 2)))

@jit
def loss(params, R, targets):
  return energy_loss(params, R, targets[0]) #+ force_loss(params, R, targets[1])

# %%
opt = optax.chain(
  optax.clip_by_global_norm(1.0), optax.adam(1e-3)
)

@jit
def update_step(params, opt_state, R, labels):
  updates, opt_state = opt.update(grad(loss)(params, R, labels),
                                  opt_state)
  return optax.apply_updates(params, updates), opt_state

@jit
def update_epoch(params_and_opt_state, batches):
  def inner_update(params_and_opt_state, batch):
    params, opt_state = params_and_opt_state
    b_xs, b_labels = batch

    return update_step(params, opt_state, b_xs, b_labels), 0
  return lax.scan(inner_update, params_and_opt_state, batches)[0]

# %%
dataset_size = train_positions.shape[0]
batch_size = 32

lookup = onp.arange(dataset_size)
onp.random.shuffle(lookup)

@jit
def make_batches(lookup):
  batch_Rs = []
  batch_Es = []
  batch_Fs = []

  for i in range(0, len(lookup), batch_size):
    if i + batch_size > len(lookup):
      break

    idx = lookup[i:i + batch_size]

    batch_Rs += [train_positions[idx]]
    batch_Es += [train_energies[idx]]
    batch_Fs += [train_forces[idx]]

  return np.stack(batch_Rs), np.stack(batch_Es), np.stack(batch_Fs)

batch_Rs, batch_Es, batch_Fs = make_batches(lookup)

# %%
train_epochs = 5000  # Adjust as necessary.

opt_state = opt.init(params)

train_energy_error = []
test_energy_error = []
print("Starting training.")

for iteration in tqdm(range(train_epochs)):
  train_energy_error += [float(np.sqrt(energy_loss(params, batch_Rs[0], batch_Es[0])))]
  test_energy_error += [float(np.sqrt(energy_loss(params, test_positions, test_energies)))]
 
  params, opt_state = update_epoch((params, opt_state), 
                                   (batch_Rs, (batch_Es, batch_Fs)))

  onp.random.shuffle(lookup)
  batch_Rs, batch_Es, batch_Fs = make_batches(lookup)
print("Training complete.")
# %%
with open(f"params_{xc}.pkl", "wb") as f:
    pickle.dump(params, f)
    print("Parameters saved.")

fig, ax = plt.subplots(1, 3,figsize=(12, 8))

predicted_energies = vectorized_energy_fn(params, test_positions)
ax[0].plot(test_energies, predicted_energies, 'o', label="predicted energies")
ax[0].plot(test_energies, test_energies, '--', label="true energies")
ax[0].set_xlabel("Energy")
ax[0].set_ylabel("Energy")
ax[0].set_title("Energy posterior")
ax[0].legend()

predicted_forces = vectorized_force_fn(params, test_positions)
ax[1].plot(test_forces.reshape((-1,)),
         predicted_forces.reshape((-1,)), 
         'o', label="predicted forces")
ax[1].plot(
    test_forces.reshape((-1,)),
    test_forces.reshape((-1,)),label="true forces",
)
ax[1].set_xlabel("Force")
ax[1].set_ylabel("Force")
ax[1].set_title("Force posterior")
ax[1].legend()

ax[2].plot(train_energy_error, label="train energy error")
ax[2].plot(test_energy_error, label="test energy error")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("MSE")
ax[2].set_title("Training error")
ax[2].legend()

plt.savefig(f"posterior_{xc}.png")
print("Posterior saved.")

# %%
def compute_energy_metrics(energy_predictions: np.ndarray, energy_targets: np.ndarray):
    """
    Compute the energy metrics.

    Parameters:
    -----------
    energy_predictions: np.ndarray
        The predicted energies.
    energy_targets: np.ndarray
        The target energies.
    
    Returns
    -------
    mae: float
        The mean absolute error per atom in the system
    pearson: float
        The Pearson correlation coefficient.
    """
      
    mae = np.mean(np.abs(energy_predictions - energy_targets))
    pearson = np.corrcoef(energy_predictions, energy_targets)[0, 1]
    l4 = np.mean(np.abs(energy_predictions - energy_targets) ** 4)

    return mae, pearson, l4

def compute_force_metrics(force_predictions: np.ndarray, force_targets: np.ndarray):
    """
    Compute the energy metrics.

    Parameters:
    -----------
    force_predictions: np.ndarray
        The predicted forces.
    force_targets: np.ndarray
        The target forces.
    
    Returns
    -------
    mae: float
        The mean absolute error per atom in the system
    pearson: float
        The Pearson correlation coefficient.
    l4: float
        The L4 error.
    """
        
    mae = np.mean(np.abs(force_predictions - force_targets))
    pearson = np.corrcoef(force_predictions.flatten(), force_targets.flatten())[0, 1]
    l4 = np.mean(np.abs(force_predictions - force_targets) ** 4)
  
    return mae, pearson, l4

# %%
# Save your model parameters.



