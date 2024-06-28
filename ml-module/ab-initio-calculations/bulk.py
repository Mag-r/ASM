# %%
# Linalg modules
import numpy as np

# Plotting modules
import matplotlib.pyplot as plt

# ASE modules
import ase
from ase.calculators.cp2k import CP2K
import pickle
from tqdm import tqdm
import os
os.environ["OMP_NUM_THREADS"] = "4"
# %%
trajectory = 0  # Load your data here
with open("/tikhome/mgern/Desktop/AdvancedSimMethods/ml-module/configuration-space-exploration/selected_configurations.pkl",'rb') as f:
     # Load one data frame here
    trajectory = pickle.load(f)

# %%
# This is required for ASE to work with CP2K. If you are working from your own computer, you may need to change this path.
CP2K.command = "/group/allatom/cp2kv2024.1/exe/local/cp2k_shell.psmp"

# This will make things faster by restarting wavefunctions guesses from past calculations.
restart_inp = """
&FORCE_EVAL
&DFT
&SCF
            SCF_GUESS RESTART
            IGNORE_CONVERGENCE_FAILURE
&END SCF
&END DFT
&END FORCE_EVAL
"""
xc="BEEF"
print(xc)
calculator = CP2K(xc=xc, inp=restart_inp, cutoff =250)  # Change this to fit your DFT settings.

# %%
energies = []
forces = []
atoms_objects = []

# In principle, this can be parallelized. I will leave this as an exercise for you.
for frame in tqdm(trajectory):
    atoms = ase.Atoms("Ar64", positions=frame, pbc=[1, 1, 1], cell=np.ones(3) * 14.474693)
    energy = calculator.get_potential_energy(atoms)
    force = calculator.get_forces(atoms)

    energies.append(energy)
    forces.append(force)
    atoms_objects.append(atoms)

# %%
# Save the data
np.save(f"bulk_energy_{xc}.npy", energies)
np.save(f"bulk_forces_{xc}.npy", forces)
np.save(f"bulk_atoms_{xc}.npy", atoms_objects)