# %%
# Linalg modules
import numpy as np

# Plotting modules
import matplotlib.pyplot as plt

# ASE modules
import ase
import pickle
from tqdm import tqdm
from ase.calculators.cp2k import CP2K
import os

# %%
os.environ["OMP_NUM_THREADS"] = "4"
frame=0
with open("/tikhome/mgern/Desktop/AdvancedSimMethods/ml-module/configuration-space-exploration/selected_configurations.pkl",'rb') as f:
     # Load one data frame here
    data = pickle.load(f)
    frame=data[0]

# %%
def build_cp2k_iniut(cutoff: int) -> str:
    """
    Build the CP2K input file for varying cutoff.

    Parameters
    ----------
    cutoff : int
        The cutoff energy in Rydberg.
    """
    return f"""
        &FORCE_EVAL
        &DFT
        # &MGRID
        #             CUTOFF {cutoff}
        # &END MGRID
        &SCF
                    SCF_GUESS RESTART
                    IGNORE_CONVERGENCE_FAILURE
        &END SCF
        &END DFT
        &END FORCE_EVAL
    """

# %%
# This is required for ASE to work with CP2K. If you are working from your own computer, you may need to change this path.
CP2K.command = "/group/allatom/cp2kv2024.1/exe/local/cp2k_shell.psmp"

cutoffs = np.linspace(20,500,100)  # Fill this in

energies = []
forces = []
xc="BEEF"
for cutoff in tqdm(cutoffs):
    cp2k_input = build_cp2k_iniut(cutoff)
    calculator = CP2K(xc=xc, inp=cp2k_input, cutoff=cutoff)  # Adapt this functional for your system.
    # Create the atoms object
    atoms = ase.Atoms("Ar64", positions=frame, pbc=[1, 1, 1], cell=np.ones(3) * 14.474693)
    energies.append(calculator.get_potential_energy(atoms))


# %%
# Plot energy and force as a function of cutoff
np.savetxt(f"energies_{xc}.txt", energies)
plt.plot(cutoffs, energies, "o-")
plt.xlabel("Cutoff (eV)")
plt.ylabel("Energy (eV)")
# fig, ax = plt.subplots(2, 1, figsize=(6, 8))
# ax[0].plot(cutoffs, energies, "o-")
# ax[0].set_xlabel("Cutoff (Rydberg)")
# ax[0].set_ylabel("Energy (eV)")
# ax[1].plot(cutoffs, forces, "o-")
# ax[1].set_xlabel("Cutoff (Rydberg)")
# ax[1].set_ylabel("Force (eV/Angstrom)")
plt.savefig(f"convergence_{xc}.png")
plt.show()



