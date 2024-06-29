# Define the path to the .lammpstrj file
from MDAnalysis.core.universe import np
import matplotlib.pyplot as plt
import os
# l = np.linspace(0.2,10,99)
l=np.arange(0.1,1.4,0.1)
all_charges=[]
for i in l:
    if i%1<0.05:
        file_path = f'wallcharge_{i:.0f}.lammpstrj'
    else:
        file_path = f'wallcharge_{i:.1f}.lammpstrj'
    n_atoms = 1440
    # Initialize an empty list to store charges
    charges = []
    # print(f"Reading file: {file_path} at i = {i}")
    # Read the file if it exists
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Flag to indicate if we are in the ATOMS section
    in_atoms_section = False

    # Loop through each line in the file
    for line in lines:
        # Check if the line starts with "ITEM: ATOMS"
        if line.startswith("ITEM: ATOMS"):
            in_atoms_section = True
            continue
        
        # If in the ATOMS section, extract the charge
        if in_atoms_section:
            if line.startswith("ITEM:"):  # If a new ITEM section starts, stop reading ATOMS
                in_atoms_section = False
            # Split the line by spaces and get the charge (last element)
            else:
                charge = float(line.split()[-1])
                charges.append(charge)
    # Print the extracted charges
    print(f"read {len(charges)} charges from file: {file_path}")
    charges=np.array(charges)
    per_timestep =np.array([np.abs(charges[i:i+n_atoms])/2 for i in range(0, len(charges), n_atoms)])
    all_charges.append(np.mean(per_timestep)*n_atoms)
plt.scatter(l,all_charges,label = "simulation")
# plt.axhline(0.745,label = "analytical")
plt.xlabel(r"$\frac{\eta}{\AA^{-1}}$")
plt.ylabel(r"$\frac{\text{charge on one plate}}{e}$")
plt.legend()
# print(l[np.argmin(np.abs(all_charges-np.array([0.745]*len(all_charges))))])


plt.savefig('pot.png')