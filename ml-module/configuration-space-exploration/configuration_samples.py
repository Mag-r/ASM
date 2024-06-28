import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
pos_for_learn=[]
with open('simulation_data.pkl', 'rb') as f:
    data = pickle.load(f)
    #6,2000,64,3
    forces = np.linalg.norm(data["all"]['force'], axis=3)

    min=1
    max=0
    for i in range(3):
        for j in range(2000):
            for k in range(64):
                if forces[i,j,k] < min:
                    min = forces[i,j,k]
                elif forces[i,j,k] > max:
                    max = forces[i,j,k]
    N=2500
    eps=5E-7
    pos_for_learn=[]
    i=0
    while(len(pos_for_learn)<N):
        random_force=np.random.uniform(min,max)
        i+=1
        for j in range(3):
            for l in range(2000):
                for k in range(64):
                    if forces[j,l,k] < random_force+eps and forces[j,l,k] > random_force-eps:
                        pos_for_learn.append(data["all"]['trajectories'][j][l])
                        print(f"len(pos_for_learn): {len(pos_for_learn)} iteration: {i}")

    
with open('selected_configurations.pkl', 'wb') as f:
    pickle.dump(pos_for_learn, f)


