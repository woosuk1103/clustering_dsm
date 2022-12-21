import numpy as np
import random
import math

import matplotlib.pyplot as plt
import plotly.express as px

from environment import Moduleviser
env = Moduleviser()

trajectories = []
trajectories_base = []
correlation_matrices = []
new_sorted_component_lists = []

correlation_matrix = np.zeros((38,38))

for episode in range(10): # n_trajectories : 10
    trajectory = []
    trajectory_base = []
    step = 0

    env.reset()
    # print("episode %d"%episode)


    while True: 

        # make optimal step for each timestep
        # print("step", step)

        action = -1
        action_reasonable = False
        while not action_reasonable:
            action = random.randint(0, 1443)
            row, col = divmod(action, 38)
            
            # avoid diagonal terms to be modified
            if math.sqrt(action) == int(math.sqrt(action)):
                continue
            
            # let friendly components aggregated
            if (env.state[row][col] == 0) and (env.base_matrix[row][col] == 0) and (env.opt_base_matrix[row][col] == 1):
                action_reasonable = True
            
            # let adversarial components divided
            if (env.state[row][col] == 1) and (env.base_matrix[row][col] == 1) and (env.opt_base_matrix[row][col] == 0):
                action_reasonable = True

        # check = np.ones(shape=(1444,), dtype=np.int32)
        
        state, clustered_matrix, base_matrix, CE, name_list, reward, correlation_matrix, new_sorted_component_list, done, _ = env.step(action, correlation_matrix)
        
        # should set calculated figure to show the modularity
        # print("CE:", CE)

        # visualization part        
        # plt.matshow(state)
        # plt.show()
        # fig = px.imshow(state, labels=dict(x="components", y="components", color="I/F"), x = name_list, y = name_list)
        # fig.show()

        
        # saving part
        trajectory.append(state)
        trajectory.append(action)
        trajectory_base.append(base_matrix)
        
             
        if done == True:
            break

        step += 1

    trajectories.append(trajectory)
    trajectories_base.append(trajectory)
    correlation_matrices.append(correlation_matrix)
    new_sorted_component_lists.append(new_sorted_component_list)



np_trajectories = np.array(trajectories, dtype=object)
np_trajectories_base = np.array(trajectories_base, dtype=object)
np_correlation_matrices = np.array(correlation_matrices, dtype=object)
np_new_sorted_component_lists = np.array(new_sorted_component_lists, dtype=object)

print(np_trajectories)
print(np_trajectories_base)

np.save("expert_trajectories", arr=np_trajectories)
np.save("expert_trajectories_base", arr=np_trajectories_base)
np.save("correlation_matrices", arr=np_correlation_matrices)
np.save("new_sorted_component_lists", arr=np_new_sorted_component_lists)