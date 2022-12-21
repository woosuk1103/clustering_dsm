import numpy as np
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
    print("episode %d"%episode)

    while True: 

        ################################ 1               
        # make step anyway!
        print("step", step)

        input_row = input("Insert the row number you want to modify: ")
        input_col = input("Insert the column number you want to modify: ")

        row = int(input_row)
        col = int(input_col)

        action = row * 38 + col
        ################################ 1
        
        state, clustered_matrix, base_matrix, CE, name_list, reward, correlation_matrix, new_sorted_component_list, done, _ = env.step(action, correlation_matrix)
            
        # should set calculated figure to show the modularity
        print("CE:", CE)

        # visualization part        
        # plt.matshow(state)
        # plt.show()
        fig = px.imshow(state, labels=dict(x="components", y="components", color="I/F"), x = name_list, y = name_list)
        fig.show()

        
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



np_trajectories = np.array(trajectories, float)
np_trajectories_base = np.array(trajectories_base, float)
np_correlation_matrices = np.array(correlation_matrices, float)
np_new_sorted_component_lists = np.array(new_sorted_component_lists, dtype=object)

np.save("expert_trajectories", arr=np_trajectories)
np.save("expert_trajectories_base", arr=np_trajectories_base)
np.save("correlation_matrices", arr=np_correlation_matrices)
np.save("new_sorted_component_lists", arr=np_new_sorted_component_lists)