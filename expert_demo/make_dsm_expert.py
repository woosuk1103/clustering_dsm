import numpy as np
from environment import Moduleviser
import plotly.express as px
import pandas as pd

env = Moduleviser()

trajectories = []
correlation_matrices = []
new_sorted_component_lists = []
episode_step = 0

correlation_matrix = np.zeros((38,38))

colorscale = [[0, '#272D31'],[.5, '#ffffff'],[1, '#ffffff']]

for episode in range(10): # n_trajectories : 10
    trajectory = []
    step = 0

    env.reset()
    print("episode_step", episode_step)

    while True: 
        # env.render()
               
        print("step", step)

        input_row = input("Insert the row number you want to modify: ")
        input_col = input("Insert the column number you want to modify: ")

        row = int(input_row)
        col = int(input_col)

        action = row * 38 + col
        state, CE, name_list, reward, correlation_matrix, new_sorted_component_list, done, _ = env.step(action, correlation_matrix)

        temp = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == 1:
                    temp += 1
        print(temp)
        print(new_sorted_component_list)
        print("CE:", CE)
        # global 

        if ((step % 10) == 0):
            df = pd.DataFrame(state)
            df.to_csv('%dth sample.csv'%step, index=False)
        
        fig = px.imshow(state, labels=dict(x="components", y="components", color="I/F"), x = name_list, y = name_list)
        fig.show()

        # for i in range(len(correlation_matrix)):
        #     print(correlation_matrix[i])

        # colorscale = [[0, '#272D31'],[.5, '#ffffff'],[1, '#ffffff']]
        # # font=['#FCFCFC', '#00EE00', '#008B00', '#004F00', '#660000', '#CD0000', '#FF3030']

        # fig = ff.create_table(state, colorscale=colorscale)
        # fig.layout.width=250
        # fig.show()


        if step > 200: # trajectory_length : 150
            break

        state_in_array = np.zeros(1444)
        for i in range(len(state)):
            for j in range(len(state[0])):
                idx = i * 38 + j
                state_in_array[idx] = state[i][j]
                
        state_in_array = np.append(state_in_array, np.array(action))
        trajectory.append(state_in_array)
        step += 1

    # trajectory_numpy = np.array(trajectory, float)
    # print("trajectory_numpy.shape", trajectory_numpy.shape)
    trajectories.append(trajectory)

    # correlation_matrix_numpy = np.array(correlation_matrix, float)
    # print("correlation_matrix_numpy.shape", correlation_matrix_numpy.shape)

    correlation_matrices.append(correlation_matrix)
    new_sorted_component_lists.append(new_sorted_component_list)
    episode_step += 1

# k = correlation_matrices[-1][-1]
# k = k[:-1]
# k = np.reshape(k, (38, 38))
# for i in range(len(k)):
#     print(k[i])

print(new_sorted_component_lists[-1][-1])



np_trajectories = np.array(trajectories, float)
np_correlation_matrices = np.array(correlation_matrices, float)
np_new_sorted_component_lists = np.array(new_sorted_component_lists, dtype=object)
# print("np_trajectories.shape", np_trajectories.shape)
# print("np_correlation_matrices.shape", np_correlation_matrices.shape)

np.save("expert_trajectories", arr=np_trajectories)
np.save("correlation_matrices", arr=np_correlation_matrices)
np.save("new_sorted_component_lists", arr=np_new_sorted_component_lists)