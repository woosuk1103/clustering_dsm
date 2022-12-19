import numpy as np
from typing import Optional
# import cufflinks as cf
import gym
import copy
from gym import spaces

from gym.error import DependencyNotInstalled
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
# cf.go_offline(connected=True)
# 38 
name_list = ['Adapter A-piliar roof rail', 'Adapter B-piliar roof rail', 'A-pillar inner', 'A-pillar reinforcement', 'Back panel', 'Back panel side', 'Back panel upper', 
             'Body side', 'B-Pillar', 'Channel', 'Cowl', 'Crosstrack rear floor', 'Dash cross member', 'Dash panel', 'Floor panel', 'Front header', 'Front side rail', 
             'Front suspension Housing', 'Heelkick', 'Rear floor panel', 'Rear floor side', 'Rear header', 'Rear panel inner lower', 'Rear panel Inner Upper', 
             'Rear side floor', 'Rear side rail', 'Rear side rail center', 'Rear side rail frt', 'Reinforcement rocker rear', 'Rocker', 'Roof bow', 'Roof panel', 'Roof rail',
             'Seat crossmember front', 'Seat crossmember rear', 'Shotgun', 'Spare wheel well', 'Wheelhouse']

optimal_modularized_result = [['Adapter B-piliar roof rail', 'Body side', 'Front header', 'Rear header', 'Rear panel Inner Upper', 'Roof bow', 'Roof panel', 'Roof rail'], 
                 ['Channel', 'Dash cross member', 'Floor panel', 'Front side rail', 'Rear side rail center', 'Seat crossmember front', 'Seat crossmember rear'], 
                 ['Back panel', 'Back panel side', 'Back panel upper', 'Rear floor side', 'Rear side rail', 'Spare wheel well'], 
                 ['Adapter A-piliar roof rail', 'A-pillar inner', 'A-pillar reinforcement', 'Cowl', 'Dash panel', 'Front suspension Housing', 'Shotgun'], 
                 ['B-Pillar', 'Crosstrack rear floor', 'Heelkick', 'Rear floor panel', 'Rear panel inner lower', 'Rear side floor', 'Rear side rail frt', 'Reinforcement rocker rear', 'Rocker', 'Wheelhouse']]


init_if_matrix = np.eye(38, dtype=int)

opt_if_matrix = []

init_base_matrix = []
opt_base_matrix = []

class Moduleviser(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):

        self.CE = 0

        self.low = np.zeros(shape=(1444,), dtype=np.int32)
        self.high = np.ones(shape=(1444,), dtype=np.int32)

        self.action_space = spaces.Discrete(2888)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.name_list = name_list

    def step(self, action, correlation_matrix):

        state = np.reshape(self.state, (38,38))

        row, col = divmod(action, 38)

        if state[row][col] == 0:
            state[row][col] = 1
            state[col][row] = 1
        else:
            state[row][col] = 0
            state[col][row] = 0
        
        self.state, self.CE, self.name_list, reward, correlation_matrix, new_sorted_component_list = self.clustering(state, correlation_matrix)
        # done = bool(self.CE >= 0.03)
        done = True
        return np.array(self.state, dtype=np.int32), self.CE, self.name_list, reward, correlation_matrix, new_sorted_component_list, done, {}

    def reset(self):

        state = []
        for i in range(38):
            state.append([0])
        
        state[0]  = [1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0]
        state[1]  = [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        state[2]  = [1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0]
        state[3]  = [1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0]
        state[4]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
        state[5]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[6]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        state[7]  = [1,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0]
        state[8]  = [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0]
        state[9]  = [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0]
        state[10] = [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        state[11] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0]
        state[12] = [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[13] = [0,0,1,1,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[14] = [0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0]
        state[15] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0]
        state[16] = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        state[17] = [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        state[18] = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0]
        state[19] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0]
        state[20] = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,1]
        state[21] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        state[22] = [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1]
        state[23] = [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1]
        state[24] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1]
        state[25] = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,1]
        state[26] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0]
        state[27] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1]
        state[28] = [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,1]
        state[29] = [0,0,1,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,0,0]
        state[30] = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0]
        state[31] = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0]
        state[32] = [1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0]
        state[33] = [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0]
        state[34] = [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0]
        state[35] = [1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        state[36] = [0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
        state[37] = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,1]

        state = np.array(state)
        self.state = np.reshape(state, (-1,1))

        return self.state, {}

    def clustering(self, state, correlation_matrix):
        
        # initialize weight matrix
        w = np.zeros((5,38))
        for i in range(len(w)):
            for j in range(len(w[1])):
                w[i][j] = np.random.random()

        # initialize result matrix
        cluster_result = np.zeros((5,38))
        learning_rate = 0.03

        #calculate Euclidean distances between each row and weight vectors
        for i in range(len(state)): # fix one row at state matrix
            distances = np.zeros(len(w))
            for j in range(len(distances)):
                dist = 0
                for k in range(len(state[1])):
                    dist += (state[i][k] - w[j][k]) ** 2
                distances[j] = dist
            idx = np.argmax(distances)
            cluster_result[idx][i] = 1

            # update the winning neuron
            for m in range(len(state[1])):
                w[idx][m] += learning_rate * (state[i][m] - w[idx][m])

            # update the neighbor neurons
            if idx == 0:
                w[1][m] += learning_rate * (state[i][m] - w[idx][m])

            elif idx == 1:
                w[0][m] += learning_rate * (state[0][m] - w[0][m])
                w[2][m] += learning_rate * (state[2][m] - w[2][m])

            elif idx == 2:
                w[1][m] += learning_rate * (state[1][m] - w[1][m])
                w[3][m] += learning_rate * (state[3][m] - w[3][m])
            
            elif idx == 3:
                w[2][m] += learning_rate * (state[2][m] - w[2][m])
                w[4][m] += learning_rate * (state[4][m] - w[4][m])

            else:
                w[3][m] += learning_rate * (state[3][m] - w[3][m])

        a = 0.5
        b = 0.5

        # for i in range(len(cluster_result)):
        #     print(cluster_result[i])
        classify_components_into_modules = cluster_result.sum(axis = 1)
        # print(classify_components_into_modules)
        sorted_classify_components_into_modules = copy.deepcopy(classify_components_into_modules)
        sorted_classify_components_into_modules.sort()
        
        reversed = sorted_classify_components_into_modules[::-1]
        # print(reversed)
        new_order = []
        new_name_list = []
        check = np.ones(shape=(38,), dtype=np.int32)
        for i in range(len(reversed)):
            for j in range(len(classify_components_into_modules)):
                if reversed[i] == classify_components_into_modules[j]:
                    for k in range(38):
                        if cluster_result[j][k] == 1 and check[k] == 1:
                            new_order.append(k)
                            # re-organize the sequence of the components
                            new_name_list.append(self.name_list[k])
                            check[k] = 0

        # find out newly sorted module by splitting 38 components into 5 groups
        new_name_list_second = copy.deepcopy(new_name_list)
        new_sorted_component_list = [[], [], [], [], []]
        for i in range(len(reversed)):
            for j in range(int(reversed[i])):
                new_sorted_component_list[i].append(new_name_list_second[0])
                del new_name_list_second[0]

        # print(new_sorted_component_list)

        # convert new_sorted_component_list into original numbers

        for i in range(len(new_sorted_component_list)):
            for j in range(int(len(new_sorted_component_list[i]))):
                for k in range(len(name_list)):
                    if new_sorted_component_list[i][j] == name_list[k]:
                        new_sorted_component_list[i][j] = k
        
        # print(new_sorted_component_list)

        # update correlation_matrix based on new_sorted_component_list
        for i in range(len(new_sorted_component_list)):
            for j in range(int(len(new_sorted_component_list[i]))):
                for k in range(int(len(new_sorted_component_list[i]))):
                    if j != k:
                        # print(new_sorted_component_list[i][j])
                        # print(new_sorted_component_list[i][k])
                        # print(correlation_matrix)
                        correlation_matrix[new_sorted_component_list[i][j]][new_sorted_component_list[i][k]] += 1
                
        total = 0
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix[0])):
                total += correlation_matrix[i][j]

        correlation_matrix /= total

        # for i in range(len(self.correlation_matrix)):
        #     print("self.correlation_matrix:[%d]"%i,self.correlation_matrix[i])

        # update the components sequence        
        clustered_matrix = np.eye(38, dtype=np.int32)

        # make new DSM matrix
        for i in range(len(clustered_matrix)):
            for j in range(i, len(clustered_matrix)):
                if state[i][j] == 1:
                    clustered_matrix[new_order.index(i)][new_order.index(j)] = 1
                    clustered_matrix[new_order.index(j)][new_order.index(i)] = 1

        # convert modularized_result into base_matrix
        modularized_result = []
        base_matrix = self.modularized_result_to_base_matrix(modularized_result)

        optimal_base_matrix = self.modularized_result_to_base_matrix(optimal_modularized_result)

        # calculate reward compared to optimal_modularized_result
        reward = 0.0
        for i in range(len(base_matrix)):
            for j in range(len(base_matrix)):
                if base_matrix[i][j] == optimal_base_matrix[i][j]:
                    reward += 1.0

        # initialize the S_in and S_out
        S_in = 0
        for i in range(len(classify_components_into_modules)):
            S_in += 0.5 * classify_components_into_modules[i] * (classify_components_into_modules[i] - 1)
        S_out = 0

        for i in range(len(state)):
            for j in range(i+1,len(state[0])):
                if state[i][j] == 1:
                    # comp_i and comp_j are in same module
                    if [row[i] for row in cluster_result] == [row[j] for row in cluster_result]: 
                        S_in -= 1
                        
                    else: # comp_i and comp_j are not in same module
                        S_out += 1
                        
                        
        CE = 1 / (a * S_in + b * S_out)



        # calculate the reward based on rule-based contents.
        # if CE < 0.9:
        #     reward = 1
        # see if components are in single module
        # 1. 'Adapter A-piliar roof rail', 'Adapter B-piliar roof rail', 'A-pillar inner', 'A-pillar reinforcement', 
        # 2. 'Back panel', 'Back panel side', 'Back panel upper', 'Body side', 'B-Pillar', 
        # 3. 'Floor panel', 'Front header', 'Front side rail', 'Front suspension Housing', 
        # 4. 'Rear floor panel', 'Rear floor side', 'Rear header', 'Rear panel inner lower', 'Rear panel Inner Upper', 'Rear side floor', 
        #    'Rear side rail','Rear side rail center', 'Rear side rail frt'
        #    

        return clustered_matrix, CE, new_name_list, reward, correlation_matrix, new_sorted_component_list

    def modularized_result_to_base_matrix(self, modularized_result):

        base_matrix = np.identity(38)

        for i in range(len(modularized_result)):
            for j in range(len(modularized_result[i])):
                for k in range(j, len(modularized_result[i])):
                    base_matrix[name_list.index(modularized_result[i][j])][name_list.index(modularized_result[i][k])] = 1

        return base_matrix