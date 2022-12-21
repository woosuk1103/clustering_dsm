import copy
from typing import Optional

import numpy as np

import gym
from gym import spaces

import cv2
import matplotlib.pyplot as plt
import plotly.express as px

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

# make optimal base matrix
opt_base_matrix = np.eye(38, dtype=int)

opt_base_matrix[0][2] = 1
opt_base_matrix[0][3] = 1
opt_base_matrix[0][10] = 1
opt_base_matrix[0][13] = 1
opt_base_matrix[0][17] = 1
opt_base_matrix[0][35] = 1

opt_base_matrix[1][7] = 1
opt_base_matrix[1][15] = 1
opt_base_matrix[1][21] = 1
opt_base_matrix[1][23] = 1
opt_base_matrix[1][30] = 1
opt_base_matrix[1][31] = 1
opt_base_matrix[1][32] = 1

opt_base_matrix[2][3] = 1
opt_base_matrix[2][10] = 1
opt_base_matrix[2][13] = 1
opt_base_matrix[2][17] = 1
opt_base_matrix[2][35] = 1

opt_base_matrix[3][10] = 1
opt_base_matrix[3][13] = 1
opt_base_matrix[3][17] = 1
opt_base_matrix[3][35] = 1

opt_base_matrix[4][5] = 1
opt_base_matrix[4][6] = 1
opt_base_matrix[4][20] = 1
opt_base_matrix[4][25] = 1
opt_base_matrix[4][36] = 1

opt_base_matrix[5][6] = 1
opt_base_matrix[5][20] = 1
opt_base_matrix[5][25] = 1
opt_base_matrix[5][36] = 1

opt_base_matrix[6][20] = 1
opt_base_matrix[6][25] = 1
opt_base_matrix[6][36] = 1

opt_base_matrix[7][15] = 1
opt_base_matrix[7][21] = 1
opt_base_matrix[7][23] = 1
opt_base_matrix[7][30] = 1
opt_base_matrix[7][31] = 1
opt_base_matrix[7][32] = 1

opt_base_matrix[8][11] = 1
opt_base_matrix[8][18] = 1
opt_base_matrix[8][19] = 1
opt_base_matrix[8][22] = 1
opt_base_matrix[8][24] = 1
opt_base_matrix[8][27] = 1
opt_base_matrix[8][28] = 1
opt_base_matrix[8][29] = 1
opt_base_matrix[8][37] = 1

opt_base_matrix[9][12] = 1
opt_base_matrix[9][14] = 1
opt_base_matrix[9][16] = 1
opt_base_matrix[9][26] = 1
opt_base_matrix[9][33] = 1
opt_base_matrix[9][34] = 1

opt_base_matrix[10][13] = 1
opt_base_matrix[10][17] = 1
opt_base_matrix[10][35] = 1

opt_base_matrix[11][18] = 1
opt_base_matrix[11][19] = 1
opt_base_matrix[11][22] = 1
opt_base_matrix[11][24] = 1
opt_base_matrix[11][27] = 1
opt_base_matrix[11][28] = 1
opt_base_matrix[11][29] = 1
opt_base_matrix[11][37] = 1

opt_base_matrix[12][14] = 1
opt_base_matrix[12][16] = 1
opt_base_matrix[12][26] = 1
opt_base_matrix[12][33] = 1
opt_base_matrix[12][34] = 1

opt_base_matrix[13][17] = 1
opt_base_matrix[13][35] = 1

opt_base_matrix[14][16] = 1
opt_base_matrix[14][26] = 1
opt_base_matrix[14][33] = 1
opt_base_matrix[14][34] = 1

opt_base_matrix[15][21] = 1
opt_base_matrix[15][23] = 1
opt_base_matrix[15][30] = 1
opt_base_matrix[15][31] = 1
opt_base_matrix[15][32] = 1

opt_base_matrix[16][26] = 1
opt_base_matrix[16][33] = 1
opt_base_matrix[16][34] = 1

opt_base_matrix[17][35] = 1

opt_base_matrix[18][19] = 1
opt_base_matrix[18][22] = 1
opt_base_matrix[18][24] = 1
opt_base_matrix[18][27] = 1
opt_base_matrix[18][28] = 1
opt_base_matrix[18][29] = 1
opt_base_matrix[18][37] = 1

opt_base_matrix[19][22] = 1
opt_base_matrix[19][24] = 1
opt_base_matrix[19][27] = 1
opt_base_matrix[19][28] = 1
opt_base_matrix[19][29] = 1
opt_base_matrix[19][37] = 1

opt_base_matrix[20][25] = 1
opt_base_matrix[20][36] = 1

opt_base_matrix[21][23] = 1
opt_base_matrix[21][30] = 1
opt_base_matrix[21][31] = 1
opt_base_matrix[21][32] = 1

opt_base_matrix[22][24] = 1
opt_base_matrix[22][27] = 1
opt_base_matrix[22][28] = 1
opt_base_matrix[22][29] = 1
opt_base_matrix[22][37] = 1

opt_base_matrix[23][30] = 1
opt_base_matrix[23][31] = 1
opt_base_matrix[23][32] = 1

opt_base_matrix[24][27] = 1
opt_base_matrix[24][28] = 1
opt_base_matrix[24][29] = 1
opt_base_matrix[24][37] = 1

opt_base_matrix[25][36] = 1

opt_base_matrix[26][33] = 1
opt_base_matrix[26][34] = 1

opt_base_matrix[27][28] = 1
opt_base_matrix[27][29] = 1
opt_base_matrix[27][37] = 1

opt_base_matrix[28][29] = 1
opt_base_matrix[28][37] = 1

opt_base_matrix[29][37] = 1

opt_base_matrix[30][31] = 1
opt_base_matrix[30][32] = 1

opt_base_matrix[31][32] = 1

opt_base_matrix[33][34] = 1

# make matrix symmetical
for i in range(len(opt_base_matrix)):
    for j in range(len(opt_base_matrix[0])):
        if opt_base_matrix[i][j] == 1:
            opt_base_matrix[j][i] = 1

class Moduleviser(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):

        self.CE = 0

        self.low = np.zeros(shape=(1444,), dtype=np.float32)
        self.high = np.ones(shape=(1444,), dtype=np.float32)

        self.action_space = spaces.Discrete(2888)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.name_list = name_list
        self.opt_base_matrix = opt_base_matrix

    def step(self, action, correlation_matrix):

        state = self.state

        row, col = divmod(action, 38)

        if state[row][col] == 0:
            state[row][col] = 1
            state[col][row] = 1
        else:
            state[row][col] = 0
            state[col][row] = 0
        
        self.state, clustered_matrix, self.base_matrix, self.CE, self.name_list, reward, correlation_matrix, new_sorted_component_list = self.clustering(state, correlation_matrix)
        
        done = False
        # compare present base_matrix with optimal base_matrix 
        if np.array_equal(self.base_matrix, self.opt_base_matrix):
            done = True
        
        return self.state, clustered_matrix, self.base_matrix, self.CE, self.name_list, reward, correlation_matrix, new_sorted_component_list, done, {}

    def reset(self):

        init_if_matrix = np.eye(38, dtype=int)

        init_if_matrix[0][2] = 1
        init_if_matrix[0][3] = 1
        init_if_matrix[0][7] = 1
        init_if_matrix[0][10] = 1
        init_if_matrix[0][32] = 1
        init_if_matrix[0][35] = 1

        init_if_matrix[1][7] = 1
        init_if_matrix[1][8] = 1
        init_if_matrix[1][32] = 1

        init_if_matrix[2][10] = 1
        init_if_matrix[2][13] = 1
        init_if_matrix[2][14] = 1
        init_if_matrix[2][29] = 1
        init_if_matrix[2][35] = 1

        init_if_matrix[3][7] = 1
        init_if_matrix[3][13] = 1
        init_if_matrix[3][29] = 1
        init_if_matrix[3][35] = 1

        init_if_matrix[4][5] = 1
        init_if_matrix[4][6] = 1
        init_if_matrix[4][20] = 1
        init_if_matrix[4][25] = 1
        init_if_matrix[4][36] = 1

        init_if_matrix[5][6] = 1
        init_if_matrix[5][20] = 1

        init_if_matrix[6][20] = 1
        init_if_matrix[6][22] = 1
        init_if_matrix[6][23] = 1
        init_if_matrix[6][37] = 1

        init_if_matrix[7][8] = 1
        init_if_matrix[7][14] = 1
        init_if_matrix[7][22] = 1
        init_if_matrix[7][23] = 1
        init_if_matrix[7][28] = 1
        init_if_matrix[7][29] = 1
        init_if_matrix[7][30] = 1
        init_if_matrix[7][31] = 1
        init_if_matrix[7][32] = 1
        init_if_matrix[7][33] = 1
        init_if_matrix[7][34] = 1

        init_if_matrix[8][28] = 1
        init_if_matrix[8][29] = 1

        init_if_matrix[9][12] = 1
        init_if_matrix[9][13] = 1
        init_if_matrix[9][18] = 1
        init_if_matrix[9][33] = 1
        init_if_matrix[9][34] = 1

        init_if_matrix[10][13] = 1
        init_if_matrix[10][17] = 1
        init_if_matrix[10][35] = 1

        init_if_matrix[11][19] = 1
        init_if_matrix[11][27] = 1
        init_if_matrix[11][36] = 1

        init_if_matrix[12][13] = 1

        init_if_matrix[13][14] = 1
        init_if_matrix[13][16] = 1
        init_if_matrix[13][17] = 1

        init_if_matrix[14][16] = 1
        init_if_matrix[14][18] = 1
        init_if_matrix[14][26] = 1
        init_if_matrix[14][27] = 1
        init_if_matrix[14][29] = 1

        init_if_matrix[15][31] = 1
        init_if_matrix[15][32] = 1

        init_if_matrix[16][17] = 1
        init_if_matrix[16][26] = 1

        init_if_matrix[17][35] = 1

        init_if_matrix[18][19] = 1
        init_if_matrix[18][24] = 1
        init_if_matrix[18][27] = 1
        init_if_matrix[18][28] = 1
        init_if_matrix[18][29] = 1

        init_if_matrix[19][24] = 1
        init_if_matrix[19][27] = 1
        init_if_matrix[19][36] = 1

        init_if_matrix[20][24] = 1
        init_if_matrix[20][25] = 1
        init_if_matrix[20][27] = 1
        init_if_matrix[20][36] = 1
        init_if_matrix[20][37] = 1

        init_if_matrix[21][31] = 1

        init_if_matrix[22][23] = 1
        init_if_matrix[22][27] = 1
        init_if_matrix[22][28] = 1
        init_if_matrix[22][29] = 1
        init_if_matrix[22][37] = 1

        init_if_matrix[23][31] = 1
        init_if_matrix[23][37] = 1

        init_if_matrix[24][25] = 1
        init_if_matrix[24][27] = 1
        init_if_matrix[24][28] = 1
        init_if_matrix[24][29] = 1
        init_if_matrix[24][37] = 1

        init_if_matrix[25][27] = 1
        init_if_matrix[25][36] = 1
        init_if_matrix[25][37] = 1

        init_if_matrix[26][27] = 1
        init_if_matrix[26][29] = 1

        init_if_matrix[27][28] = 1
        init_if_matrix[27][29] = 1
        init_if_matrix[27][37] = 1

        init_if_matrix[28][29] = 1
        init_if_matrix[28][37] = 1

        init_if_matrix[29][33] = 1
        init_if_matrix[29][34] = 1

        init_if_matrix[30][31] = 1
        init_if_matrix[30][32] = 1

        init_if_matrix[31][32] = 1

        # make matrix symmetical
        for i in range(len(init_if_matrix)):
            for j in range(len(init_if_matrix[0])):
                if init_if_matrix[i][j] == 1:
                    init_if_matrix[j][i] = 1
                    
        # there is no optimal state for if_matrix, it exists only for base_matrix

        self.state = init_if_matrix

        # set initial base matrix
        self.base_matrix = np.eye(38, dtype=int)
        
        return self.state, self.base_matrix, {}

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

        base_matrix = self.base_matrix

        for i in range(len(cluster_result)):
            # find nonzero term's index
            nonzero_indexes = np.nonzero(cluster_result[i])
            nonzero_indexes = np.array(nonzero_indexes)
            nonzero_indexes = np.squeeze(nonzero_indexes)
            
            print(nonzero_indexes)
            # modify base_matrix based on modularized result
            for j in nonzero_indexes:
                for k in nonzero_indexes:
                    base_matrix[j][k] = 1
        
        self.base_matrix = base_matrix
        

        classify_components_into_modules = cluster_result.sum(axis = 1)
        sorted_classify_components_into_modules = copy.deepcopy(classify_components_into_modules)
        sorted_classify_components_into_modules.sort()
        
        reversed = sorted_classify_components_into_modules[::-1]

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


        # convert new_sorted_component_list into original numbers
        for i in range(len(new_sorted_component_list)):
            for j in range(int(len(new_sorted_component_list[i]))):
                for k in range(len(name_list)):
                    if new_sorted_component_list[i][j] == name_list[k]:
                        new_sorted_component_list[i][j] = k
        

        # update correlation_matrix based on new_sorted_component_list
        for i in range(len(new_sorted_component_list)):
            for j in range(int(len(new_sorted_component_list[i]))):
                for k in range(int(len(new_sorted_component_list[i]))):
                    if j != k:
                        correlation_matrix[new_sorted_component_list[i][j]][new_sorted_component_list[i][k]] += 1
                
        total = 0
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix[0])):
                total += correlation_matrix[i][j]

        correlation_matrix /= total
        
        # update the components sequence        
        clustered_matrix = np.eye(38, dtype=np.int32)

        # make new DSM matrix
        for i in range(len(clustered_matrix)):
            for j in range(i, len(clustered_matrix)):
                if state[i][j] == 1:
                    clustered_matrix[new_order.index(i)][new_order.index(j)] = 1
                    clustered_matrix[new_order.index(j)][new_order.index(i)] = 1

        # calculate reward compared to optimal_modularized_result
        reward = 0.0
        for i in range(len(base_matrix)):
            for j in range(len(base_matrix)):
                if base_matrix[i][j] == opt_base_matrix[i][j]:
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
                        
        a = 0.5
        b = 0.5
        CE = 1 / (a * S_in + b * S_out)

        return self.state, clustered_matrix, self.base_matrix, CE, new_name_list, reward, correlation_matrix, new_sorted_component_list