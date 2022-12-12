import gym 
import pylab
import numpy as np
import random

import plotly.express as px
import cufflinks as cf
cf.go_offline(connected=True)

from maxent import *
from expert_demo.environment import Moduleviser

n_states = 75 * 75
n_actions = 1444

q_table = np.zeros((n_states, n_actions))
feature_matrix = np.eye((n_states))

gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05

np.random.seed(1)
correlation_matrix = np.zeros((38,38))

def idx_demo():
    raw_demo = np.load(file="expert_demo/expert_trajectories.npy")

    demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))

    for x in range(len(raw_demo)):
        for y in range(len(raw_demo[0])):
            demonstrations[x][y][1] = raw_demo[x][y][-1]
            a_1, a_2 = 0, 0
            for z in range(len(raw_demo[0][0])-1):
                row, col = divmod(z,38)
                if ((row>=10 and row<=19)and(col>=0 and col <=4))or((row>=5 and row<=9)and(col>=15 and col<=19)):
                    if raw_demo[x][y][z] == 1:
                        a_1 += 1
                    if ((row>=5 and row <=9)and(col >=0 and col <=4))or((row>=5 and row <=9)and(col >=10 and col<=14))or((row>10 and row<=14)and(col>=15 and col<=19)):
                        if raw_demo[x][y][z] == 1:
                            a_2 += 1
            state_idx = 75*a_1 + a_2

            demonstrations[x][y][0] = state_idx
    return demonstrations

def idx_state(state):
    a_3, a_4 = 0, 0
    for x in range(len(state)):
        for y in range(len(state[0])):
            if ((x>=10 and x<=19)and(y>=0 and y<=4))or((x>=5 and x<=9)and(y>=15 and y<=19)):
                if state[x][y] == 1:
                    a_3 += 1
            if ((x>=5 and x<=9)and(y>=0 and y<=4))or((x>=5 and x<=9)and(y>=10 and y<=14))or((x>=10 and x<=14)and(y>=15 and y<=19)):
                if state[x][y] == 1:
                    a_4 += 1
    state_idx = 75*a_3 + a_4
    return state_idx

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action] # previous value
    q_2 = reward + gamma * max(q_table[next_state]) # q_value calculation based on bellman equation
    q_table[state][action] += q_learning_rate * (q_2-q_1)

def main():
    env = Moduleviser()
    demonstrations = idx_demo()

    expert = expert_feature_expectations(feature_matrix, demonstrations)
    learner_feature_expectations = np.zeros(n_states)
    theta = -(np.random.uniform(size=(n_states,)))
    episodes, scores = [], []
    correlation_matrix = np.zeros((38,38))

    raw_demo_new_sorted_component_lists = np.load(file="expert_demo/new_sorted_component_lists.npy", allow_pickle=True)

    for episode in range(2000):
        state = env.reset()
        score = 0

        if (episode != 0 and episode == 100) or (episode > 100 and episode % 50 == 0):
            learner = learner_feature_expectations / episode
            maxent_irl(expert, learner, theta, theta_learning_rate)

        while True:
            # RL part
            state_idx = idx_state(state)
            raw_demo_correlation_matrices = np.load(file="expert_demo/correlation_matrices.npy")

            # based on raw_demo_correlation_matrices, sample the action
            i_opt = 0
            j_opt = 0
            sample = random.random()
            for i in range(len(raw_demo_correlation_matrices[0])):
                for j in range(len(raw_demo_correlation_matrices[0][0])):
                    if sample < raw_demo_correlation_matrices[0][i][j]:
                        i_opt = i
                        j_opt = j
                        break
                    sample -= raw_demo_correlation_matrices[0][i][j]

            action = i_opt * 38 + j_opt
            # action = np.argmax(q_table[state_idx])
            next_state, CE, name_list, reward, correlation_matrix, new_sorted_component_list, done, _ = env.step(action, correlation_matrix)

            # fig = px.imshow(correlation_matrix, text_auto=True, labels=dict(x="components", y="components", color="I/F"),
            # x=['Ad.. B-pil. roof rail', 'Body side', 'Front header', 'Rear header', 
            #     'Rear pan. In. Upp.', 'Roof bow', 'Roof panel', 'Roof rail', 'Channel', 'Dash cross mem.', 'Floor panel', 'Front side rail', 
            #     'R. side rail center', 'Seat crossm. fr.', 'Seat crossm. rear', 'Back panel', 'Back panel side', 'Back panel upper','Rear floor side', 'Rear side rail',
            #     'Spare wheel well', 'Ad. A-pil. roof rail', 'A-pillar inner', 'A-pillar reinforc.', 'Cowl', 'Dash panel', 'Front susp. Hous.', 'Shotgun', 'B-Pillar',
            #     'Crosstr. rear floor', 'Heelkick', 'Rear floor panel', 'R. panel in. lower', 'Rear side floor', 'Rear side rail frt', 'Reinf. rocker rear', 'Rocker', 'Wheelhouse'],
            # y=['Ad.. B-pil. roof rail', 'Body side', 'Front header', 'Rear header', 
            #     'Rear pan. In. Upp.', 'Roof bow', 'Roof panel', 'Roof rail', 'Channel', 'Dash cross mem.', 'Floor panel', 'Front side rail', 
            #     'R. side rail center', 'Seat crossm. fr.', 'Seat crossm. rear', 'Back panel', 'Back panel side', 'Back panel upper','Rear floor side', 'Rear side rail',
            #     'Spare wheel well', 'Ad. A-pil. roof rail', 'A-pillar inner', 'A-pillar reinforc.', 'Cowl', 'Dash panel', 'Front susp. Hous.', 'Shotgun', 'B-Pillar',
            #     'Crosstr. rear floor', 'Heelkick', 'Rear floor panel', 'R. panel in. lower', 'Rear side floor', 'Rear side rail frt', 'Reinf. rocker rear', 'Rocker', 'Wheelhouse'])
            # fig.show()
            # print("HI!")

            # IRL part
            irl_reward = get_reward(feature_matrix, theta, n_states, state_idx)
            next_state_idx = idx_state(next_state)
            update_q_table(state_idx, action, irl_reward, next_state_idx)
            learner_feature_expectations += feature_matrix[int(state_idx)]

            score += reward
            # print("score: ", score)
            state = next_state

            if done or (score < -10):
                scores.append(score)
                episodes.append(episode)
                break
        
        if episode % 100 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./learning_curves/maxent_30000.png")
            np.save("./results/maxent_q_table", arr=q_table)
            fig = px.imshow(state, text_auto=True, labels=dict(x="components", y="components", color="I/F"),
            x=['Ad.. B-pil. roof rail', 'Body side', 'Front header', 'Rear header', 
                'Rear pan. In. Upp.', 'Roof bow', 'Roof panel', 'Roof rail', 'Channel', 'Dash cross mem.', 'Floor panel', 'Front side rail', 
                'R. side rail center', 'Seat crossm. fr.', 'Seat crossm. rear', 'Back panel', 'Back panel side', 'Back panel upper','Rear floor side', 'Rear side rail',
                'Spare wheel well', 'Ad. A-pil. roof rail', 'A-pillar inner', 'A-pillar reinforc.', 'Cowl', 'Dash panel', 'Front susp. Hous.', 'Shotgun', 'B-Pillar',
                'Crosstr. rear floor', 'Heelkick', 'Rear floor panel', 'R. panel in. lower', 'Rear side floor', 'Rear side rail frt', 'Reinf. rocker rear', 'Rocker', 'Wheelhouse'],
            y=['Ad.. B-pil. roof rail', 'Body side', 'Front header', 'Rear header', 
                'Rear pan. In. Upp.', 'Roof bow', 'Roof panel', 'Roof rail', 'Channel', 'Dash cross mem.', 'Floor panel', 'Front side rail', 
                'R. side rail center', 'Seat crossm. fr.', 'Seat crossm. rear', 'Back panel', 'Back panel side', 'Back panel upper','Rear floor side', 'Rear side rail',
                'Spare wheel well', 'Ad. A-pil. roof rail', 'A-pillar inner', 'A-pillar reinforc.', 'Cowl', 'Dash panel', 'Front susp. Hous.', 'Shotgun', 'B-Pillar',
                'Crosstr. rear floor', 'Heelkick', 'Rear floor panel', 'R. panel in. lower', 'Rear side floor', 'Rear side rail frt', 'Reinf. rocker rear', 'Rocker', 'Wheelhouse'])
    fig.show()
    print("CE:",CE)

    #new_sorted_component_list
    #raw_demo_new_sorted_component_lists
    #compare!
    correctness = 0
    correct_component = []
    np_new_sorted_component_list = np.array(new_sorted_component_list, dtype=object)
    
    print(np_new_sorted_component_list)
    print(raw_demo_new_sorted_component_lists[-1])

    for i in range(len(np_new_sorted_component_list)):
        for j in range(len(np_new_sorted_component_list[i])):
            if np_new_sorted_component_list[i][j] in raw_demo_new_sorted_component_lists[-1][i]:
                correctness += 1
                correct_component.append(np_new_sorted_component_list[i][j])
    
    successful_rate_modularization = correctness / 38
    print("successful_rate_modularization:",successful_rate_modularization)
    print("correct_component:",correct_component)

if __name__ == '__main__':
    main()