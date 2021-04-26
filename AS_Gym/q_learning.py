import numpy as np
import random
import csv
from configparser import ConfigParser
from os.path import dirname, abspath, join


class QAgent:
    def __init__(self, is_chasing, states_in_env, observation_space, action_space, config):
        # config2 = ConfigParser()
        # config2.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))

        self.learning_rate = float(config['learning_settings']['learning_rate'])
        self.discount_rate = float(config['learning_settings']['discount_rate'])
        self.goal_reward = float(config['learning_settings']['goal_reward'])
        self.obstacle_reward = float(config['learning_settings']['obs_reward'])
        self.epsilon = float(config['learning_settings']['epsilon'])

        self.is_chasing = is_chasing
        self.states_in_env = states_in_env
        self.observation_space = observation_space
        self.action_space = action_space

        # q-learning observations only include the state of the agent and its destination state
        # including the locations of other drones would increase training time greatly
        obs_space_n = self.states_in_env * self.states_in_env
        self.Q_table = np.zeros((int(obs_space_n), int(action_space.n)))
        self.previous_state = -1
        self.current_state = -1

    def update(self, last_state, expected_state, action, reward, obs, done):
        # update occurs after movement is complete,
        # state = last state, next state = next state given action (even if blocked)
        state = last_state + (self.states_in_env * obs[1])
        next_state = expected_state + (self.states_in_env * obs[1])

        # R is always 0 unless at end state
        if reward == 1:
            final_reward = self.goal_reward
        elif reward == -1:
            final_reward = self.obstacle_reward
        else:
            final_reward = reward

        # updates the reward of the agent when it enters a state
        # Q(s(t), a(t)) = Q(s(t), a(t)) + A[R(t+1) + Y MaxQ(s(t+1), a(t+1)) - Q(s(t), a(t))
        # where: t = current timestep, A = learning rate, Y = discount rate,
        #     MaxQ = best reward possible in next state, r = reward for s,a pair

        Q_ta = self.Q_table[state, action]
        # next_state = int(self.env.get_next_state(state, action))
        maxQ = self.Q_table[next_state, 0]
        for acs in range(1, int(self.action_space.n) - 1):
            if self.Q_table[next_state, acs] > maxQ:
                maxQ = self.Q_table[next_state, acs]
        Y_maxQ = self.discount_rate * maxQ

        q = Q_ta + (self.learning_rate * (final_reward + Y_maxQ - Q_ta))
        self.Q_table[state, action] = q

    def get_action(self, obs):
        state = obs[0] + (self.states_in_env * obs[1])
        # epsilon greedy policy
        rand = random.uniform(0, 1)
        if rand <= self.epsilon:
            action = random.randint(0, self.action_space.n - 1)
        else:
            maxQ = 0
            action = random.randint(0, self.action_space.n - 1)
            for acs in range(0, int(self.action_space.n) - 1):
                if self.Q_table[state, acs] > maxQ:
                    maxQ = self.Q_table[state, acs]
                    action = acs

        return action

    def save_data(self, file_name):
        with open(file_name, mode='w') as q_file:
            csv_writer = csv.writer(q_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(0, len(self.Q_table)):
                for acs in range(0, len(self.Q_table[i])):
                    csv_writer.writerow([
                        self.Q_table[i, acs]])

    def read_data(self, file_name):
        print("Reading in saved data...")
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            line = 0
            action = 0
            state = 0
            for row in csv_reader:
                self.Q_table[state, action] = row[0]
                line += 1
                if action == 5:
                    state += 1
                    action = 0
                else:
                    action += 1
        print("Data read in...")
