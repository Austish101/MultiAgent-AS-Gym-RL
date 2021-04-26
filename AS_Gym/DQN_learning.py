from collections import deque

import numpy as np
import random
import csv
# import tensorflow as tf
# import keras
from configparser import ConfigParser
from os.path import dirname, abspath, join

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


class DQN:
    def __init__(self, is_chasing, states_in_env, observation_space, action_space,
                 acs_format, obs_format, blue_agents, red_agents, config):
        self.memory = deque(maxlen=2000)

        # config2 = ConfigParser()
        # config2.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))
        self.learning_rate = float(config['learning_settings']['learning_rate'])
        self.discount_rate = float(config['learning_settings']['discount_rate'])
        self.goal_reward = float(config['learning_settings']['goal_reward'])
        self.obstacle_reward = float(config['learning_settings']['obs_reward'])
        self.epsilon = 1.0  # float(config['learning_settings']['epsilon'])
        self.gamma = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.05
        self.is_chasing = is_chasing
        self.states_in_env = states_in_env
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_format = obs_format
        self.acs_format = acs_format
        self.current_state = -1
        self.previous_state = -1
        self.blue_agents = blue_agents
        self.red_agents = red_agents

        # DeepMind suggests using 2 models, one for prediction and one for actual values
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.obs_format.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_space.n, activation="softmax"))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_future * self.gamma
            self.model.fit(state, target, epoch=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self, state):
        # epsilon greedy with decay
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        rand = random.uniform(0, 1)
        if rand <= self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.model.predict(state))

    def update(self, last_state, expected_state, action, reward, obs, done):
        # format obs
        if (self.blue_agents == 2) and (self.red_agents == 2):
            state = np.array([obs[1], obs[2][0], obs[2][1], obs[2][2], obs[2][3]])
        elif (self.blue_agents == 2) and (self.red_agents == 1):
            state = np.array([obs[1], obs[2][0], obs[2][1], obs[2][2]])
        elif (self.blue_agents == 2) and (self.red_agents == 0):
            state = np.array([obs[1], obs[2][0], obs[2][1]])
        elif (self.blue_agents == 1) and (self.red_agents == 0):
            state = np.array([obs[1], obs[2][0]])
        elif (self.blue_agents == 1) and (self.red_agents == 1):
            state = np.array([obs[1], obs[2][0], obs[2][2]])
        elif (self.blue_agents == 1) and (self.red_agents == 2):
            state = np.array([obs[1], obs[2][0], obs[2][2], obs[2][3]])

        new_state = obs[0]

        # R is always 0 unless at end state
        if reward == 1:
            R = self.goal_reward
        elif reward == 0:
            R = self.normal_reward
        elif reward == -1:
            R = self.obstacle_reward

        self.remember(state, action, reward, new_state, done)
        self.replay()
        self.target_train()

    def get_action(self, obs):
        # format obs
        if (self.blue_agents == 2) and (self.red_agents == 2):
            state = np.array([obs[1], obs[2][0], obs[2][1], obs[2][2], obs[2][3]])
        elif (self.blue_agents == 2) and (self.red_agents == 1):
            state = np.array([obs[1], obs[2][0], obs[2][1], obs[2][2]])
        elif (self.blue_agents == 2) and (self.red_agents == 0):
            state = np.array([obs[1], obs[2][0], obs[2][1]])
        elif (self.blue_agents == 1) and (self.red_agents == 0):
            state = np.array([obs[1], obs[2][0]])
        elif (self.blue_agents == 1) and (self.red_agents == 1):
            state = np.array([obs[1], obs[2][0], obs[2][2]])
        elif (self.blue_agents == 1) and (self.red_agents == 2):
            state = np.array([obs[1], obs[2][0], obs[2][2], obs[2][3]])

        return self.act(state)

    def save_data(self, file_name):
        with open(file_name, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # weights = self.model.get_weights()
            # target_weights = self.target_model.get_weights()


            # for i in range(0, len(self.Q_table)):
            #     for acs in range(0, len(self.Q_table[i])):
            #         q_writer.writerow([
            #             self.Q_table[i, acs, 0], self.Q_table[i, acs, 1], self.Q_table[i, acs, 2]])

    def read_data(self, file_name):
        print("Reading in saved q-table...")
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

        #     line = 0
        #     action = 0
        #     state = 0
        #     for row in q_reader:
        #         # self.Q_table[state, action, 0] = row[0]
        #         # self.Q_table[state, action, 1] = row[1]
        #         # self.Q_table[state, action, 2] = row[2]
        #         line += 1
        #         if action == 5:
        #             state += 1
        #             action = 0
        #         else:
        #             action += 1
        # print("Q-table read in")
