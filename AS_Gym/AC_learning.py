from abc import ABC

import numpy as np
import random
import csv
import tensorflow as tf
from typing import Any, List, Sequence, Tuple
from configparser import ConfigParser
from os.path import dirname, abspath, join
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from collections import deque


class ACAgent:
    def __init__(self, is_chasing, states_in_env, observation_space, action_space,
                 acs_format, obs_format, blue_agents, red_agents, config):
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
        self.memory = deque(maxlen=2000)
        self.sess = tf.compat.v1.Session()
        self.blue_agents = blue_agents
        self.red_agents = red_agents
        tf.compat.v1.keras.backend.set_session(self.sess)

        self.is_chasing = is_chasing
        self.states_in_env = states_in_env
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_format = obs_format
        self.acs_format = acs_format
        self.current_state = -1
        self.previous_state = -1

        self.actor_state_input, self.actor_model = self.create_actor_model()
        target_actor_input, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32, [None, self.acs_format.shape[0]])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def create_actor_model(self):
        state_input = Input(shape=self.obs_format.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.acs_format.shape[0], activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.obs_format.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.acs_format.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self.train_critic(samples)
        self.train_actor(samples)

    def train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, placeholder = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={self.critic_state_input: cur_state,
                                                                self.critic_action_input: predicted_action})[0]
            self.sess.run(self.optimize, feed_dict={self.actor_state_input: cur_state, self.actor_critic_grad: grads})

    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return self.actor_model.predict(cur_state)

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
        self.train()

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
