import numpy as np
import random
from AIr.nn.core import NeuralNetwork
from copy import deepcopy
from collections import deque

class DQN:
    def __init__(self, n_input, n_actions, batch_size=64, epsilon=1.0, epsilon_decay=0.99, epsilon_minimum=0.05, epsilon_decay_interval=1000, gamma=0.99, eta=0.00001, train_interval=1, sync_interval=100):
        schema = {"n_input": n_input,
                  "layers": [{"n": 128, "activation": "ReLU"},
                             {"n": 128, "activation": "ReLU"},
                             {"n": n_actions, "activation": "Linear"}],
                  }

        self.target_nn = NeuralNetwork(schema, loss_method="Huber")
        self.online_nn = deepcopy(self.target_nn)

        self.n_input = n_input
        self.n_actions = n_actions
        self.experiences = deque(maxlen=10000)
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_minimum = epsilon_minimum
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_interval = epsilon_decay_interval
        self.train_interval = train_interval
        self.sync_interval = sync_interval
        self.batch_size = batch_size

        self.epsilon_decay_count = 0
        self.sync_count = 0
        self.train_count = 0

    def save(self, file_path):
        self.online_nn.store(file_path)

    def load(self, file_path):
        self.online_nn = NeuralNetwork.load_from_file(file_path)
        self.target_nn = NeuralNetwork.load_from_file(file_path)

    def experience_to_training(self, size, gamma):
        X = np.empty((size, self.n_input))
        Y = np.empty((size, self.n_actions))

        batch = random.sample(self.experiences, size)
        for i, exp in enumerate(batch):
            X[i] = deepcopy(exp[0])

            if not exp[4]:
                self.target_nn.forward(exp[3])
                q_max = np.max(self.target_nn.output)
                y = deepcopy(exp[2] + gamma * q_max)
            else:
                y = deepcopy(exp[2])

            self.online_nn.forward(exp[0])
            Y[i] = self.online_nn.output.copy()
            Y[i, exp[1]] = y

        return X,Y

    def remember(self, state, action, reward, next_state, terminated):
        if type(state) == np.ndarray and type(action) == np.ndarray and type(reward) == np.ndarray and type(next_state) == np.ndarray and type(terminated) == np.ndarray:
            for s, a, r, n, t in zip(state, action, reward, next_state, terminated):
                self.experiences.append((s, a, r, n, t))
        else:
            self.experiences.append((state, action, reward, next_state, terminated))

    def _handle_one_action(self, state):
        action = 0
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions-1)
        else:
            self.online_nn.forward(state)
            action = np.argmax(self.online_nn.output)
        self.epsilon_decay_count += 1

        if self.epsilon_decay_count > self.epsilon_decay_interval:
            self.epsilon_decay_count = 0
            self.epsilon = max(self.epsilon_minimum, self.epsilon * self.epsilon_decay)

        return action


    def action(self, state):
        if type(state) == np.ndarray:
            return np.array([self._handle_one_action(s) for s in state])
        else:
            return self._handle_one_action(state)


    # environment should be a function that takes in (state, action) and returns (return, new_state)
    def train(self):
        self.train_count += 1
        if self.train_count > self.train_interval:
            if len(self.experiences) < self.batch_size:
                return
            self.train_count = 0
            X,Y = self.experience_to_training(self.batch_size, self.gamma)
            self.online_nn.train((X,Y), eta=self.eta, silent=True)
            self.sync_count += 1

        if self.sync_count > self.sync_interval:
            self.sync_count = 0
            self.target_nn = deepcopy(self.online_nn)

