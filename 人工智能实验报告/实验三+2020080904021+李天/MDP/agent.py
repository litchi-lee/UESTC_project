import numpy as np
import math

class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # dimension of acgtion
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma # 衰减系数
        self.epsilon = 0.1
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 300
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        self.sample_count += 1
        # epsilon的更新
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            # 选取对应Q最大的动作
            action = np.argmax(self.Q_table[int(state)])
        else:
            # 随机选取动作
            action = np.random.choice(self.action_dim)
        return action

    def predict(self, state):
        action = np.argmax(self.Q_table[int(state)])
        return action

    def update(self, state, action, reward, next_state, done):
        # 计算Q估计
        Q_predict = self.Q_table[int(state)][action]
        # 计算Q现实
        if done:
            # 如果回合结束，则直接等于当前奖励
            Q_target = reward
        else:
            # 如果回合每结束，则按照
            Q_target = reward + self.gamma * np.max(self.Q_table[int(next_state)])
        # 根据Q估计和Q现实，差分地更新Q表格
        self.Q_table[int(state)][action] += self.lr * (Q_target - Q_predict)

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
