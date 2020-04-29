import tensorflow as tf
import numpy as np
import random

from collections import deque

class DQN:
    REPLAY_MEMORY = 10000
    BATCH_SIZE = 32
    # 오래된 상태의 가중치를 줄이기 위한 하이퍼파라미터
    GAMMA = 0.99
    # 한번에볼 프레임의 수?
    STATE_LEN = 4

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque()
        self.state = None

        # 게임의 상태를 입력받음
        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])
        # 각 상태를 만들어낸 액션의 값을 입력받음 - 행동을 나타내는 숫자 그대로 입력받음 1 2 3
        self.input_A = tf.placeholder(tf.int64, [None])
        # 손실값 계산에 사용할 값을 입력받음
        self.input_Y = tf.placeholder(tf.float32, [None])

        # 학습 신경망 생성
        self.Q = self._build_network('main')

        # 손실함수  생성
        self.cost, self.train_op = self._build_op()

        self.target_Q = self._build_network('target')

    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.n_action, activation=None)

        return Q

    # DQN의 손실함수
    def _build_op(self):
        # 원핫 인코딩
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)

        # 훈련 신경망에 더한다음
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})
        action = np.argmax(Q_value[0])

        return action

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()

        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X: next_state})

        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op, feed_dict={self.input_X: state, self.input_A: action, self.input_Y: Y})

    def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        # 상태들을 마지막 차원으로 쌓아올린 형태 - 즉 state 베이스다
        self.state = np.stack(state, axis=2)


    def remember(self, state, action, reward, terminal):
        # 학습데이터로 현재의 상태만이 아닌, 과거의 상태까지 고려하여 계산하도록 하였고,
        # 이 모델에서는 과거 3번 + 현재 = 총 4번의 상태를 계산하도록 하였으며,
        # 새로운 상태가 들어왔을 때, 가장 오래된 상태를 제거하고 새로운 상태를 넣습니다.
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

        # 플레이결과, 즉, 액션으로 얻어진 상태와 보상등을 메모리에 저장합니다.
        self.memory.append((self.state, next_state, action, reward, terminal))

        # 저장할 플레이결과의 갯수를 제한합니다.
        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state


    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal
