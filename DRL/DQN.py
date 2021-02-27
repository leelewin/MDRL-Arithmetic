import tensorflow as tf
import pandas as pd
import numpy as np

class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        epsilon_greedy=0.9,
        replace_target_iter,
        memory_size,
        batch_size,
        
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_greedy
        self.replace_target_iter = replace_target_iter #算法中的C
        self.memory_size = memory_size  #算法中的N
        self.batch_size = batch_size
        #
        self.epsilon = self.epsilon_max
        self.learn_step_counter = 0
        #初始化记忆[s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features*2+2))
        #创建target网络和evaluate网络
        self._build_net()
        #


    def _build_net(self):
        #--------eval神经网络-----------#
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        n_l1 = 10
        w_initializer = tf.random_normal_initializer(0, 0.3)
        b_initializer = tf.constant_initializer(0.1)

        #eval 网络第一层
        w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
        b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
        l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

        #eval 网络第二层
        w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
        b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
        self.q_eval = tf.matmul(l1, w2) + b2

        # 损失函数
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        #梯度下降
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        #-------target神经网络----------#
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], initializer=w_initializer, collections=c_names)

        c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

        #target网络第一层
        w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
        b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
        l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

        #target网络第二层
        w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
        b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
        self.q_next = tf.matmul(l1, w2) + b2

    def select_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = tf.session.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions) 
        return action

    def store_transition(self, s, a, r, s_):
        pass
    def learn(self):
        pass
    def plot_cost(self):
        pass