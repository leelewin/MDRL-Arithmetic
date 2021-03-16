import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, models

class PolicyGradient():
    def __init__(
        self, 
        actions_size, 
        feature_size, 
        learning_rate=0.01,  #神经网络学习率
        reward_decay=0.98,   #价值折扣
        output_graph=False
        ):
        self.n_actions = actions_siz
        self.n_features = feature_size
        self.alpha = learning_rate
        self.gamma = reward_decay

        self.observation_store = []
        self.action_store = []
        self.reward_store = []

        self.network = self._build_net()  #建立策略网络


    def _build_net(self):

        model = models.Sequential()
        model.add(layers.Input(shape=(self.n_features, )))
        model.add(layers.Dense(10, activation=tf.nn.tanh, 
                    kernel_regularization=tf.random_normal_initializer(mean=0, stddev=0.3), 
                    bias_initializer=tf.constant_initializer(0.1)))
        model.add(layers.Dense(self.n_actions, activation=tf.nn.softmax(),
                    kernel_regularization=tf.random_normal_initializer(mean=0, stddev=0.3), 
                    bias_initializer=tf.constant_initializer(0.1)))
        # model.summary()
        model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha)
                loss=
        )
        return model


    def choose_action(self, observation):
        prob = self.network.predict(observation) #所有动作的概率
        action = np.random.choice(range(self.n_actions), p=prob) #根据各动作的概率随机选择一个动作
        #有问题需要改
        return action

    def store_transition(self, s, a, r):
        self.observation_store.append(s)
        self.action_store.append(a)
        self.reward_store.append(r)


    def learn(self):
        discounted_rewords_norm = self._discount_and_norm_rewards()

        #train network
        #数据输入
        self.network.fit()


        #清空此回合的数据
        self.observation_store = []
        self.action_store = []
        self.reward_store = []

        return discounted_rewords_norm

    def _discount_and_norm_rewards(self):
        discounted_rewords = np.zeros_like(self.reward_store)
        discounted_add = 0

        for t in reversed(range(0, len(self.reward_store))):
            discounted_add = discounted_add * self.gamma + self.reward_store[t]
            discounted_rewords[t] = discounted_add

        #要normalize?这样处理的目的是希望得到相同尺度的数据，避免因为数值相差过大而导致网络无法收敛
        return discounted_rewords

        

