import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hiden, n_outer):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_feature, n_hiden)
        self.fc2 = torch.nn.Linear(n_hiden, n_outer)

        self.norm_layer(self.fc1)
        self.norm_layer(self.fc2, std=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        action_distribution = torch.distributions.Categorical(probs=x)
        return action_distribution 

    @staticmethod
    def norm_layer(layer, std=1.0, bias_constant=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_constant)


class PolicyGradient:
    def __init__(
            self, 
            actions_size, 
            feature_size, 
            learning_rate=0.001,  #神经网络学习率
            reward_decay=0.98,   #价值折扣
            output_graph=False
        ):
        self.n_actions = actions_size
        self.n_features = feature_size
        self.alpha = learning_rate
        self.gamma = reward_decay

        self.observation_store = []
        self.action_store = []
        self.reward_store = []

        self.network = Net(self.n_features, 128, self.n_actions)  #建立策略网络
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.alpha)

    def choose_action(self, observation):
        action_distri = self.network(observation)
        action = action_distri.sample()
        #其他实现
        # torch.Tensor(np.random.choice(range(self.n_actions), p=prob))
        # print(prob, action)
        return action

    def store_transaction(self, s, a, r):
        self.observation_store.append(s)
        self.action_store.append(a)
        self.reward_store.append(r)

    def learn(self):
        discounted_rewards = self._reward_to_go()
        #训练
        self.optimizer.zero_grad()

        for i in range(len(self.observation_store)):
            # self.optimizer.zero_grad()
            s = self.observation_store[i]
            a = self.action_store[i]
            dis_r = discounted_rewards[i]

            distri = self.network(s)
            loss = 0

            loss += -distri.log_prob(a) * dis_r
            print(loss)

        loss.backward()
        self.optimizer.step()
        # self.optimizer.zero_grad()
        reward_sum = torch.sum(torch.tensor(discounted_rewards).float())

        #清空回合的data
        self.observation_store = []
        self.action_store = []
        self.reward_store = []

        return reward_sum 

    def _reward_to_go(self):
        discounted_rewards = np.zeros_like(self.reward_store)
        running_sum = 0

        for i in reversed(range(0, len(self.reward_store))):
            running_sum = running_sum * self.gamma + self.reward_store[i]
            discounted_rewards[i] = running_sum

        #normalize reward
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        # print(discounted_rewards)

        return discounted_rewards


         







