import torch
import numpy as np
from torch import nn
from torchkeras import summary
from collections import namedtuple
import gym
import math
from torch.utils.tensorboard import SummaryWriter
#定义参数类
class Parameters():
    def __init__(self):
        # self.env_name = 'Humanoid-v2'  #states: 376  actions:17
        self.env_name = 'Walker2d-v2'
        self.init_layer = True
        self.num_hidden = 64
        self.lr = 3e-4
        self.seed = 1234
        self.num_episode = 2000
        self.batch_size = 2048
        self.max_steps_per_round = 2000
        self.state_norm = True       #待验证
        self.lamda = 0.97
        self.discount = 0.995
        self.EPS = 1e-10
        self.norm_advantage = True
        self.num_epochs = 10
        self.minibatch_size = 256
        self.clip = 0.2     #也可以设置成动态变化
        self.coeff1 = 0.5
        self.coeff2 = 0.01
        self.dynamic_lr = True
        self.log_num_episode = True
        self.dynamic_clip = True


#2D Tensor
#定义演说家-评论家网络
class ActorCritic(nn.Module):
    def __init__(self, num_features, num_actions, num_hidden=64, init_layer=True):
        super(ActorCritic, self).__init__()
        self.actor_fc1 = nn.Linear(num_features, num_hidden)
        self.actor_fc2 = nn.Linear(num_hidden, num_hidden) 
        self.actor_fc3 = nn.Linear(num_hidden, num_actions)
        self.actor_logstd = nn.Parameter(torch.zeros((1, num_actions))) #是个参数，还需要研究一下

        self.critic_fc1 = nn.Linear(num_features, num_hidden)
        self.critic_fc2 = nn.Linear(num_hidden, num_hidden)
        self.critic_fc3 = nn.Linear(num_hidden, 1)

        if init_layer:
            self._init_layer(self.actor_fc1, std=1.0)
            self._init_layer(self.actor_fc2, std=1.0)
            self._init_layer(self.actor_fc3, std=0.01)

            self._init_layer(self.critic_fc1, std=1.0)
            self._init_layer(self.critic_fc2, std=1.0)
            self._init_layer(self.critic_fc3, std=1.0)

    def forward(self, states):
        action_mean, action_logstd= self._forward_actor(states)
        value = self._forward_critic(states)
        return action_mean, action_logstd, value


    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        value = self.critic_fc3(x)
        return value

    @staticmethod
    def _init_layer(layer, std=1.0, bias_val=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_val)


    def select_action(self, action_mean, action_logstd, return_logproba=True):
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._norm_logproba(action, action_mean, action_logstd)
        return action, logproba


    @staticmethod
    def _norm_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x-mean).pow(2) / (2 * std.pow(2)) #log(f(x))展开
        return logproba.sum(1)

    def get_logproba(self, state, action):
        action_mean, action_logstd = self._forward_actor(state)
        logproba = self._norm_logproba(action, action_mean, action_logstd)
        return logproba




# args = Parameters()
# net = ActorCritic(6, 4, args.num_hidden, args.init_layer)
# summary(net, (6, ))

#定义内存回放
class Memory():
    def __init__(self):
        self.memory = []
        self.transition = namedtuple('transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))

    def store(self, *args):
        self.memory.append(self.transition(*args))
    
    def sample(self):
        return self.transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

# mem = Memory()
# for i in range(5):
#     mem.store(1, 2, 3, 4, 5, 6, 7)
#     mem.store(3, 4, 5, 6, 7, 8, 9)

# all = mem.sample()
# print(all.state)
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shap


class PreFilter():
    '''
    state 和 reward 都是在交互过程中产生的，我们无法在预先知道其平均值和方差，
    于是我们采用运行时均值和方差作为近似代替。
    y = (x - mean) / std
    '''
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shapes




def ppo(args):
    env = gym.make(args.env_name)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    network = ActorCritic(num_state, num_action, num_hidden=64, init_layer=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    running_state = PreFilter((num_state, ), clip=5.0)

    writer = SummaryWriter('./datappo1/tensorboard')
    writer.add_graph(network, input_to_model=torch.rand(1, num_state))

    lr_now = args.lr
    clip_now = args.clip

    for i_episode in range(args.num_episode):
        memory = Memory()
        num_steps = 0
        reward_list = []
        len_list = []

        #根据当前的策略收集足迹（一个batch大小）
        while num_steps < args.batch_size:
            states = env.reset()
            # print(states)
            if args.state_norm:    #对原始的输入数据进行标准化处理
                states = running_state(states)
            reward_sum = 0

            for t in range(args.max_steps_per_round):   #
                action_mean, action_logstd, value = network(torch.tensor(states, dtype=torch.float).unsqueeze(0))
                action, logproba = network.select_action(action_mean, action_logstd)
                action = action.data.numpy()[0]
                logproba = logproba.data.numpy()[0]
                states_, reward, done, _ = env.step(action)
                if done:
                    mask = 0
                else:
                    mask = 1

                if args.state_norm:
                    states_ =  running_state(states_) 
                reward_sum += reward
                memory.store(states, value, action, logproba, mask, states_, reward)

                if done:
                    break
                states = states_

            num_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
        
        # print(len(memory))
        print(np.mean(len_list))
        batch = memory.sample()
        batch_size = len(memory)
        state = torch.tensor(batch.state, dtype=torch.float)
        value = torch.tensor(batch.value, dtype=torch.float)
        action = torch.tensor(batch.action, dtype=torch.float)
        logproba = torch.tensor(batch.logproba, dtype=torch.float)
        mask = torch.tensor(batch.mask, dtype=torch.float)  #控制着各自足迹回报的计算
        state_ = torch.tensor(batch.next_state, dtype=torch.float)
        reward = torch.tensor(batch.reward, dtype=torch.float)
        #计算优势函数
        returns = torch.zeros(batch_size, dtype=torch.float)
        deltas = torch.zeros(batch_size, dtype=torch.float)
        advantages = torch.zeros(batch_size, dtype=torch.float) 

        pre_return = 0
        pre_value = 0
        pre_advantage = 0
        ##参考PPO论文，公式11，12    GAE(lamda, gamma)
        for i in reversed(range(batch_size)):
            returns[i] = reward[i] + args.discount * pre_return * mask[i]
            deltas[i] = returns[i] + args.discount * pre_value * mask[i] - value[i]
            advantages[i] = deltas[i] + args.lamda * args.discount * pre_advantage * mask[i]

            pre_return = returns[i]
            pre_value = value[i]
            pre_advantage = advantages[i]
        if args.norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + args.EPS)

        #小批量随机梯度下降
        for i_epoch in range(int(args.num_epochs * batch_size / args.minibatch_size)):
            minibatch_id = np.random.choice(batch_size, size=args.minibatch_size, replace=False)
            minibatch_state = state[minibatch_id]
            minibatch_value = value[minibatch_id]
            minibatch_action = action[minibatch_id]
            minibatch_oldlogproba = logproba[minibatch_id]
            minibatch_newlogproba = network.get_logproba(minibatch_state, minibatch_action)
            minbatch_reward = reward[minibatch_id]
            minibatch_advantages = advantages[minibatch_id]
            minibatch_newvalues = network._forward_critic(minibatch_state).flatten()
            minibatch_returns = returns[minibatch_id]
            # print(minibatch_returns)

            ration = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ration * minibatch_advantages
            surr2 = ration.clip(1-clip_now, 1+clip_now) * minibatch_advantages
            loss_surr = -torch.mean(torch.min(surr1, surr2))

            # loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2) )
            minibatch_return_6std = 6 * minibatch_returns.std()
            loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std

            loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            loss_total = loss_surr + args.coeff1 * loss_value + args.coeff2 * loss_entropy
            optimizer.zero_grad()
            loss_total.backward()
            # print(loss_total.grad)
            optimizer.step()

        #动态rl clip
        if args.dynamic_lr:
            r = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * r
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if args.dynamic_clip:
            c = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * c

        #控制台打印信息
        if i_episode % args.log_num_episode == 0:
            print('finished_episode: {} reward: {} total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}'.format(i_episode, np.mean(reward_list), loss_total.data, loss_surr.data, args.coeff1, loss_value.data, args.coeff2, loss_entropy))

        writer.add_scalar('loss_surr', loss_surr.data, global_step=i_episode)
        writer.add_scalar('loss_value', loss_value.data, global_step=i_episode)
        writer.add_scalar('loss_entropy', loss_entropy.data, global_step=i_episode)
        writer.add_scalar('loss_total', loss_total .data, global_step=i_episode)
        writer.add_scalar('reward', np.mean(reward_list), global_step=i_episode)


if __name__ == '__main__':
    args = Parameters()
    ppo(args)
