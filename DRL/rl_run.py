import gym
from pg_torch import PolicyGradient
import torch
import matplotlib.pyplot as plt

#     Observation:
#         Type: Box(4)
#         Num     Observation               Min                     Max
#         0       Cart Position             -4.8                    4.8
#         1       Cart Velocity             -Inf                    Inf
#         2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
#         3       Pole Angular Velocity     -Inf                    Inf
#     Actions:
#         Type: Discrete(2)
#         Num   Action
#         0     Push cart to the left
#         1     Push cart to the right
#  Reward:
#         Reward is 1 for every step taken, including the termination step

env = gym.make('CartPole-v0')
env.seed(1)
# print(env.action_space)
# print(env.observation_space.shape[0])
rl = PolicyGradient(env.action_space.n, env.observation_space.shape[0], learning_rate=0.01)
rewards_list = []

for i_episode in range(2000):
    observation = env.reset()
    while True:
        env.render()
        # torch.FloatTensor()

        action = rl.choose_action(torch.tensor(observation).float())

        observation_, reward, done, info = env.step(action.tolist())


        rl.store_transaction(torch.tensor(observation).float(), action, torch.tensor(reward).float())

        if done:
            #一个回合结束后，进行更新
            reward_sum = rl.learn()
            # print(sum(vt))
            # if i_episode == 0:
            #     plt.plot(vt)
            #     plt.show()
            rewards_list.append(reward_sum)
            # print(loss, end=",")
                
            break

        observation = observation_
plt.plot(rewards_list)
plt.show()
env.close()


