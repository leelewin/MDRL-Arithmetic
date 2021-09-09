import torch
import numpy as np
import matplotlib.pyplot as plt
# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data)

# print(torch_data)

# data = [1, 2, 3, 4]
# tensor = torch.FloatTensor(data)
# x = torch.unsqueeze(torch.linspace(1, 3, 50), dim=1)
# y = x.pow(2) + 0.2 * torch.rand(x.size())

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
# print(x.size(), y)

# m = torch.distributions.Categorical(torch.Tensor([0.3, 0.3, 0.4]))
# print(m.sample())
# b = [1.2, 3.4]
# a = np.array([1.2, 3.4])
# print(a.dtype)
# print(torch.Tensor(a).dtype)
# n = torch.IntTensor(3)
# print(n)

# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         print(type(action))
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

m = torch.distributions.Categorical(torch.tensor([0.1, 0.3, 0.6]).float())
print(m.sample())
log_pro = m.log_prob(torch.tensor(3).float())
print(log_pro)