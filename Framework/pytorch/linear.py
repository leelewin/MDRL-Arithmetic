import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

#创建数据集
x = torch.unsqueeze(torch.linspace(-1, 3, 50), dim=1)
# y = x.pow(2) + 0.2 * torch.rand(x.size())
print(x)
x = np.linspace(-1, 3, 50).reshape(50, 1)
y = np.power(x, 2) + 0.2 * np.random.normal(size=x.shape)
# x = torch.from_numpy(x).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)
# print(x)
#批训练




#创建和配置网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net1 = Net(1, 10, 1)
# print(net1)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net1.parameters(), lr=0.05)

#网络训练以及可视化
for t in range(500):
    prediction = net1(x)
    print(type(prediction), prediction)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy())
        plt.pause(0.1)




