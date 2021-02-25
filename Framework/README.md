### tensorflow2
tensorflow2 使用keras和eager execution构建模型   
`tf.keras`允许创建复杂拓扑。同时低级tensorflow API始终可以使用

#### 加速神经网络训练
方法有以下几种：
- Stochastic Gradient Descent (SGD)   
分批计算，损失一定精度提高速度
- Momentum
更改更新神经网络参数的方法
- AdaGrad
更改学习率，使得每个参数的更新都要独特的学习率
- RMSProp
结合上面两种方法，但不完善
- Adam
更好的结合了Momentum和AdaGrad     

SGD 是最普通的优化器, 也可以说没有加速效果, 而 Momentum 是 SGD 的改良版, 它加入了动量原则. 后面的 RMSprop 又是 Momentum 的升级版. 而 Adam 又是 RMSprop 的升级版.     

tensorflow的优化器  

Tensorboard 可视化    
在代码中加入视图框架后，就可以在终端执行`tensorboard --logdir logs`
然后在浏览器输入网址：http://localhost:6006/    

还可以可视化训练过程，添加图表曲线等

### pytorch
<code><pre>
from torch.optim as Optimizer                       #Pytorch中优化器接口
from torch import nn                                #Pytorch中神经网络模块化接口

Class XXmodel(nn.Module) :                          #nn.Module所有网络的基类
  def init(self,~):
     #在这里设计模型结构
  def forward(self, input) :
     #在这里计算一次前向传播结果 
      
optimizer = Optimizer.XXX()                          #这里是实例化一个optimizer，也可以是自己定义的一个继承了Optimizer的Class

optimizer.zero_grad()                                #梯度清零

input = torch.tensor( data , dtype = torch.xxdtype)  #把数据转换成tensor

model = XXmodel()                                    #实例化网络模型

ouput = model(input)                                 #一次前向传播

loss = loss_fn(output, target)                       #计算损失

optimizer.backward(loss)                                      #计算梯度

optimizer.step()                                     #一次梯度下降
</code></pre>