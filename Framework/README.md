### tensorflow2
tensorflow2 使用keras和eager execution构建模型   
`tf.keras`允许创建复杂拓扑。同时低级tensorflow API始终可以使用


#### 数据预处理
- 结构化数据     
pandas中的DataFrame进行处理
- 图片数据
在tensorflow中准备图片数据的常用方案有两种，    
第一种是使用tf.keras中的ImageDataGenerator工具构建图片数据生成器。     
该方法更为简单，其使用范例可以参考以下文章。 https://zhuanlan.zhihu.com/p/67466552
第二种是使用tf.data.Dataset搭配tf.image中的一些图片处理方法构建数据管道。     
该方法是TensorFlow的原生方法，更加灵活，使用得当的话也可以获得更好的性能。

- 文本数据
- 时序数据


#### 构建模型
使用Keras接口有以下3种方式构建模型：   
- 使用Sequential按层顺序构建模型，    
- 使用函数式API构建任意结构模型，    
- 继承Model基类构建自定义模型。    

#### 训练模型
训练模型通常有3种方法，   
- 内置fit方法，    
- 内置train_on_batch方法，    
- 以及自定义训练循环。    

#### 评估模型



#### 保存模型与提取
可以使用Keras方式保存模型，也可以使用TensorFlow原生方式保存。    
前者仅仅适合使用Python环境恢复模型，后者则可以跨平台进行模型部署。    

推荐使用后一种方式进行保存

#### Tensorboard 可视化    
在代码中加入视图框架后，就可以在终端执行`tensorboard --logdir logs`
然后在浏览器输入网址：http://localhost:6006/    

还可以可视化训练过程，添加图表曲线等


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

一般神经网络建模使用的都是`torch.float32`类型。   
而ndarray中的浮点数据如果不特别说明为float64，

#### 张量和numpy关系

