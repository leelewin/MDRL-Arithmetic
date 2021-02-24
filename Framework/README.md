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

### pytorch