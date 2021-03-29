### 表格型
#### Q-learning


#### Sarsa







### 表格型近似
Why:
- 当状态空间扩展到任意大时，我们将没有足够的大的内存来存储大表格，并且精确的填充耗费的时间也会是一个问题
- 我们需要在没有见过的状态下做出合理的决定，这是很常见的
- 要想将强化学习系统应用到人工智能和大型工程应用中，系统必须有能力进行泛化。

How:      
为了能够实现这个目标，现存的大量的有监督学习函数逼近的任何方法都可以使用，只要将每次更新时涉及的二元组`s->g`作为训练样本来使用就可以了。

#### 神经网络
逼近价值函数的参数化方法(最适合的有监督学习方法)：      
通过某种学习算法来调整某个逼近函数的参数，来近似在整个状态空间上定义的价值函数。     

非线性方法包括用反向传播和各种SGD训练的人工神经网络，称为深度强化学习     


#### DQN
论文：
- Playing Atari with Deep Reinforcement Learning
- Human-level control through deep reinforcement learning（进阶版本）

DQN算法由以下几部分构成：
- 基本的Q learning
- 经验回放       
将智能体在每个时刻的经验（四元组（$S_t,A_t,R_t,S_{t+1}$））存储到一个“回放内存”中，然后通过访问这个内存来执行权重更新。这个“回放内存”在同一游戏的许多比赛中将经验进行累积。与通常的Q learning形式比较，$S_{t+1}$不再是下次更新的$S_t$，取而代之的是，使用从回放内存中提取的不相关的经验作为下次更新的数据      
优势：            
去除连续经验对当前权重的依赖      
- 神经网络计算Q值      
使用多层ANN作为函数逼近来实现Q学习的半梯度      
网络结构由3个隐卷积层，一个隐全连接层，一个输出层组成
- 暂时冻结q_target参数（切断相关性）    
原因：    
当采用参数化函数逼近方法来表示动作价值时，目标就是具有相同参数的函数，而这些参数又是被更新的参数。i.e. $\mathbf{w}_{t+1}=\mathbf{w}_{t}+\alpha\left[R_{t+1}+\gamma \max _{a} \hat{q}\left(S_{t+1}, a, \mathbf{w}_{t}\right)-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}_{t}\right)\right] \nabla \hat{q}\left(S_{t}, A_{t}, \mathbf{w}_{t}\right)$ 。更新公式对$w_t$的依赖使过程复杂了。形成对比的是，有监督学习中的目标不依赖与被更新的参数，i.e.  $L(w)=\frac{1}{2} \sum_{i=1}^{N}\left(f\left(x_{i}, w\right)-y_{i}\right)^{2}$      
方法：          
每当对动作价值网络的权重$w$进行了$C$次更新，他们将网络的当前权值插入到另一个网络，并将这些复制的权值固定，用于$w$的下一组$C$次更新。这个复制的网络在下一组$C$更新的$w$输出被用作Q learning的目标     

#### PG
对策略进行参数化    
策略参数化的方法1: 动作偏好值的柔性最大化(softmax)    

相对于对动作价值函数进行参数化的好处：   
策略参数化选择动作的概率会平滑变化，而epxilon-贪心选择某个动作的概率会突然变化很大，所以基于策略梯度的方法有更强的收敛性



### Tool
#### gym
安装atari game   
`pip install 'gym[atari]' ` with zsh      

常用的方法：    
step(action): 用于和环境交互，返回采取该动作时环境的observation，reward 以及done(done等于True时表示这个episode结束了)和info   
reset(): 返回初始的observation   
render(): 可视化游戏环境     
close(): 关闭可视化    

重要属性：   
action_space   
observation_space     

基本框架：   
<code><pre>
import gym

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
</code></pre>






