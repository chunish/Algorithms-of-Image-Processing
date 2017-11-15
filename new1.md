稀疏编码
早年学者通过研究发现，通过收集的大量黑白风景图片中提取出许多16*16 的碎片特征，几乎所有的图像都可由这些碎片64种正交组合得到，且数量很少，即稀疏。

稀疏表达：使用少量的基本特征组合拼装出更高层抽象的特征，通常通过多层神经网来实现。

自编码器：
对于将数据抽象化，如果有很多已标注的数据，则可以训练成深层的神经网络。然而，未被标注的数据，我们通常使用无监督学习的自编码器来提取特征。
自编码器用自身的高阶特征来编码自己，也是神经网络的一种，它的输入输出一致（类似于数学的证明题），借助稀疏编码的思想使用一些稀疏的高阶特征重新组合来
重构自己。
主要特征有：
1. 期望输入/输出一致
2. 高阶重构，而非复制像素点

Deep Belief Networks：由多层RBM（限制玻尔兹曼机）堆叠而成，先由自编码器进行无监督预训练，提取特征并赋予初始化权重，然后使用标注信息进行有监督训练。

对于自编码器的少量高阶特征重构表示，通常会加以限制，主要有2种：1. 限制隐藏层节点数，如使中间隐藏层的节点数小于输入输出节点数——降维（剔除部分次要特征）
，将一些相关度低的内容去除；2. 给数据加入噪声，即成为Denoising AutoEncoder（去噪自编码器），在噪音中学习特征，最常用的是加性Gauss噪声（Additive Gaussian Noise，AGN）


去噪自编码器是最具代表性，使用最广泛的自编码器之一，其他的几种自编码器都可在此基础上微调。主要步骤如下：
1. 导入常用库，其中有：Numpy，SciKit-Learn中的preprocessing模块，tensorflow，以及tensorflow中的MNIST数据集
2. 定义参数初始化方法，这里使用Xavier initialization
3. 定义去噪自编码的class，方便后面使用，此类包含一个构建函数 _init_()以及一些常用的成员函数。其中_init_中包含 n_input(输入变量数）n_hidden（隐含层节点数）、transfer_function（隐含层激活函数，默认softplus）、optimizer（优化器，默认Adam）、Scale（Gauss噪声系数，默认0.1）等
4. 定义网络结构，为输入x创建一个维度为n_input的placeholder
5. 定义自编码器的损失函数，这里使用平方误差作为cost 


```python
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

def xavier+init(fan_in, fan_out, constant = 1):
  low = -cosntant * np.sqrt(6.0 / (fan_in + fan_out))
  high = cosntant * np.sqrt(6.0 / (fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)
  
class AdditiveGaussNoiseAutoEncoder(object):
  def _init_(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.nn.train.AdamOptimizer(), scale = 0.1):
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.transfer = transfer_function
    self.scale = tf.placeholder(tf.float32)
    self.training_scale = scale
    network_weights = self.initialize_weights()
    self.weights = network_weights
    
    self.x = tf.placeholder(tf.float32, [None, self.n_input])
    self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,), self.weights['w1']), self.weights['b1']))
    self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
    
    self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
    self.optimizer = minimize(self.cost)
    
    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)
    
  def _initialize_weights(self):
    all_weights = dict()
    all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
    all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
    all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
    all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
    return all_weights
  
  def partial_fit(self, X):
    cost, opt = self.sess.run((self.cost, self.optimizer),feed_dict = {self.x: X, self.scale: self.training_scale})
    return cost
    
  def transform(self, X):
    return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})
    
  def generate(self, hidden = None):
    if hidden is None:
      hidden = np.random.normal(size = self.weights['b1'])
    return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    
  def reconstruct(self, X):
    return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.train_scale})
    
  def getWeights(self):
    return self.sess.run(self.weights['w1'])
    
  def getBialse(self):
    return self.sess,run()
      


```



最近读了一些经济类的书，mark一下。
关于重农主义
关于亚当斯密的理论
关于李嘉图的理论
