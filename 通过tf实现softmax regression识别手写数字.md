### 关于Tensorflow的一些基本格式问题

1. 常量： tf.constan()
2. 变量： tf.Variable()
3. Session: tf.Session()/InteractiveSession
4. loss_function
5. Cross_Entropy
6. 优化算法
7. 全局参数初始化
8. 进行迭代训练
9. 模型准确率进行检验





#### 通过Tensorflow 实现SoftMax Regression识别手写数字

```python
# 1. 载入tf库
import tensorflow as tf   
# 2. 创建InteractiveSession，注册为默认的session，之后的运算默认都传回session，每个session相互独立
sess = tf.InteractiveSession()    

# 3. 创建placeholder，输入数据的地方，参数1表示数据类型，2为数据尺寸，NONE代表不限条数输入，784表示出入的为784维向量
x = tf.placeholder(tf.float32,[NONE,784])  

# 4. 对softmax函数的配置
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 5. 对loss function的配置： cross-entropy
y_ = tf.placeholder(tf.float32, [NONE, 784])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

# 6. 定义优化算法---随机梯度下降法SGD
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # 设置学习速率为0.5，优化目标为corss_entropy

# 7. 全局参数初始化
tf.global_variables_initializer().run()

# 8. 迭代进行训练
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})
  
# 9. 对模型的准确率进行验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 10. 统计全部样本预测的accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 11. 将测试数据特征和label输入测评流程accuracy，并打印，得准确率
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

```
