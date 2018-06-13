#coding=utf-8
#多层感知机,MLP
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST/', one_hot=True)
sess = tf.InteractiveSession()

#前向传播
input_units = 784
h1_units = 300
x = tf.placeholder(tf.float32, [None, input_units])
W1 = tf.Variable(tf.truncated_normal([input_units,h1_units]))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.truncated_normal([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob=keep_prob)  #一定比例的树突随机失活
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 10])

#反向传播
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

#训练
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys, keep_prob:0.75})

#预测
#此时模型数据已经迭代更新完毕，接下来用于计算test.label
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))    #测试集x传入correct_prediction中y的x










