#!/usr/bin/python
# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#tensorflow实现softmax回归，线性分类器，单层
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()  #创建默认session

#前向传播
x = tf.placeholder(tf.float32, [None,784])  #placeholder，参数为类型+shape
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  #预测结果为一个10行向量
y_ = tf.placeholder(tf.float32, [None, 10]) #横向one-hot编码

#后向传播
right_prodict = tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])    #reduction_indices=1表示求和为1列，=0表示求和成一行
cross_entropy = tf.reduce_mean(-right_prodict)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  #梯度下降算法优化器，通过每次提供一个batch实现随机梯度下降SGD

#训练
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

#预测
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  #选取softmax的输出与one-hot编码对比
accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuray.eval({x:mnist.test.images, y_:mnist.test.labels}))





