'''
摘要：采用的是3-1 线性回归模型，多了模型保存和恢复

参考：Tensorflow模型的保存与恢复
链接：https://blog.csdn.net/zhouyidaniuniu/article/details/80335808

作者：lebhoryi@gmail.com
时间：2019/01/27

'''

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

#### 数据准备
# 训练数据x
train_X = np.linspace(-1, 1, 100)
# 训练标签y，y=2x，但是加入了噪声
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
# 显示模拟数据点
plt.plot(train_X, train_Y, 'ro', label="Original data")
# 展示出每个数据对应的名称
plt.legend()
# 显示整个图
plt.show()

#### 创建模型
# placeholder变量
# 训练x，y占位符
X = tf.placeholder(dtype="float32", name="x")
Y = tf.placeholder(dtype="float32", name="y")
# 模型参数
weight = tf.Variable(tf.random_normal([1]), name="weight")
bias = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
y_pre = tf.multiply(X, weight) + bias

# 反向优化
# 损失值计算
cost = tf.reduce_mean(tf.square(Y - y_pre))
# 学习率
learning_rate = 0.01
# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

#### 迭代训练
# 训练参数
# 将所有数据训练的轮数
training_epochs = 20
# 每隔几个epoch展示一次数据
display_step = 2

# Save参数设定
saver = tf.train.Saver()
# 设置保存路径
savedir = "./log"

# 开启Session
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)

    plotdata = {"batchsize":[], "loss":[]}
    # 训练所有的数据
    for epoch in range(training_epochs):
        # fit train data
        # zip:打包为元组的列表
        for (x, y) in zip(train_X, train_Y):
            # start training
            sess.run(optimizer, feed_dict={X:x, Y:y})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print(f"Epoch: {epoch+1}; loss: {loss}; weight: {sess.run(weight)};"
                  f" bias= {sess.run(bias)}")

            if not loss == "NA":
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print("Finished!")
    # 保存模型
    saver.save(sess, os.path.join(savedir, "linermodel.ckpt"))
    # print(f"loss: {sess.run(cost, feed_dict={X:train_X, Y:train_Y})}")
    print(f"loss: {cost.eval(feed_dict={X:train_X, Y:train_Y})}; weight: {weight.eval()}; bias: {bias.eval()}")

    # 图形化显示
    plt.plot(train_X, train_Y, "ro", label="Original data")
    plt.plot(train_X, sess.run(weight) * train_X + sess.run(bias), label="Fitted line")
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], "b--")
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training loss.")

    plt.show()

    print(f"x=0.2, y_pre={sess.run(y_pre, feed_dict={X:0.2})}")

# 重启一个session，恢复模型
with tf.Session() as sess2:
    sess2.run(init)
    saver.restore(sess2, os.path.join(savedir, "linermodel.ckpt"))
    print(f"x=0.2, y_pre={sess2.run(y_pre, feed_dict={X:0.2})}")
