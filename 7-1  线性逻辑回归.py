# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:57:46 2017
@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


#模拟数据点   
def generate(num_classes, sample_size, mean, cov, diff, regression):
    # 定义生成类的个数
    # num_classes = 2 #len(diff)
    # 每个类别拥有500样本数
    samples_per_class = int(sample_size/2)

    # 生成一个多元正态分布矩阵 X0:500个一维二列数据, Y0:500个[0,0,0,0,...0]
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    # print(f"X0: \n{len(X0)}")
    # print(f"Y0: \n{len(Y0)}")
    # plt.figure()  #创建绘图对象
    # plt.plot(X0, Y0)
    # plt.scatter(X0[:, 0], X0[:, 1])
    # plt.show()


    for ci, d in enumerate(diff):
        # print(f"ci:{ci}")
        # print(f"d:{d}")

        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)
        # plt.scatter(X1[:, 0], X1[:, 1])
        # plt.show()
    
        # 数组拼接,axis=0
        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
        
    if regression==False: #one-hot  0 into the vector "1 0
        class_ind = [Y==class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    # 打乱顺序
    X, Y = shuffle(X0, Y0)
    
    return X,Y    

# 输入维度
input_dim = 2                    
# 定义随机数的种子值（这样可以保证每次运行代码时生成的随机值都一样）
np.random.seed(10)
# 定义生成类的个数num_classes=2
num_classes =2
# 随机生成两个固定数
mean = np.random.randn(num_classes)
# print(f"mean: {mean}")
# 生成对角矩阵, np.random.multivariate_normal需要
cov = np.eye(num_classes)
# print(f"cov: \n{cov}")
# 3.0是表明两类数据的x和y差距3.0。传入的最后一个参数regression =True表明使用非one-hot的编码标签。
# 生成1000个数据
X, Y = generate(num_classes, 1000, mean, cov, [3.0], True)
# 颜色选项
colors = ['r' if l == 0 else 'b' for l in Y[:]]
# # X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据
# print(f"X:{X}")
# print(f"Y:{Y}")
# plt.scatter(X[:,0], X[:,1], c=colors)  # 散点图
# plt.xlabel("Scaled age (in yrs)")  # x轴标签
# plt.ylabel("Tumor size (in cm)")  # y轴标签
# plt.show()  # 显示图
lab_dim = 1


# tf Graph Input
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])
# Set model weights
W = tf.Variable(tf.random_normal([input_dim,lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim]), name="bias")

output = tf.nn.sigmoid(tf.matmul(input_features, W) + b)
cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output))
ser = tf.square(input_labels - output)
loss = tf.reduce_mean(cross_entropy)
err = tf.reduce_mean(ser)
optimizer = tf.train.AdamOptimizer(0.04) #尽量用这个--收敛快，会动态调节梯度
train = optimizer.minimize(loss)  # let the optimizer train

epochs = 50  # 整个数据集迭代50次
minibatchSize = 25  # 每次的minibatchsize取25条

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        sumerr=0
        # 所有的数据按照batch循环
        for i in range(np.int32(len(Y)/minibatchSize)):
            # 将x切片操作,切出batch个数据
            x1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
            # 将y reshape
            y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            tf.reshape(y1,[-1,1])
            # 开始训练
            _, lossval, outputval, errval = sess.run([train,loss,output,err], feed_dict={input_features: x1, input_labels:y1})
            sumerr += errval

        print ("Epoch:", "{:.4d}".format(epoch+1), "cost=", "{:.9f}".format(lossval),"err=", sumerr/np.int32(len(Y)/minibatchSize))

#图形显示
    # 取100个测试点，在图像上显示出来，接着将模型以一条直线的方式显示出来
    train_X, train_Y = generate(100, mean, cov, [3.0], True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    #plt.colorbar()
    '''
    模型生成的z用公式可以表示为z=x1w1+x2*w2+b， 如果将x1和x2映射到直角坐标
    系中的x和y坐标， 那么z就可以被分为小于0和大于0两部分。 当z=0时， 就代表直线本身， 令上面
    的公式中z等于零， 就可以将模型转化成如下直线方程：
    x2=-x1* w1/w2-b/w2， 即： y=-x* （w1/w2） -（b/w2）
    '''

#    x1w1+x2*w2+b=0
#    x2=-x1* w1/w2-b/w2
    x = np.linspace(-1,8,200)
    y = -x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    plt.plot(x,y, label='Fitted line')
    plt.legend()
    plt.show()