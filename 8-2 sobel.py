'''
摘要： 通过卷积操作来实现本章开篇所讲的sobel算子， 将彩色的图片生成带有边缘化信息的图片。
      本例中先载入一个图片， 然后使用一个“3通道输入， 1通道输出的3×3卷积核”（即sobel算
      子） ， 最后使用卷积函数输出生成的结果

作者：lebhoryi@gmail.com
时间：2019/02/27
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import tensorflow as tf

img_path = "~/Pictures/touxiang.jpg"
my_img = mping.imread(os.path.expanduser(img_path))  # 只能读取绝对路径

# plt.imshow(my_img)
# plt.axis("off")  # 不显示坐标轴
# plt.show()
# print(my_img.shape[0])
length = my_img.shape[0]
width = my_img.shape[1]

full = np.reshape(my_img, [1, length, width, 3])
# print(full.shape)
input_full = tf.Variable(tf.constant(1.0, shape=[1, length, width, 3]))

filter =  tf.Variable(tf.constant([[-1.0,-1.0,-1.0],  [0,0,0],  [1.0,1.0,1.0],
                                    [-2.0,-2.0,-2.0], [0,0,0],  [2.0,2.0,2.0],
                                    [-1.0,-1.0,-1.0], [0,0,0],  [1.0,1.0,1.0]],shape = [3, 3, 3, 1]))
# 3个通道输入，生成1个feature ma
train_op = tf.nn.conv2d(input_full, filter, strides=[1, 1, 1, 1], padding="SAME")
loss = tf.cast(((train_op-tf.reduce_min(train_op))/(tf.reduce_max(train_op)-tf.reduce_min(train_op)) ) *255 ,tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    t, f = sess.run([train_op, filter], feed_dict={input_full:full})
    # print(f"f:{f}")
    t = np.reshape(t, [length, width])

    plt.imshow(t, cmap="Greys_r")
    plt.axis("off")
    plt.show()
