# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:53:41 2017

@author: 代码医生
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf


labels = [[0,0,1],[0,1,0]]
logits = [[2,  0.5,6],
          [0.1,0,  3]]
logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)


result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result2_1 = tf.reduce_sum(result2)
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)


with tf.Session() as sess:
    print ("scaled=",sess.run(logits_scaled))    
    print ("scaled2=",sess.run(logits_scaled2)) #经过第二次的softmax后，分布概率会有变化
    

    print ("rel1=",sess.run(result1),"\n")#正确的方式
    print ("rel2=",sess.run(result2),"\n")#如果将softmax变换完的值放进去会，就相当于算第二次softmax的loss，所以会出错
    print ("rel2_1=", sess.run(result2_1), "\n")
    print ("rel3=",sess.run(result3))


#
# #标签总概率为1
# labels = [[0.4,0.1,0.5],[0.3,0.6,0.1]]
# result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# with tf.Session() as sess:
#     print ("rel4=",sess.run(result4),"\n")
#
# #sparse
# labels = [2,1] #其实是0 1 2 三个类。等价 第一行 001 第二行 010
# result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# with tf.Session() as sess:
#     print ("rel5=",sess.run(result5),"\n")
#
# #注意！！！这个函数的返回值并不是一个数，而是一个向量，
# #如果要求交叉熵loss，我们要对向量求均值，
# #就是对向量再做一步tf.reduce_mean操作
# loss=tf.reduce_mean(result1)
# with tf.Session() as sess:
#     print ("loss=",sess.run(loss))
#
# labels = [[0,0,1],[0,1,0]]
# loss2 = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_scaled),1) )
# with tf.Session() as sess:
#     print ("loss2=",sess.run(loss2))