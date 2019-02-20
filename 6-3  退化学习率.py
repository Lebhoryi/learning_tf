# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 06:36:33 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)

initial_learning_rate = 0.1 #初始学习率

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step,
                                           decay_steps=10,decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)

add_global = global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(20):
        g, rate = sess.run([add_global, learning_rate])
        print(g, rate)
