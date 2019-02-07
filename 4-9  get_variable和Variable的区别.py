# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 05:05:33 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import tensorflow as tf

tf.reset_default_graph()

var1 = tf.Variable(1.0 , name='firstvar')
print ("var1:",var1.name)
var1 = tf.Variable(2.0 , name='firstvar')
print ("var1:",var1.name)
var2 = tf.Variable(3.0 )
print ("var2:",var2.name)
var2 = tf.Variable(4.0 )
print ("var1:",var2.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=",var1.eval())
    print("var2=",var2.eval())



get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.3))
print ("get_var1:",get_var1.name)

#get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.4))
#print ("get_var1:",get_var1.name)

get_var1 = tf.get_variable("firstvar1",[1], initializer=tf.constant_initializer(0.4))
print ("get_var1:",get_var1.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("get_var1=",get_var1.eval())
    
    
    
