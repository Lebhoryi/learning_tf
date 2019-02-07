# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 06:45:57 2017
@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import numpy as np
import tensorflow as tf 

# 1 创建图的方法
c = tf.constant(0.0)
print(c.graph)

g = tf.Graph()
with g.as_default():
  c1 = tf.constant(0.0)
  print(c1.graph)
  print(g)

g2 =  tf.get_default_graph()
print(g2)

# tf.reset_default_graph函数（见代码第15行），相当于重新建了一张图来代替原来的默认图
# 在使用tf.reset_default_graph函数时必须保证当前图的资源已经全部释放，否则会报错。
tf.reset_default_graph()
g3 =  tf.get_default_graph()
print(g3)

# 2.	获取tensor

print(c1.name)
t = g.get_tensor_by_name(name = "Const:0")
print(t)

# 3 获取op
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name,tensor1) 
test = g3.get_tensor_by_name("exampleop:0")
print(test)

print(tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")
print(testop)


with tf.Session() as sess:
    test =  sess.run(test)
    print(test) 
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print (test) 

#4 获取所有列表

#返回图中的操作节点列表
tt2 = g.get_operations()
print(tt2)
#5
tt3 = g.as_graph_element(c1)
print(tt3)
print("________________________\n")


#练习
with g.as_default():
 c1 = tf.constant(0.0)
 print(c1.graph)
 print(g)
 print(c.graph)
 g3 = tf.get_default_graph()
 print(g3)


































  