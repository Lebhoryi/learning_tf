# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:08:22 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf

tf.reset_default_graph() 

with tf.variable_scope("scope1") as sp:
     var1 = tf.get_variable("v", [1])

print("sp:",sp.name)    
print("var1:",var1.name)      

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])
    
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])
          
        with tf.variable_scope("23") :
            var4 = tf.get_variable("v4", [1])
            
print("sp1:",sp1.name)  
print("var2:",var2.name)
print("var3:",var3.name)
print("var4:",var4.name)
# with tf.variable_scope("scope"):
#     with tf.name_scope("bar"):
#         v = tf.get_variable("v", [1])
#         x = 1.0 + v
#         with tf.name_scope(""):
#             y = 1.0 + v
# print("v:",v.name)
# print("x.op:",x.op.name)
# print("y.op:",y.op.name)
