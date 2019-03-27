# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:33:30 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf  

#创建长度为100的队列  
queue = tf.FIFOQueue(100,"float")  

c = tf.Variable(0.0)  #计数器  
#加1操作 
op = tf.assign_add(c,tf.constant(1.0))  
#操作:将计数器的结果加入队列  
enqueue_op = queue.enqueue(c)  
  
#创建一个队列管理器QueueRunner，用这两个操作向q中添加元素。目前我们只使用一个线程:  
qr = tf.train.QueueRunner(queue,enqueue_ops=[op,enqueue_op]) 

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
       
    coord = tf.train.Coordinator()  
      
    ## 启动入队线程, Coordinator是线程的参数  
    enqueue_threads = qr.create_threads(sess, coord = coord,start=True)  # 启动入队线程  
      
    # 主线程  
    for i in range(0, 10):  
        print ("-------------------------")  
        print(sess.run(queue.dequeue()))  
      
     
    coord.request_stop()  #通知其他线程关闭 其他所有线程关闭之后，这一函数才能返回  


    #join操作经常用在线程当中,其作用是等待某线程结束  
    #coord.join(enqueue_threads) 

