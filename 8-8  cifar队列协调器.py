
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:27:15 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
#放在cifar目录下
import  cifar10_input
import tensorflow as tf
import pylab 

#取数据
batch_size = 12
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    
    image_batch, label_batch = sess.run([images_test, labels_test])
    print("__\n",image_batch[0])
    
    print("__\n",label_batch[0])
    pylab.imshow(image_batch[0])
    pylab.show()
    coord.request_stop()
