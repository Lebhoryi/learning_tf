'''
摘要: 对Cifar-10 进行一系列操作学习

作者: Lebhoryi@gmail.com
时间: 2019/02/26

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from cifar10 import cifar10_input
import tensorflow as tf
import pylab

# 读取数据
batch_size = 128
data_dir = "/tmp/cifar10_data/cifar-10-batches-bin/"
# 如果使用训练数据集， 可以将第一个参数传入eval_data=False
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# tf.train.start_queue_runners()
# image_batch, label_batch = sess.run([images_test, labels_test])

# print("========\n", image_batch[0])
# print("========\n", label_batch[0])
# pylab.imshow(image_batch[0])
# pylab.show()


with tf.Session() as sess:
   tf.global_variables_initializer().run()
   tf.train.start_queue_runners()
   image_batch, label_batch = sess.run([images_test, labels_test])
   print("__\n",image_batch[0])

   print("__\n",label_batch[0])
   pylab.imshow(image_batch[0])
   pylab.show()
