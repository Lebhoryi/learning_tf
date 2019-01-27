'''
摘要： Hello, Tensorflow

作者：lebhoryi@gmail.com
时间：2019/01/27
'''


import tensorflow as tf
hello = tf.constant("Hello, Tensorflow!")
with tf.Session() as sess:
    print(sess.run(hello))
    
