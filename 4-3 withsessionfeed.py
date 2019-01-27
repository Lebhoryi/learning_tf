'''
摘要： with tf.Session() as sess 用法

作者：lebhoryi@gmail.com
时间：2019/01/27
'''


import tensorflow as tf
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    print("add: ", sess.run(add, feed_dict={a:3, b:4}))
    print("multiply: ", sess.run(mul, feed_dict={a:3, b:4}))
