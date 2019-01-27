'''
摘要：指定GPU运行

作者：lebhoryi@gmail.com
时间：2019/01/27
'''

import tensorflow as tf

# log_device_placement=True：是否打印设备分配日志。
# allow_soft_placement=True：如果指定的设备不存在，允许TF自动分配设备
# config = tf.ConfigProto(log_decive_placement=True, allow_soft_placement=True)
# 按需分配
# config.gpu_options.allow_growth = True
# 以给GPU分配固定大小的计算资源。
# config.per_process_gpu_memory_fraction = True
# sess = tf.Session(config=config))

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add = tf.add(a, b)
    print(sess.run(add, feed_dict={a:10, b:12}))

