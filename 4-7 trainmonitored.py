"""
摘要：tf.train.MonitoredTraining Session
    该函数可以直接实现保存及载入检查点模型的文件。与前面的方式不同，本例中并不是按
    照循环步数来保存，而是按照训练时间来保存的。通过指定save_checkpoint_secs参
    数的具体秒数，来设置每训练多久保存一次检查点。

作者：Lebhoryi@gmail.com
时间：2019/01/28
"""

import tensorflow as tf

tf.reset_default_graph()
global_steps = tf.train.get_or_create_global_step()
step = tf.assign_add(global_steps, 1)

with tf.train.MonitoredTrainingSession(checkpoint_dir="log/checkpoints", save_checkpoint_secs=2) as sess:
    print(sess.run([global_steps]))
    while not sess.should_stop():
        i = sess.run(step)
        print(i)