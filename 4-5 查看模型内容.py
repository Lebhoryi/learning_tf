'''
摘要： 将保存的线性回归模型里面的内容打印出来

作者：lebhoryi@gmail.com
时间：2019/01/28
'''

import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


savedir = "./log"
print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt", None, True)
# print_tensors_in_checkpoint_file(os.path.join(savedir, "linermodel.ckpt"), "weight", False)


#### 储存固定的weight和bias进模型，并打印出来
# W = tf.Variable(1.0, name="weight")
# b = tf.Variable(2.0, name="bias")
#
# # 放到一个字典里
# saver = tf.train.Saver({"weight":W, "bias":b})
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     saver.save(sess, os.path.join(savedir, "w_b.ckpt"))
#
# print_tensors_in_checkpoint_file(os.path.join(savedir, "w_b.ckpt"), None, True)
#
#### 一个快速找到ckpt文件的方式
#
# ckpt = tf.train.get_checkpoint_state(ckpt_dir)
# if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
