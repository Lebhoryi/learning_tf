'''
摘要: log 日志级别设置

    os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

参考：https://www.zhihu.com/question/268375146/answer/357375938

测试代码:
    测试Tensorflow 调用gpu还是cpu,查看输出信息是否简化

个人理解:
    Tensorflow默认的logging information太多,去除掉多余的log信息,只保留所需部分,例如如下测试是否调用gpu代码

Author: Lebhoryi@gmail.com
Github: https://github.com/Lebhoryi/learning_tf
Date:   2019/02/26

'''

import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

import tensorflow as tf


with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:1'):
    c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))

'''
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

结果:
(tf) dl@dl:~/Lebhoryi$ python test_gpu.py 
2019-02-26 09:42:08.102487: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2019-02-26 09:42:08.102516: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2019-02-26 09:42:08.102522: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2019-02-26 09:42:08.102528: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2019-02-26 09:42:08.102535: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2019-02-26 09:42:08.216220: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-02-26 09:42:08.216546: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.645
pciBusID 0000:01:00.0
Total memory: 10.90GiB
Free memory: 10.55GiB
2019-02-26 09:42:08.216563: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2019-02-26 09:42:08.216569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2019-02-26 09:42:08.216578: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0
2019-02-26 09:42:08.307192: I tensorflow/core/common_runtime/direct_session.cc:300] Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0

init: (NoOp): /job:localhost/replica:0/task:0/gpu:0
2019-02-26 09:42:08.308285: I tensorflow/core/common_runtime/simple_placer.cc:872] init: (NoOp)/job:localhost/replica:0/task:0/gpu:0
add: (Add): /job:localhost/replica:0/task:0/gpu:0
2019-02-26 09:42:08.308302: I tensorflow/core/common_runtime/simple_placer.cc:872] add: (Add)/job:localhost/replica:0/task:0/gpu:0
b: (Const): /job:localhost/replica:0/task:0/cpu:0
2019-02-26 09:42:08.308312: I tensorflow/core/common_runtime/simple_placer.cc:872] b: (Const)/job:localhost/replica:0/task:0/cpu:0
a: (Const): /job:localhost/replica:0/task:0/cpu:0
2019-02-26 09:42:08.308319: I tensorflow/core/common_runtime/simple_placer.cc:872] a: (Const)/job:localhost/replica:0/task:0/cpu:0
[2. 4. 6.]

'''

'''
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

结果:
(tf) dl@dl:~/Lebhoryi$ python test_gpu.py 
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0
init: (NoOp): /job:localhost/replica:0/task:0/gpu:0
add: (Add): /job:localhost/replica:0/task:0/gpu:0
b: (Const): /job:localhost/replica:0/task:0/cpu:0
a: (Const): /job:localhost/replica:0/task:0/cpu:0
[2. 4. 6.]

'''