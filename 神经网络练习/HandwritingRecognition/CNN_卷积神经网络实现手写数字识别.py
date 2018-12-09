# cnn  本质  卷积运算
'''
     ABC 
    # A: 激励函数+矩阵 乘法加法
    # A CNN :  pool（激励函数+矩阵 卷积 加法）
    # C：激励函数+矩阵 乘法加法（A-》B）
    # C：激励函数+矩阵 乘法加法（A-》B） + softmax（矩阵 乘法加法）
    # loss：tf.reduce_mean(tf.square(y-layer2))
    # loss：code
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#  1  load  data 
mnist = input_data.read_data_sets('C:\\Users\\KeNan\\Desktop\\TL\\数字 _手写识别\\MNIST_data',one_hot=True)

#  input 
imageInput = tf.placeholder(tf.float32,[None,784])  #  图片大小  28 * 28
labelInput = tf.placeholder(tf.float32,[None,10])   # 类似knn

# 数据维度的调整  reshape
imageInputReshape = tf.reshape(imageInput,[-1,28,28,1])  #  [None,784]  ->  M  * 28 * 28 *1  2D  -> 4D  1  个通道

#  实现卷积运算  定义权重矩阵  -> 卷积内核 5*5   output  = 32
#  偏移矩阵
w0 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev = 0.1))
b0 = tf.Variable(tf.constant(0.1,shape = [32]))  
#  layer :激励函数  +  卷积运算  M * 28 * 28 *1   w0  :  5,5,1,32
layer1 = tf.nn.relu(tf.nn.conv2d(imageInputReshape,w0,strides=[1,1,1,1],padding='SAME')+b0)

# M  * 28 * 28 *32  
# 添加池化层  完成  采样   下采样   数据量减小了很多      (M  * 28 * 28 *32)/ksize  =  M  *  7  *  7  * 32  
#  池化层方法  就数据缩小  4  倍   
layer1_pool = tf.nn.max_pool(layer1,ksize=[1,4,4,1],strides = [1,4,4,1],padding = 'SAME')

#  实现第二层  输出层  layer2   out ： 激励函数  +  乘加运算   -》 （激励函数  +  乘加运算）- softmax
#  w1  2  维【7*7*32,1024】
w1 = tf.Variable(tf.truncated_normal([7*7*32,1024],stddev = 0.1))
b1 = tf.Variable(tf.constant(0.1,shape = [1024]))

#  将4  维  数据，转化为 2 维数据
h_reshape = tf.reshape(layer1_pool,[-1,7*7*32])  # M 7*7*32  -》N * N1
#  N * N1  [N*7*7*32]   [7*7*32,1024]  =N *1024
h1 = tf.nn.relu(tf.matmul(h_reshape,w1)+b1)

# 7.1 softmax
w2 = tf.Variable(tf.truncated_normal([1024,10],stddev = 0.1))
b2 = tf.Variable(tf.constant(0.1,shape = [10]))
pred = tf.nn.softmax(tf.matmul(h1,w2)+b2)  # N*1024   1024 *10  =  N*10
#  N *10 (概率)  N1  （0.1,0.2,0.4,0.1,0.2，。。。。）
#  label  [1*10]
loss0 = labelInput *tf.log(pred)
loss1 = 0

#  7.2  
for m in range(0,100):  #  test  100
    for n in range (0,10):
        loss1 = loss1 - loss0[m,n]
loss = loss1/100

#  train  
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#  9  run 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        images,labels = mnist.train.next_batch(500)
        sess.run(train,feed_dict = {imageInput:images,labelInput:labels})
        
        pred_test = sess.run(pred,feed_dict={imageInput:images,labelInput:labels})
        acc = tf.equal(tf.arg_max(pred_test,1),tf.arg_max(mnist.test.labels,1))
        acc_float = tf.reduce_mean(tf.cast(acc,tf.float32))
        acc_result = sess.run(acc_float,feed_dict={imageInput:mnist.test.images,labeInput:mnist.test.labels})
        print (acc_result)
        