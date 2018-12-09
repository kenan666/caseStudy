#  KNN  最近 邻域 法  实现手写识别
#  1  加载数据  load  data、   
#  2  knn  test  train   -> distance
#  3  knn  找到  k  个 最近的图片  在500张  找到  4 张最相近的   500  -》 4
#  4  k  个最近的图片  -》parse  centent  label
#  5    label  -》  数字  p9  测试图片  -》  数据  判断是否正确  ？--》 #  6   统计概率
#  6  检测  概率 统计

import tensorflow as tf
import numpy as np
import random 
from tensorflow.examples.tutorials.mnist import input_data

#  load  data   2   one_hot  :1  0000   1  filename       数据加载
mnist = input_data.read_data_sets('C:\\Users\\KeNan\\Desktop\\TL\\数字 _手写识别\\MNIST_data',one_hot=True)
#  属性设置
trainNum = 55000
testNum = 10000
trainSize = 500  #  改变训练  样本大小  以及  测试样本大小  可查看概率变化     
testSize = 5      #  训练数据和测试数据可修改  修改的同时记得修改代码中数据  即可进行计算与检测
k = 4   #   k  值 可修改

#  data  分解  trainsize   范围  replace = False
trainIndex = np.random.choice(trainNum,trainSize,replace = False)
testIndex = np.random.choice(testNum,testSize,replace = False)
trainData = mnist.train.images[trainIndex] #  获取训练图片
trainLabel = mnist.train.labels[trainIndex] #  训练标签 
testData = mnist.test.images[testIndex]
testLabel = mnist.test.labels[testIndex]

print ('trainData.shape=',trainData.shape)  #  500 * 784     图片个数  *  784  像素点
print ('trainLabel.shape=',trainLabel.shape)  #   500  * 10   图片个数  *  
print ('testData.shape=',testData.shape)  #  5 * 784
print ('testLabel.shape=',testLabel.shape)  #  5* 10
print ('testLabel = ',testLabel)  # 4  3  6

#  tf  input    784 -> 表明一个完整的图片 image
trainDataInput = tf.placeholder(shape = [None,784],dtype = tf.float32)
trainLabelInput = tf.placeholder(shape = [None,10],dtype = tf.float32)
testDataInput = tf.placeholder(shape = [None,784],dtype = tf.float32)
testLabelInput = tf.placeholder(shape = [None,784],dtype = tf.float32)

#  KNN   距离  distance
#  test  train    之间的距离    5  500  784  （3D数据）  2500 组合  *  784  
f1 = tf.expand_dims(testDataInput,1)    #   维度转换
f2 = tf.subtract(trainDataInput,f1)   #  784  sun (784)
f3 = tf.reduce_sum(tf.abs(f2),reduction_indices=2)  #  完成数据累加  784  像素点之间的差值 abs（绝对值）
#   5  *  500
f4 = tf.negative(f3)   #  取反 
f5,f6 = tf.nn.top_k(f4,k=4)   # 选取  f4  中最大的四个值   == f3中最小的四个值  
# f6  index  获取 标签   -  
f7 = tf.gather(trainLabelInput,f6)
#  f8  num    获取
f8 = tf.reduce_sum(f7,reduction_indices=1)  #  竖直方向的累加
#  选取在某一个维度上最大的值   并且找出最大的 index 
f9 = tf.argmax(f8,dimension=1)   #所有的检测图片的  5  个  num

#  核心部分  距离的计算
with tf.Session() as sess:
    #   运行  f1    -》 testData   5 张图片
    p1 = sess.run(f1,feed_dict={testDataInput:testData[0:5]})
    print ('p1 = ',p1.shape)   #p1 =  (5, 1, 784)  
    p2 = sess.run(f2,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print ('p2 =',p2.shape)  #  p2 = (5, 500, 784)  实现test  train  之间像素作差的  效果
    p3 = sess.run(f3,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print ('p3 = ',p3.shape)   #  p3 =  (5, 500)  
    print ('p3[0,0]',p3[0,0])  #  p3[0,0] 123.47845   两者之间的距离  计算
    
    p4 = sess.run(f4,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print ('p4 = ',p4.shape)
    print ('p4[0,0]',p4[0,0])
    
    p5,p6 = sess.run((f5,f6),feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    #  p5  = (5,4)   5  行  4  列   每一张测试图片（5张）分别对应4张最近图片
    #  p6  = (5,4)
    print ('p5 = ',p5.shape)
    print ('p6 = ',p6.shape)
    print ('p5[0,0]',p5[0,0])
    print ('p6[0,0]',p6[0,0])  #  221  p6  index
    #print ('p5[0,0]',p5[0])
    #print ('p6[0,0]',p6[0])
    
    p7 = sess.run(f7,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print ('p7 = ',p7.shape)  # p7 =  (5, 4, 10)
    print ('p7[]',p7)
    
    p8 = sess.run(f8,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print ('p8 = ',p8.shape)
    print ('p8[] = ',p8)
    
    p9 = sess.run(f9,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print ('p9 = ',p9.shape)
    print ('p9[] = ',p9)
    
    p10 = np.argmax(testLabel[0:5],axis = 1)
    print ('p10[]=',p10)
j = 0
for i in range(0,5):
    if p10[i] == p9[i]:
        j = j+1
print ('ac = ',j *100 /5)