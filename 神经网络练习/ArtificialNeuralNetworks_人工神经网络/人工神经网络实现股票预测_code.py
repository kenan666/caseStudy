import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

date = np.linspace(1,15,15)
endprice = np.array([2511.90,2538.26,2510.68,2591.66,2732.98,2701.69,2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,
                     2823.58,2864.90,2919.08])
beginprice = np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,2697.47,2695.24,2678.23,2722.13,2674.93,2744.13,
                       2717.46,2832.73,2877.40])
print (date)
plt.figure()
for i in range(0,15):
    # 柱状图
    dateone = np.zeros([2])
    dateone[0] = i;
    dateone[1] = i;
    priceone = np.zeros([2])
    priceone[0] = beginprice[i]
    priceone[1] = endprice[i]
    if endprice[i] > beginprice[i]:
        plt.plot(dateone,priceone,'r',lw = 8)
    else:
        plt.plot(dateone,priceone,'g',lw = 8)
#plt.show()
#  三层结构  15*1   1*10  15*1
datenormal = np.zeros([15,1])
pricenormal = np.zeros([15,1])
for i in range(0,15):
    datenormal[i,0] = i/14.0;
    pricenormal[i,0] = endprice[i]/3000.0;
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
#B
w1 = tf.Variable(tf.random_uniform([1,10],0,1))
b1 = tf.Variable(tf.zeros([1,10]))
wb1 = tf.matmul(x,w1)+b1
layer1 = tf.nn.relu(wb1)  #激励函数
# B
w2 = tf.Variable(tf.random_uniform([10,1],0,1))
b2 = tf.Variable(tf.zeros([15,1]))
wb2 = tf.matmul(layer1,w2)+b2
layer2 = tf.nn.relu(wb2)
loss = tf.reduce_mean(tf.square(y-layer2))  #  y  真实  lyear2 计算
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #  梯度下降法  训练算法
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0,10000):
        sess.run(train_step,feed_dict = {x:datenormal,y:pricenormal})
    #  得到好的  wb1  wb2  A+wb --> layer2
    pred = sess.run(layer2,feed_dict = {x:datenormal})
    predprice = np.zeros([15,1])
    for i in range(0,15):
        predprice[1,0] = (pred * 3000)[i,0]
    plt.plot(date,predprice,'b',lw = 1)
plt.show()