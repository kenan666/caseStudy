import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设定  mnist  数据集相关参数

# input_node  输入层的节点数。  对于mnist数据集来说，这个就等于图片的像素
INPUT_NODE = 784
#  输出层的节点数  。  这个等于类别的数目 。因为mnist 数据集中  需要分别的是  0-9  这10个数字，所以输出层的节点数为10
OUTPUT_NODE = 10

#  配置神经网络参数
#隐藏层节点数  。这里使用  只有一个隐藏层的网络结构   设置节点为  500个节点
LAYER1_NODE = 500

#一个训练  batch  中训练数据集的个数。  数字越小，训练过程越接近随机下降梯度，   数字越大时，训练越接近梯度下降
BATCH_SIZE = 100

#基础学习率
LEARNING_RATE_BASE = 0.8

#  学习率的衰减率
LEARNING_RATE_DECAY = 0.99

# 描述  模型  复杂度的正则化项  在损失函数中的系数
REGULARIZATION_RATE = 0.0001
#  训练轮数
TRAINING_STEPS = 30000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

'''
一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
  1- 在这里定义一个ReLU激活函数的三层神经网络。通过加入隐藏层实现了剁成神经网络结构
  2- 通过ReLU激活函数实现了去线性化，在这个函数中也支持传入用于计算参数平均值的类，方便在测试中使用滑动平均模型
'''
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    # 当没有提供滑动 平均类型的时候，直接使用当前参数的值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用  ReLU激励函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
    
        '''
        计算输出层  的前向传播结果。  因为计算损失函数时  会一并计算softmax 函数   ，所以在这里不需要加入激活函数，而且不加入softmax
        不会影响预测结果。因为预测结果使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果  不会产生影响。
        所以在整个神经网络的前向传播时，可以不加入最后的softmax层
        '''
        return tf.matmul(layer1,weights2) + biases2
    
    else:
        # 首先使用avg_class。average  函数来计算变量的滑动平均值，然后在计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

#  训练  模型 过程
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')
    
    #  生成 隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
    
    #生成  输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    
    #  计算在当前参数下神经网络前向传播的结果。   其中这里给出的用于计算滑动平均的类 为None  。  所以函数不会使用参数的滑动平均值
    y = inference(x,None,weights1,biases1,weights2,biases2)
    
    #  定义存储  训练 轮数  的变量 。  这个变量不需要计算滑动平均值，所以这里制定这个变量为 不可训练变量，（trainable = False）
    #  在使用TensorFlow  训练神经网络的时候，一般会将代表训练轮数的变量指定为补课训练的参数
    global_step = tf.Variable(0,trainable= False)
    
    #  给定滑动平均衰减率  和  训练  变量， 初始化平均类
    #  其中通过给定  训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    
    '''
    在所有代表神经网络参数的变量上使用滑动平均。
    其他辅助变量（比如global_step ）就不需要了
    tf.trainable_variables  返回的就是图上集合，
    GraphKeys.TRAINABLE_VARIABLES 中的元素 。这个元素就是所有没有指定  trainable = False  的参数
    '''
    variables_average_op = variable_average.apply(tf.trainable_variables())
    
    '''
    计算使用了滑动平均之后的前向传播结果。
    滑动平均不会改变变量自身的取值，而是维护一个影子变量来记录其滑动平均值。  所以当需要使用滑动平均值时，需要明确调用average函数
    '''
    average_y = inference (x,variable_averages,weights1,biases1,weights2,biases2)
    
    '''
    计算 交叉熵 作为 刻画预测值和真实值之间差距的损失函数。
    这里使用了TensorFlow中提供的sparse_softmax_cross_entropy_with_logits 函数来计算交叉熵
    当分类问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。
    MNIST问题的图片中只包含  0  -  9  中的一个数字，所以可以使用这个函数来计算交叉熵损失。
    这个函数的第一个参数是神经网络不包含softmax  层  的前向传播结果。第二个是训练数据的正确答案。
    因为标准答案是一个长度为10的一位数组，而该函数需要提供的是一个正确答案的数字，所以需要使用tf.argmax 函数来得到正确答案对应的类别编号
    '''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= y ,labels=tf.argmax(y_,1))
    # 计算当前batch 中所有样例的交叉熵的 平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 计算L2 正则化 损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    # 计算 模型  的正则化损失  。一般 只计算神经网络边上权重的正则化损失，而不使用偏置顶
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失  等于交叉熵 损失  和  正则化 损失的和
    loss = cross_entropy_mean + regularization
    
    #  设置 指数衰减的学习率
   
    #四个参数分别为  基础学习率  ，迭代轮数  过完所有数据所需要的迭代轮数   学习衰减率   
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    
#  使用tf.train.GradientDescentOptimizer 优化算法   来优化损失函数， 这里损失函数包含了交叉熵损失  和  L2 正则化损失
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    
'''
在训练神经网络模型的时候，每过一遍数据急需要通过反向传播来更新  神经网络中的参数，有需要更新每一个参数的滑动平均值,
为了一次完成多个操作，TensorFlow 提供了tf.control_dependencies  和  tf.group  两种机制  
'''
# train_op = tf.group(train_step,variables_average_op)
with tf.control_dependencies([train_step,variables_average_op]):
    train_op = tf.no_op(name = 'train')
    
'''
检查使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y,1)计算每一个样例的预测答案。
其中average_y是一个batch_size * 10 的二维数组。每一行表示一个样例的前向传播结果。
tf.argmax 的第二个参数  1  表示选取的最大值的操作仅在第一个维度中进行，也就是说，只在每一行选取最大值对应的下标。
于是得到的结果是一个长度为batch 的一维数组，这个一维数组中的值就表示了每一个阳历对应的数字识别结果。
tf.equal 判断两个张量的每一维是否相等，如果相等返回True，否则返回False
'''
correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

# 这个运算首先将一个布尔型的竖直转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
accurancy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#  初始化会话并开始训练过程
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # 准备验证数据 。一般神经网络的训练过程中 会通过验证数据大致判断体质条件 和  评判训练的效果
    validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
    
    #  准备测试数据。  在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的最后评判标准
    test_feed = {x:mnist.test.images,y_:mnist.test.labels}
    
    # 迭代训练神经网络
    for i in range (TRAINING_STEPS):
        #  每 1000轮输出一次结果
        if i % 100 == 0:
            '''
            计算平均滑动模型在验证数据上的结果。
            mnist 数据集比较小，所以一次可以处理所有的验证数据。  为了计算方便，案例中将数据划分为更小的batch。
            当神经网络模型比较复杂或者验证数据比较大时，太大的batch会导致计算时间过长甚至发生内存外溢的错误。
            '''
            validate_acc = sess.run(accurancy,feed_dict = validate_feed)
            print("After %d training step(s) ,validation accuracy using average model is &g " % (i,validate_acc))
        # 产生这一轮使用的一个batch 的训练数据，并运行训练过程
        xs,ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_op,feed_dict = {x:xs,y_:ys})
    
    # 在训练结束后，在测试数据上检测神经网络模型的最终正确率
    test_acc = sess.run(accurancy,feed_dict = test_feed)
    print ("After %d training step(s) ,test accurancy using average model is %g " % (TRAINING_STEPS,test_acc))
    
#主程序
def main(argv = None):
    # 声明mnist数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets('C:\\Users\\KeNan\\Desktop\\TF_Google\\CH5_手写识别_training',one_hot = True)
    train(mnist)
    
#tensorflow 提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ =='__mian__':
    tf.app.run()