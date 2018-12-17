#  相关参数的定义

# 目标类别
classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 为目标类别编号
classes_no = [i for i in range(classes_name)]

# 类名  对应  编号
classes_dict = dict(zip(classes_name,classes_no))

#  统计数量
num_class = len(classes_name)

image_size = 448
cell_size = 7
box_per_cell = 2
alpha_relu = 0.1

#  损失函数中有无目标，目标类别 和选框 所占比例
object_scale = 2.0 
no_object_scale = 1.0
classes_dict = 2.0
coordinate_scale = 5.0

# 是否水平翻转图像
flipped = True

#  数据路径
data_path = 'C:/Users/KeNan/Desktop/TF_Case/'

#预训练，模型路径
small_path='C:/Users/KeNan/Desktop/TF_Case/'

#模型保存路径
model_path='C:/Users/KeNan/Desktop/TF_Case/model/'

#graph保存路径
train_graph='C:/Users/KeNan/Desktop/TF_Case/graph/train/'
val_graph='C:/Users/KeNan/Desktop/TF_Case/graph/val/'

#保存好模型格式的数据路径
train_path='C:/Users/KeNan/Desktop/TF_Case/data/train.pkl'
val_path='C:/Users/KeNan/Desktop/TF_Case/data/val.pkl'

#图片路径
image_path='C:/Users/KeNan/Desktop/TF_Case/picture/'

#保存数据为tfrecord的路径
tfrecord_path='C:/Users/KeNan/Desktop/TF_Case/data/tfrecord/'


decay_step=30000
decay_rate=0.92
momentum=0.5
learning_rate=0.0001
dropout=0.5
batch_size=16
epoch=1
#保存模型次数
checkpont=5

threshold=0.2
IOU_threshold=0.2
train_percentage=0.9

#测试数据使用model类型,1是使用预训练模型，2是使用自己训练模型
model_type='2'

#测试输出类型，1是自己的图片，2是视频
output_type='1'

#测试图片文件夹
picture='C:/Users/KeNan/Desktop/TF_Case/picture/'

#生成图片保存文件夹
output_path='C:/Users/KeNan/Desktop/TF_Case/output/'