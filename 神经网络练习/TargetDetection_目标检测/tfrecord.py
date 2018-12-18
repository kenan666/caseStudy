import os
import tensorflow as tf
import numpy as np
from utils.pascal_voc import VOC
import utils.config as Config

#  路径设置
trian_path = Config.tfrecord_path + 'train/'
val_path = Config.tfrecord_path + 'val/'
data = VOC()

# 数据读取
writer = tf.python_ioTFRecordWriter(trian_path + "train.tfrecords")
for k in range(len(data.gt_labels_train)):
    imnama = data.gt_labels_train['imname']
    flipped = data.gt_labels_train['flipped']
    image = data.read_image(image,flipped)
    label = data.gt_labels_train[k]['label']

    label_raw = label.tobytes()
    img_raw = image.tobytes()  #将图像转化成原生 bytes
    example = tf.train.Example(features = tf.train.Features(feature = {
            'label' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [label_raw])),
            'img_raw' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))
    }))

    writer.write(example.SerializeToString())
    print (k)
writer.close()

#  验证集 tfrecord
writer = tf.python_io.TFRecordWriter(val_path + 'val.tfrecords')
for k in range(len(data.gt_labels_val)):
    imname = data.gt_labels_val['imname']
    flipped = data.gt_labels_val['flipped']
    image = data.read_image(image,flipped)
    label = data.gt_labels_val[k]['label']

    label_raw = label.tobytes()
    img_raw = image.tobytes()  #将图像转化成原生 bytes
    example = tf.train.Example(features = tf.train.Features(feature = {
            'label' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [label_raw])),
            'img_raw' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))
    }))

    writer.write(example.SerializeToString())
    print (k)
writer.close()