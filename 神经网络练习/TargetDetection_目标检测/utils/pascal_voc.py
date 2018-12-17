'''
数据格式转换 --》voc 格式

坐标提取

坐标转换
'''

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import utils.config as Config

class VOC:
    # 将 voc 数据转换成 模型输入输出格式
    def __init__(self):
        self.data_path = Config.data_path
        self.image_path = Config.image_path
        self.image_size = Config.image_size
        self.cell_size = Config.cell_size
        self.classes = Config.classes_name
        self.classes_to_id = Config.classes_dict
        self.flipped = Config.flipped

        #  label  格式 shape[batch，7,7,25]
        # 训练数据
        self.gt_labels_train = None

        # 验证数据
        self.gt_labels_val = None
        self.prepare()

    #  读取图片
    def read_image(self,imname,flipped = False):
        image = cv2.imread(imname)
        image = cv2.resize(image,(self.image_size,self.image_size))
        if flipped:
            image = image[:,::-1,:]
        return image
    def prepate(self):
        get_labels_train,get_labels_val = self.load_labels()
        train_file = Config.train_path
        val_file = Config.val_path
        if not (os.path.isfile(train_file) and os.path.isfile(val_file)):
            if self.flipped:
                print('将水平方翻转图像加入数据集')
                get_labels_cp = copy.deepcopy(get_labels_train[:len(get_labels_train)//2])

                for idx in range (len(get_labels_cp)):
                    get_labels_cp[idx]['flipped'] = True
                    get_labels_cp[idx]['label'] = get_labels_cp[idx]['flipped'][:,::-1,:]

                    for i in range(self.cell_size):
                        for j in range (self.cell_size):
                            if get_labels_cp[idx]['label'][i][j][0] == 1:
                                get_labels_cp[idx]['label'][i][j][4] = self.image_size-1-get_labels_cp[idx]['label'][i][j][1]
                get_labels_train += get_labels_cp
            np.random.shuffle(get_labels_train)   

            #  将数据处理后的结果保存
            print ('保存训练数据 ：' + Config.train_path)
            with open(Config.train_path,'wb') as f :
                pickle.dunp(get_labels_val,f)

        self.gt_labels_train = get_labels_train
        self.gt_labels_val = get_labels_val

    #  提取坐标
    def load_pascal_annotation(self,index):
        imname = os.path.join(self.image_path,index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size,self.cell_size,25))
        filename = os.path.join(self.data_path,'Annotations',index +'.xml')
        tree = ET.parse(filename)

        # 确定所有目标
        objs = tree.findall('object')

        for obj in objs:
            #  寻找坐标
            bbox = obj.find('bndbox')

            #  转换为  image_size  尺寸
            x1 = max(min((float(bbox.find('xmin').text)-1) * w_ratio,self.image_size - 1),0)
            y1 = max(min((float(bbox.find('ymin').text)-1) * h_ratio,self.image_size - 1),0)
            x2 = max(min((float(bbox.find('xmax').text)-1) * w_ratio,self.image_size - 1),0)
            y2 = max(min((float(bbox.find('ymax').text)-1) * w_ratio,self.image_size - 1),0)

            #  目标类别
            class_id = self.classes_to_id[obj.find('name').text.lower().strip()]

            #  将坐标转换成 中心点 形式
            boxes = [(x2 + x1) / 2.0 , (y2 + y1) / 2.0 ,x2 - x1,y2 - y1]

            # 按  ceil 分坐标
            x_id = int(boxes[0] * self.cell_size/self.image_size)
            y_id = int(boxes[0] * self.cell_size/self.image_size)
            if label[y_id,x_id,0] == 1:
                continue
            label[y_id,x_id,0] = 1
            label[y_id,x_id,1:5] = boxes
            label[y_id,x_id,5 + class_id] = 1
        return label, len(objs)

    #  加载数据
    def load_labels(self):
        train_file = Config.train_path
        val_file = Config.val_path

        # 若数据格式已经处理好---》直接加载数据
        if os.path.isfile(train_file) and os.path.isfile(val_file):
            print ('训练数据加载中 ：' + train_file)
            with open(train_file,'rb') as f1:
                get_labels_train = pickle.load(f1)
            print ('验证数据加载中 ：' + val_file)
            with open(val_file,'rb') as f2:
                get_labels_val = pickle.load(f2)
            return get_labels_train,get_labels_val
        
        # 将voc数据处理成模型  需要的  格式
        print('处理数据 ： '+ self.data_path)
        self.image_index = os.listdir(Config.image_path)
        self.image_index = [i.replace('.jpg','') for i in self.image_index]
        import random 
        random.shuffle(self.image_index)

        #  划分训练集  和  测试集
        train = int(len(self.image_index) * (1 - Config.train_percentage))
        self.image_train_index = self.image_index[train:]
        self.image_val_index = self.image_index[:train]

        get_labels_train = []
        get_labels_val = []
        for index1 in self.image_train_index:
            label,num = self.load_pascal_annotation(index1)
            if num == 0:
                continue
            imname = os.path.join(self.image_size,index1 + '.jpg')
            get_labels_train.append({'imname':imname,'label':label,'flipped':False})

        for index2 in self.image_train_index:
            label,num = self.load_pascal_annotation(index2)
            if num == 0:
                continue
            imname = os.path.join(self.image_size,index2 + '.jpg')
            get_labels_train.append({'imname':imname,'label':label,'flipped':False})

        return get_labels_train,get_labels_val