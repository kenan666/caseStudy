


import SimpleITK as sitk
import cv2
import os 

'''
1、批量处理文件，要求将所有子文件夹中的PV-ART_fixed.nii.gz 3D 图像文件转换为  2d  图像数据，并编号方式为  子文件夹名+index
  1、首先要批量处理得到  2d 的数据文件 
  2、在命名的时候要得到当前子文件夹的名字
2、将所有输出图像数据分类存储到csv文件  

3、liver-dom，liver-peak，标记为1，< liver-peak  标记为0，>  liver-dom  标记为2  标记以及排序存储未完成

'''

path = 'D:/image_data/Liver_Registration/my_test_withLiverMask1/'

def read_all_file(path):
    for folder_name in os.listdir(path):
        folder_dir = os.path.join(path,folder_name)
        for root,dirs,files in os.walk(folder_dir,topdown=False):
            if files:
                for file in files:
                    #print('filename', file)
                    #  获取对应的文件数据  PV-ART_fixed.nii.gz
                    if file == "PV-ART_fixed.nii.gz":
                        ff = os.path.join(folder_dir,file)
                        print(ff)
                        img = sitk.ReadImage(ff)
                        img_array = sitk.GetArrayFromImage(img)
                        #print(img_array)                        
                        frame_num,width,height = img_array.shape
                        #  创建新文件
                        fol_name = root.split('/')[-1]
                        print(root.split('/')[-1])
                        
                        root1 = "D:/data_img2/" + root.split('/')[-1] + '/'
                        #root1 = "D:/data_img2/" + root.split('/')[-1]
                        
                        #  将新文件写入对应的文件夹
                        index = 0                        
                        for img_item in img_array:
                            index = index + 1
                            cv2.imwrite(root1 + fol_name+ '_'+ str(index)+'.png',img_item)
                            
                        if not os.path.exists(root1):
                            os.makedirs(root1)                            
                        print("done!")  
                        
path1 = 'D:/data_img2/'                   
def write_to_csv(path1):    
    fopen = open('D:/data_img/data.csv','w')
    for root,dirs,files in os.walk(path1,topdown=False):        
        if files:
            a = files
            a = sort_string(a)
            print(a,type(a)) 
            
            for file in a:
                string = file
                file = file.split('.')[0]
                fopen.write(file + '\n')                
                  
                
        print("写入csv完成-done!")
        
#  按照数字进行排序  0,1,2,3,4.。。。
def order_file(file):
    re_digits = re.compile(r'(\d+)')  #  正则表达式  匹配数字
    pieces = re_digits.split(file)
    pieces[1::2] = map (int,pieces[1::2])
    return pieces[-2]
def sort_string(lst):
    return sorted(lst, key=order_file)
        
if __name__ == "__main__":
    read_all_file(path)
    write_to_csv(path1)
    
    
if __name__ == "__main__":
    read_all_file(path)
    write_to_csv(path1)