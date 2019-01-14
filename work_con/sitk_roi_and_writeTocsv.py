import os
import json
import SimpleITK as sitk
import csv
import pandas as pd

def calculate(i, getVolname, volname, cen_posX, cen_posY, cen_posZ, dimX, dimY, dimZ):
    posx1 = int(cen_posX - dimX/2)
    posx2 = int(cen_posX + dimX/2)
    posy1 = int(cen_posY - dimY/2)
    posy2 = int(cen_posY + dimY/2)
    posz1 = int(cen_posZ - dimZ/2)
    posz2 = int(cen_posZ + dimZ/2)
    
    sitkImage = sitk.ReadImage(folder_dir + '/'+volname+'.nii')
    direction = sitkImage.GetDirection()
    origin = sitkImage.GetOrigin()
    spacing = sitkImage.GetSpacing()

    # 根据公式计算需要截取的图像的坐标
    xbegin = int(direction[0]*((posx1-origin[0])/spacing[0]))
    xend = int(direction[0] * (posx2 - origin[0]) / spacing[0])
    ybegin = int(direction[4] * (posy1 - origin[1]) / spacing[1])
    yend = int(direction[4] * (posy2 - origin[1]) / spacing[1])
    zbegin = int(direction[8] * (posz1 - origin[2]) / spacing[2])
    zend = int(direction[8] * (posz2 - origin[2]) / spacing[2])    

    return xbegin, xend, ybegin, yend, zbegin, zend

def read_json_calculate_readDate(folder_dir, file):
    a = os.path.join(folder_dir,file)
    with open (a) as f:
        getVolname = json.load(f)
        volname = str(getVolname['0']['VolumeName'])
        Volume2WindowLevel = str(getVolname['Volume2WindowLevel'][volname])
        count = getVolname['count']
        for i in range(count):

            cen_posX = getVolname[str(i)]['CenterX']
            cen_posY = getVolname[str(i)]['CenterY']
            cen_posZ = getVolname[str(i)]['CenterZ']
            dimX = getVolname[str(i)]['DimX']
            dimY = getVolname[str(i)]['DimY']
            if 'DimZ' in getVolname[str(i)]:
                dimZ = getVolname[str(i)]['DimZ']
            else:
                dimZ = 1

            xbegin, xend, ybegin, yend, zbegin, zend = calculate(i, getVolname, volname, cen_posX, cen_posY, cen_posZ, dimX, dimY, dimZ)
            coord = [xbegin,xend,ybegin,yend,zbegin,zend]
            readDate(i, volname, getVolname, Volume2WindowLevel, cen_posX, cen_posY, cen_posZ, dimX, dimY, dimZ, coord)
            
            storeImgFile(i, volname, xbegin, xend, ybegin, yend, zbegin, zend)



def readDate(i, volname, getVolname, Volume2WindowLevel, cen_posX, cen_posY, cen_posZ, dimX, dimY, dimZ, coord):

    global DF
    # 读取数据
    VerInfo = getVolname[str(i)]['VerifiedInfo']
    Causes = str(VerInfo['Causes'])
    Company = VerInfo['Company']
    Fibrosis_Stage = VerInfo['Fibrosis-Stage']
    HAI_Stage = VerInfo['HAI-Stage']
    Magnetic = VerInfo['Magnetic']
    Organ = VerInfo['Organ']
    Phases = VerInfo['Phases']

    index =i
    data = {'PatientID':patientID1,
            'Time':time,
                'nodule':index,
            'cen_posX':cen_posX,
                'cen_posY':cen_posY,
            'cen_posZ':cen_posZ,
                'dimX':dimX,'dimY':dimY,'dimZ':dimZ,
                'coord':[coord],
            'Causes':Causes,
                'Company':Company,'Fibrosis_Stage':Fibrosis_Stage,
                'HAI_Stage':HAI_Stage,'Magnetic':Magnetic,
                'Organ':Organ,'Phases':Phases,
                'VolumeName':volname,                                       
                'Volume2WindowLevel':Volume2WindowLevel}
#                             print(data)
    df1 = pd.DataFrame(data)
    DF = DF.append(df1,ignore_index = False)
    DF.to_csv('D:/12_data_roi/11_json.csv')   #  csv  存储路径
    print(DF)

def storeImgFile(i, volname, xbegin, xend, ybegin, yend, zbegin, zend):
    sitkImage = sitk.ReadImage(folder_dir + '/'+volname+'.nii')
    sitkImage1 = sitkImage[xbegin:xend,ybegin:yend,zbegin:(zend+1)]
    print([xbegin,xend,ybegin,yend,zbegin,zend+1])

    fol_name = root.split('/')[-1]
    root1 = "D:/12_data_roi/test/" + root.split('/')[-1] + '/'  # 存储文件夹命名
        #  生成文件夹--》ok
    if not os.path.exists(root1):
        os.makedirs(root1)

    filename = root1 + folder1_name + '_' + str(i) + '.nii.gz'
    sitk.WriteImage(sitkImage1,filename)
    print("done!")

if __name__ == '__main__':
    path = 'D:/12_data_roi/'  #根目录
    DF = pd.DataFrame()
    for folder_name in os.listdir(path):
        folder_dir = os.path.join(path,folder_name)
        for root,dirs,files in os.walk(folder_dir,topdown=False):
            if files:
                for file in files:
                    if file == folder_name + '_LiverMR_lwx.json':
                        lab1 = folder_name.split('_')[-3]
                        lab2 = folder_name.split('_')[-2]
                        folder1_name = lab1 + '_'+ lab2
                        print(folder_name)
                        patientID1 = folder_name.split('_')[-3]
                        time = folder_name.split('_')[-2]

                        read_json_calculate_readDate(folder_dir, file)