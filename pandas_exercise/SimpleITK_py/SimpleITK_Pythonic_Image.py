#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image',aspect = 'equal')
import SimpleITK as sitk
# Download data to work on
#%run update_path_to_download_script
from downloaddata import fetch_data as fdata

img = sitk.GaussianSource(size = [64] * 2)
plt.imshow(sitk.GetArrayViewFromImage(img))  #  sitk 利用GetArrayViewFromImage 函数显示当前图像

img = sitk.GaborSource(size = [64] * 2)
plt.imshow(sitk.GetArrayViewFromImage(img))
def myshow(img):
    nda = sitk.GetArrayViewFromImage(img)
    plt.imshow(nda)

myshow(img)

# Multi-dimension slice indexing  多维切片索引  多个维度，  x:y:z

'''
If you are familiar with numpy, sliced index then this should be cake for the SimpleITK image. 
The Python standard slice interface for 1-D object:

With this convenient syntax many basic tasks can be easily done.
'''

img[24,24]

#  cropping

myshow(img[16:48:])

myshow(img[:,16:-16])
myshow(img[:32,:32])

#  Flipping
img_corner = img[:32,:32]
myshow(img_corner)

myshow(img_corner[::-1,:])

myshow(sitk.Tile(img_corner,img_corner[::-1,::],img_corner[::,::-1],img_corner[::-1,::-1],[2,2]))

# Slice Extraction
# A 2D image can be extracted from a 3D one.

img = sitk.GaborSource(size = [64] * 3,frequency = 0.05)

myshow(img)

myshow(img[:,:,32])

myshow(img[16,:,:])


# Subsampling 
myshow(img[:,:,3,32])

# Mathematical Operators

img = sitk.ReadImage(fdata('a.png'))
img = sitk.Cast(img,sitk.sitkFloat32)   #  cast 函数数据类型转换
myshow(img)
img[150,150]

timg = img * 2
myshow(timg)
timg [150,150]

#  Division Operators  分类

#逻辑运算符
img = sitk.ReadImage(fdata('a.png'))

myshow(img)


#  比较运算符
'''
These comparative operators follow the same convention as the reset of SimpleITK for binary images. 
They have the pixel type of sitkUInt8 with values of 0 and 1.
'''
img = sitk.ReadImage(fdata('a.png'))
myshow(img)

myshow(img>90)
myshow(img>150)

myshow((img>90)+(img>150))