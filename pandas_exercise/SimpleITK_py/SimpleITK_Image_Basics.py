from __future__ import print_function

import matplotlib.pyplot as plt
import SimpleITK as sitk

'''
Image Construction
There are a variety of ways to create an image. 
All images' initial value is well defined as zero.
'''
image = sitk.Image(256,128,64,sitk.sitkInt16)
image_2D = sitk.Image(64,64,sitk.sitkFloat32)
image_2D = sitk.Image([32,32],sitk.sitkUInt32)
image_RGB = sitk.Image([128,128],sitk.sitkVectorInt8,3)

'''
Accessing Attributes
If you are familiar with ITK, then these methods will follow your expectations:
'''
print ( image.GetSize())
print(image.GetDepth())
print(image.GetDirection())
print(image.GetOrigin())
print(image.GetSpacing())
print(image.GetNumberOfComponentsPerPixel())

# The size of the image's dimensions have explicit accessors
print(image.GetWidth())
print(image.GetHeight())
print(image.GetDepth())

# Since the dimension and pixel type of a SimpleITK image is determined at run-time accessors are needed.
print(image.GetDimension())
print(image.GetPixelIDValue())
print(image.GetPixelIDTypeAsString())

# get 2D image size and depth
print(image_2D.GetDepth())
print(image_2D.GetSize())

# get the dimension and size of a Vector image
print(image_RGB.GetDimension())
print(image_RGB.GetSize())

print(image_RGB.GetNumberOfComponentsPerPixel())

# For certain file types such as DICOM, additional information about the 
# image is contained in the meta-data dictionary.
for key in image.GetMetaDataKeys():
    print("\"{0}\":\"{1}\"".format(key, image.GetMetaData(key)))

# Accessing Pixels
help(image.GetPixel)

print(image.GetPixel(0,0,0))
image.SetPixel(0,0,0,1)
print(image.GetPixel(0,0,0))

print(image[0,0,0])
image[0,0,0] = 10
print(image[0,0,0])

#  Conversion between numpy and SimpleITK
nda = sitk.GetArrayFromImage(image)
print(nda)

help(sitk.GetArrayFromImage)

# Get a view of the image data as a numpy array, useful for display
nda = sitk.GetArrayViewFromImage(image)

nda = sitk.GetArrayFromImage(image_RGB)
img = sitk.GetImageFromArray(nda)
img.GetSize()

help(sitk.GetImageFromArray)

img = sitk.GetImageFromArray(nda, isVector=True)
print(img)

# The order of index and dimensions need careful attention during conversion

import numpy as np
multi_channel_3Dimage = sitk.Image([2,4,8],sitk.sitkVectorFloat32,5)

x = multi_channel_3Dimage.GetWidth() - 1
y = multi_channel_3Dimage.GetHeight() - 1
z = multi_channel_3Dimage.GetDepth() - 1

multi_channel_3Dimage[x,y,z] = np.random.random(multi_channel_3Dimage.GetNumberOfComponentsPerPixel())

nda = sitk.GetArrayFromImage(multi_channel_3Dimage)
print("Image size: " + str(multi_channel_3Dimage.GetSize()))
print("Numpy array size: " + str(nda.shape))

# Notice the index order and channel access are different:
print("First channel value in image: " + str(multi_channel_3Dimage[x,y,z][0]))
print("First channel value in numpy array: " + str(nda[z,y,x,0]))

sitk.Show(image)

import matplotlib.pyplot as plt
z = 0
slice = sitk.GetArrayViewFromImage(image)[z,:,:]
plt.imshow(slice)  