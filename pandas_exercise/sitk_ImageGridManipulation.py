#  图像网格操作
'''
概述
有许多SimpleITK过滤器具有类似的功能，但非常重要的区别。将要比较的过滤器是：

JoinSeriesImageFilter() - 将多个ND图像连接成（N + 1）-D图像
ComposeImageFilter() - 将多个标量图像组合成多分量矢量图像
VectorIndexSelectionCastImageFilter() - 提取输入像素类型向量的选定索引（输入图像像素类型必须是向量，输出是标量）。
此外，此过滤器可以转换输出像素类型（SetOutputPixelType方法）。
ExtractImageFilter() - 使用向量将图像裁剪到选定的区域边界; 折叠尺寸，除非尺寸为2
CropImageFilter()- 类似于ExtractImageFilter()，但通过 itk::Size上限和下限裁剪图像
图像切片运算符 - 使用切片（）来提取图像的子区域img[i:j, k:l]
所有这些操作都将保持像素的物理位置，而不是修改图像的元数据。
'''
'''
合成滤镜
在JoinSeriesImageFilter()将N维中相同像素类型的多个图像合并到 N + 1 维的图像中时，ComposeImageFilter() 将标量图像组合成相同维度的矢量图像。
前者对于连接一系列连续图像很有用，而后者对于将同一对象的多个通道合并为一个图像（例如RGB）更有用。
'''

'''
提取过滤器
VectorIndexSelectionCastImageFilter()将隔离矢量图像中的单个通道并返回标量图像。在另一方面， ExtractImageFilter()并且CropImageFilter()将提取并返回一个图像的子区域，使用ExtractionRegion尺寸和索引和 itk::Size分别的。
但是，请注意只有 ExtractImageFilter()折叠尺寸。图像切片操作符也可以用于相同的目的。
'''

import SimpleITK as sitk
import sys

if len ( sys.argv ) < 3:
    print( "Usage: " +sys.argv[0]+ " <input-1> <input-2>" )
    sys.exit ( 1 )

#  Two vector images of same pixel type and dimension expected
image_1 = sitk.ReadImage(sys.argv[1])
image_2 = sitk.ReadImage(sys.argv[2])

#  join two N-D vector images to from an (N+1)-D image
join = sitk.JoinSeriesImageFilter()
joined_image = join.Execute(image_1,image_2)

# extract first three channels of joined image( assuming RGB)
select = sitk.VectorIndexSelectionCastImageFilter()
channel1_image = select.Execute(joined_image,0,sitk.sitkUInt8)
channel2_image = select.Execute(joined_image,1,sitk.sitkUInt8)
channel3_image = select.Execute(joined_image,2,sitk.sitkUInt8)

#  recompose image (should be same as joined_image)
compose = sitk.ComposeImageFilter()
composed_image = compose.Execute(channel1_image,channel2_image,channel3_image)

#  select same subregion using ExtractImageFilter
extract = sitk.ExtractImageFilter()
extract.SetSize([300,300,0])
extract.SetIndex([100,100,0])
extracted_image = extract.Execute(composed_image)

# Select same subregion using CropImageFilter (NOTE: CropImageFilter cannot reduce dimensions
# unlike ExtractImageFilter, so cropped_image is a three dimensional image with depth of 1)

crop = sitk.CropImageFilter()
crop.SetLowerBoundaryCropSize([100,100,0])
crop.SetUpperBoundaryCropSize([composed_image.GetWidth()-400, composed_image.GetHeight()-400, 1])
cropped_image = crop.Execute(composed_image)