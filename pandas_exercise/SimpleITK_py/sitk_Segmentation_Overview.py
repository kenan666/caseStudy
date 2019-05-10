from __future__ import print_function

%matplotlib inline
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

import SimpleITK as sitk

# Download data to work on
%run update_path_to_download_script
from downloaddata import fetch_data as fdata
from myshow import myshow, myshow3d

img_T1 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd"))
img_T2 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT2.nrrd"))

# To visualize the labels image in RGB with needs a image with 0-255 range
img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)
img_T2_255 = sitk.Cast(sitk.RescaleIntensity(img_T2), sitk.sitkUInt8)

myshow3d(img_T1)

#  Thresholding

seg = img_T1>200
myshow(sitk.LabelOverlay(img_T1_255, seg), "Basic Thresholding")
seg = sitk.BinaryThreshold(img_T1, lowerThreshold=100, upperThreshold=400, insideValue=1, outsideValue=0)
myshow(sitk.LabelOverlay(img_T1_255, seg), "Binary Thresholding")

otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)
otsu_filter.SetOutsideValue(1)
seg = otsu_filter.Execute(img_T1)
myshow(sitk.LabelOverlay(img_T1_255, seg), "Otsu Thresholding")

print(otsu_filter.GetThreshold() )

#  Region Growing Segmentation
seed = (132,142,96)
seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
seg.CopyInformation(img_T1)
seg[seed] = 1
seg = sitk.BinaryDilate(seg, 3)
myshow(sitk.LabelOverlay(img_T1_255, seg), "Initial Seed")

seg = sitk.ConnectedThreshold(img_T1, seedList=[seed], lower=100, upper=190)

myshow(sitk.LabelOverlay(img_T1_255, seg), "Connected Threshold")

#  Improving upon this is the ConfidenceConnected filter, which uses the initial seed or current segmentation to estimate the threshold range.
seg = sitk.ConfidenceConnected(img_T1, seedList=[seed],
                                   numberOfIterations=1,
                                   multiplier=2.5,
                                   initialNeighborhoodRadius=1,
                                   replaceValue=1)

myshow(sitk.LabelOverlay(img_T1_255, seg), "ConfidenceConnected")

img_multi = sitk.Compose(img_T1, img_T2)
seg = sitk.VectorConfidenceConnected(img_multi, seedList=[seed],
                                             numberOfIterations=1,
                                             multiplier=2.5,
                                             initialNeighborhoodRadius=1)
myshow(sitk.LabelOverlay(img_T2_255, seg))

#  Fast Marching Segmentation
seed = (132,142,96)
feature_img = sitk.GradientMagnitudeRecursiveGaussian(img_T1, sigma=.5)
speed_img = sitk.BoundedReciprocal(feature_img) # This is parameter free unlike the Sigmoid
myshow(speed_img)

fm_filter = sitk.FastMarchingBaseImageFilter()
fm_filter.SetTrialPoints([seed])
fm_filter.SetStoppingValue(1000)
fm_img = fm_filter.Execute(speed_img)
myshow(sitk.Threshold(fm_img,
                    lower=0.0,
                    upper=fm_filter.GetStoppingValue(),
                    outsideValue=fm_filter.GetStoppingValue()+1))

def fm_callback(img, time, z):
    seg = img<time;
    myshow(sitk.LabelOverlay(img_T1_255[:,:,z], seg[:,:,z]))
           
interact( lambda **kwargs: fm_callback(fm_img, **kwargs),
            time=FloatSlider(min=0.05, max=1000.0, step=0.05, value=100.0),
            z=(0,fm_img.GetSize()[2]-1))

#  Level-Set Segmentation
seed = (132,142,96)

seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
seg.CopyInformation(img_T1)
seg[seed] = 1
seg = sitk.BinaryDilate(seg, 3)

# Use the seed to estimate a reasonable threshold range.
stats = sitk.LabelStatisticsImageFilter()
stats.Execute(img_T1, seg)

factor = 3.5
lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)
print(lower_threshold,upper_threshold)

init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)

lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
lsFilter.SetLowerThreshold(lower_threshold)
lsFilter.SetUpperThreshold(upper_threshold)
lsFilter.SetMaximumRMSError(0.02)
lsFilter.SetNumberOfIterations(1000)
lsFilter.SetCurvatureScaling(.5)
lsFilter.SetPropagationScaling(1)
lsFilter.ReverseExpansionDirectionOn()
ls = lsFilter.Execute(init_ls, sitk.Cast(img_T1, sitk.sitkFloat32))
print(lsFilter)

myshow(sitk.LabelOverlay(img_T1_255, ls>0))