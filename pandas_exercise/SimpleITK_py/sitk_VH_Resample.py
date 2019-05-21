#  Resampling an Image onto Another's Physical Space
from __future__ import print_function
%matplotlib inline
import matplotlib.pyplot as plt

import SimpleITK as sitk

# If the environment variable SIMPLE_ITK_MEMORY_CONSTRAINED_ENVIRONMENT is set, this will override the ReadImage
# function so that it also resamples the image to a smaller size (testing environment is memory constrained).
%run setup_for_testing

print(sitk.Version())
from myshow import myshow
# Download data to work on
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

OUTPUT_DIR = "Output"

fixed = sitk.ReadImage(fdata("vm_head_rgb.mha"))

moving = sitk.ReadImage(fdata("vm_head_mri.mha"))

print(fixed.GetSize())
print(fixed.GetOrigin())
print(fixed.GetSpacing())
print(fixed.GetDirection())

print(moving.GetSize())
print(moving.GetOrigin())
print(moving.GetSpacing())
print(moving.GetDirection())

import sys
resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(fixed)
resample.SetInterpolator(sitk.sitkBSpline)
resample.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress: {0:03.1f}%...".format(100*resample.GetProgress()),end=''))
resample.AddCommand(sitk.sitkProgressEvent, lambda: sys.stdout.flush())
out = resample.Execute(moving)

myshow(out)

#combine the two images using a checkerboard pattern:
  #because the moving image is single channel with a high dynamic range we rescale it to [0,255] and repeat 
  #the channel 3 times
vis = sitk.CheckerBoard(fixed,sitk.Compose([sitk.Cast(sitk.RescaleIntensity(out),sitk.sitkUInt8)]*3), checkerPattern=[15,10,1])

myshow(vis)

import os

sitk.WriteImage(vis, os.path.join(OUTPUT_DIR, "example_resample_vis.mha"))

temp = sitk.Shrink(vis,[3,3,2])
sitk.WriteImage(temp, [os.path.join(OUTPUT_DIR,"r{0:03d}.jpg".format(i)) for i in range(temp.GetSize()[2])])
