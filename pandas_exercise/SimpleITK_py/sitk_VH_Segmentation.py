from __future__ import print_function
%matplotlib inline
import matplotlib.pyplot as plt

import SimpleITK as sitk
print(sitk.Version())
from myshow import myshow
# Download data to work on
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

img = sitk.ReadImage(fdata("a_vm1108.png"))
myshow(img)

seeds = [[400,500], [1500,800]]
seg = sitk.VectorConfidenceConnected(img, seedList=seeds, numberOfIterations=5, multiplier=7)

myshow(sitk.Mask(img, seg==0))

seg = sitk.BinaryClosingByReconstruction(seg, 100)

myshow(sitk.Mask(img, seg==0))