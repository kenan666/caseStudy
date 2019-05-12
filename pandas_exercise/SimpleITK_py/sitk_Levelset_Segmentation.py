%matplotlib inline
import matplotlib.pyplot as plt
import SimpleITK as sitk
from myshow import myshow, myshow3d
# Download data to work on
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

img_T1 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd"))
img_T2 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT2.nrrd"))
img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)
img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)

sitk.Show(img_T1,title = 'T1')

idx = (106,116,141)
pt = img_T1.TransformIndexToPhysicalPoint(idx)

seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
seg.CopyInformation(img_T1)
seg[idx] = 1
seg = sitk.BinaryDilate(seg, 3)
myshow3d(sitk.LabelOverlay(img_T1_255, seg), zslices=range(idx[2]-3, idx[2]+4, 3), dpi=30, title="Initial Seed")

stats = sitk.LabelStatisticsImageFilter()
stats.Execute(img_T1, seg)
print(stats)

factor = 1.5
lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)

init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)

lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
lsFilter.SetLowerThreshold(lower_threshold)
lsFilter.SetUpperThreshold(upper_threshold)
lsFilter.SetMaximumRMSError(0.02)
lsFilter.SetNumberOfIterations(100)
lsFilter.SetCurvatureScaling(1)
lsFilter.SetPropagationScaling(1)
lsFilter.ReverseExpansionDirectionOn()
ls = lsFilter.Execute(init_ls, sitk.Cast(img_T1, sitk.sitkFloat32))
print(lsFilter)

zslice_offset = 4
t = "LevelSet after "+str(lsFilter.GetNumberOfIterations())+" iterations"
myshow3d(sitk.LabelOverlay(img_T1_255, ls > 0), zslices=range(idx[2]-zslice_offset,idx[2]+zslice_offset+1,zslice_offset), dpi=20, title=t)

lsFilter.SetNumberOfIterations(25)
img_T1f = sitk.Cast(img_T1, sitk.sitkFloat32)
ls = init_ls
niter = 0
for i in range(0, 10):
    ls = lsFilter.Execute(ls, img_T1f)
    niter += lsFilter.GetNumberOfIterations()
    t = "LevelSet after "+str(niter)+" iterations and RMS "+str(lsFilter.GetRMSChange())
    fig = myshow3d(sitk.LabelOverlay(img_T1_255, ls > 0), zslices=range(idx[2]-zslice_offset,idx[2]+zslice_offset+1,zslice_offset), dpi=20, title=t)

