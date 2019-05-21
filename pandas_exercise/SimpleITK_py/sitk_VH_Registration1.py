from __future__ import print_function
%matplotlib inline
import matplotlib.pyplot as plt

import SimpleITK as sitk
print(sitk.Version())
from myshow import myshow
# Download data to work on
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

OUTPUT_DIR = "Output"

fixed_rgb = sitk.ReadImage(fdata("vm_head_rgb.mha"))
fixed_rgb = fixed_rgb[735:1330,204:975,:]
fixed_rgb = sitk.BinShrink(fixed_rgb,[3,3,1])

moving = sitk.ReadImage(fdata("vm_head_mri.mha"))
myshow(moving)

# Segment blue ice
seeds = [[10,10,10]]
fixed_mask = sitk.VectorConfidenceConnected(fixed_rgb, seedList=seeds, initialNeighborhoodRadius=5, numberOfIterations=4, multiplier=8)

# Invert the segment and choose largest component
fixed_mask = sitk.RelabelComponent(sitk.ConnectedComponent(fixed_mask==0))==1

myshow(sitk.Mask(fixed_rgb, fixed_mask));

# pick red channel
fixed = sitk.VectorIndexSelectionCast(fixed_rgb,0)

fixed = sitk.Cast(fixed,sitk.sitkFloat32)
moving = sitk.Cast(moving,sitk.sitkFloat32)

initialTransform = sitk.Euler3DTransform()
initialTransform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_mask,moving.GetPixelID()), moving, initialTransform, sitk.CenteredTransformInitializerFilter.MOMENTS)
print(initialTransform)

def command_iteration(method) :
    print("{0} = {1} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()),
              end='\n');
    sys.stdout.flush();

tx = initialTransform
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
R.SetOptimizerAsGradientDescentLineSearch(learningRate=1,numberOfIterations=100)
R.SetOptimizerScalesFromIndexShift()
R.SetShrinkFactorsPerLevel([4,2,1])
R.SetSmoothingSigmasPerLevel([8,4,2])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.1)
R.SetInitialTransform(tx)
R.SetInterpolator(sitk.sitkLinear)

import sys
R.RemoveAllCommands()
R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
outTx = R.Execute(sitk.Cast(fixed,sitk.sitkFloat32), sitk.Cast(moving,sitk.sitkFloat32))

print("-------")
print(tx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

tx.AddTransform(sitk.Transform(3,sitk.sitkAffine))

R.SetOptimizerAsGradientDescentLineSearch(learningRate=1,numberOfIterations=100)
R.SetOptimizerScalesFromIndexShift()
R.SetShrinkFactorsPerLevel([2,1])
R.SetSmoothingSigmasPerLevel([4,1])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
R.SetInitialTransform(tx)

outTx = R.Execute(sitk.Cast(fixed,sitk.sitkFloat32), sitk.Cast(moving,sitk.sitkFloat32))
R.GetOptimizerStopConditionDescription()

resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(fixed_rgb)
resample.SetInterpolator(sitk.sitkBSpline)
resample.SetTransform(outTx)
resample.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress: {0:03.1f}%...".format(100*resample.GetProgress()),end=''))
resample.AddCommand(sitk.sitkProgressEvent, lambda: sys.stdout.flush())
resample.AddCommand(sitk.sitkEndEvent, lambda: print("Done"))
out = resample.Execute(moving)

out_rgb = sitk.Cast( sitk.Compose( [sitk.RescaleIntensity(out)]*3), sitk.sitkVectorUInt8)
vis_xy = sitk.CheckerBoard(fixed_rgb, out_rgb, checkerPattern=[8,8,1])
vis_xz = sitk.CheckerBoard(fixed_rgb, out_rgb, checkerPattern=[8,1,8])
vis_xz = sitk.PermuteAxes(vis_xz, [0,2,1])

myshow(vis_xz,dpi=30)

import os

sitk.WriteImage(out, os.path.join(OUTPUT_DIR, "example_registration.mha"))
sitk.WriteImage(vis_xy, os.path.join(OUTPUT_DIR, "example_registration_xy.mha"))
sitk.WriteImage(vis_xz, os.path.join(OUTPUT_DIR, "example_registration_xz.mha"))