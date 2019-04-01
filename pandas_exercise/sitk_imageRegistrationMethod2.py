from __future__ import print_function
from functools import reduce


import SimpleITK as sitk
import sys
import os

def command_iteration(method):
    print('{0:3} = {1:7.5f} : {2}'.format(method.GetOptimizerIteration(),
                                            method.GetMetricValue(),
                                            method.GetOptimizerPosition()))

if len ( sys.argv ) < 4:
    print( "Usage: {0} <fixedImageFilter> <movingImageFile>  <outputTransformFile>".format(sys.argv[0]))
    sys.exit ( 1 )

pixelType = sitk.sitkFloat32

fixed = sitk.ReadImage(sys.argv[1],sitk.sitkFloat32)
fixed = sitk.Normalize(fixed)
fixed = sitk.DiscreteGaussian(fixed,2.0)

moving = sitk.ReadImage(sys.argv[2],sitk.sitkFloat32)
moving = sitk.Normalize(moving)
moving = sitk.DiscreteGaussian(moving,2.0)

R = sitk.ImageRegistrationMethod()

R.SetMetricAsJointHistogramMutualInformation()

R.SetOptimizerAsGradientDescentLineSearch(learningRate = 1.0,
                                            numberOfIterations = 200,
                                            convergenceMMinimumValue = le - 5,
                                            convergenceWindowSize = 5)

R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent,lambda : command_iteration(R))

outTx = R.Execute(fixed,moving)

print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

sitk.WriteTransform(outTx,sys.argv[3])

if (not 'SITK_NOSHOW' in os.environ):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed);
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
    sitk.Show( cimg, "ImageRegistration2 Composition" )