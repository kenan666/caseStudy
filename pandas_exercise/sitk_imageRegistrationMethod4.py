from __future__ import print_function

import SimpleITK as sitk
import sys,os

if len ( sys.argv ) < 4:
    print( "Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile> <numberOfBins> <samplingPercentage>".format(sys.argv[0]))
    sys.exit ( 1 )

def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))

fixed = sitk.ReadImage(sys.argv[1],sitk.sitkFloat32)
moving = sitk.ReadImage(sys.argv[2],sitk.sitkFloat32)

numberOfBins = 24
samplingPercentage = 0.10

if (len(sys.argv)) > 4 :
    numberOfBins = int(sys.argv[4])
if (len(sys.argv)) > 5:
    samplingPercentage = float(sys.argv[5])

R = sitk.ImageRegistrationMethod()
R.