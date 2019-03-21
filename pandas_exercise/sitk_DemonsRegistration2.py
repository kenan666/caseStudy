'''
This example illustrates how to use the fast symmetric forces Demons algorithm. 
As the name implies, unlike the classical algorithm, the forces are symmetric.

The underlying assumption of the demons framework is that the intensities of homologous points are equal. 
The example uses histogram matching to make the two images similar prior to registration. 
This is relevant for registration of MR images where the assumption is not valid. 
For other imaging modalities where the assumption is valid, such as CT, this step is not necessary. 
Additionally, the command design pattern is used to monitor registration progress. 
The resulting deformation field is written to file.
'''

import SimpleITK as sitk
import os
import sys

def command_iteration(filter):
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),filter.GetMetric()))

if len ( sys.argv ) < 4:
    print( "Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile>".format(sys.argv[0]))
    sys.exit ( 1 )

fixed = sitk.ReadImage(sys.argv[1])
moving = sitk.ReadImage(sys.argv[2])

matcher = sitk.HistogramMatchingImageFilter()

if (fixed.GetPixelID() in (sitk.sitkUInt8,sitk.sitkUInt8)):
    matcher.SetNumberOfHistogramLevels(128)
else:
    matcher.SetNumberOfHistogramLevels(1024)

matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving = matcher.Execute(moving,fixed)

# The fast symmetric forces Demons Registration Filter
# Note there is a whole family of Demons Registration algorithms included in SimpleITK
demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
demons.SetNumberOfIterations(200)

# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(1.0)

demons.AddCommand(sitk.sitkIterationEvent,lambda:command_iteration(demons))

if len(sys.argv) > 4 :
    initialTransform = sitk.ReadTransform(sys.argv[3])
    sys.argv[-1] = sys.argv.pop()

    toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
    toDisplacementFilter.SetReferenceImage(fixed)

    displacementFiled = toDisplacementFilter.Execute(initialTransform)
    displacementFiled = demons.Execute(fixed,moving,displacementFiled)

else:
    displacementFiled = demons.Execute(fixed,moving)

print("-------")
print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
print(" RMS: {0}".format(demons.GetRMSChange()))

outTx = sitk.DisplacementFieldTransform(displacementFiled)

sitk.WriteTransform(outTx,sys.argv[3])

if (not "SITK_NOSHOW" in os.environ):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
    sitk.Show( cimg, "DeformableRegistration1 Composition" )
