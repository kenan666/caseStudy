'''
This example illustrates how to use the classic Demons registration algorithm. 
The user supplied parameters for the algorithm are the number of iterations and the standard deviations for the 
Gaussian smoothing of the total displacement field. Additional methods which control regularization, 
total field smoothing for elastic model or update field smoothing for viscous model are available.

The underlying assumption of the demons framework is that the intensities of homologous points are equal. 
The example uses histogram matching to make the two images similar prior to registration. 
This is relevant for registration of MR images where the assumption is not valid. 
For other imaging modalities where the assumption is valid, such as CT, this step is not necessary. 
Additionally, the command design pattern is used to monitor registration progress. 
The resulting deformation field is written to file.

'''
#---------------------------------
import SimpleITK as sitk
import os 
import sys

def command_iteration(filter):
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),filter.GetMetric()))

if len ( sys.argv ) < 4:
    print( "Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile>".format(sys.argv[0]))
    sys.exit ( 1 )

fixed = sitk.ReadImage(sys.argv[1],sitk.sitkFloat32)
moving = sitk.ReadImage(sys.argv[2],sitk.sitkFloat32)

matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving = matcher.Execute(moving,fixed)

# The basic Demons Registration Filter
# Note there is a whole family of Demons Registration algorithms included in SimpleITK
demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(50)

# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(1.0)

demons.AddCommand(sitk.sitkIterationEvent,lambda : command_iteration(demons))

displacementField = demons.Execute(fixed,moving)

print('-------------')
print('Number Of Iterations :{0}'.format(demons.GetElapsedIterations()))
print('RMS : {0}'.format(demons.GetRMSChange()))

outTx = sitk.DisplacementFieldTransform(displacementField)

sitk.WriteTransform(outTx,sys.argv[3])

if (not "SITK_NOSHOW" in os.environ):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed),sitk.sitkUInt8)  # cast 函数  sitk 在可以改变图像的数据类型
    simg2 = sitk.Cast(sitk.RescaleIntensity(out),sitk.sitkUInt8)

    # Use the // floor division operator so that the pixel type is
    # the same for all three images which is the expectation for
    # the compose filter.

    cimg = sitk.Compose(simg1,simg2,simg1//2. + simg2//2.)
    sitk.Show(cimg,"DeformableRegistration1 Composition")