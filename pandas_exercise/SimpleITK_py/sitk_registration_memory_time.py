# Registration: Memory-Time Trade-off
from __future__ import print_function

import SimpleITK as sitk

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

#utility method that either downloads data from the Girder repository or
#if already downloaded returns the file name for reading from disk (cached data)
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

import registration_utilities as ru

from ipywidgets import interact, fixed


def register_images(fixed_image, moving_image, initial_transform, interpolator):

    registration_method = sitk.ImageRegistrationMethod()    
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(interpolator)     
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000) 
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    return( final_transform, registration_method.GetOptimizerStopConditionDescription())

#  load data
fixed_image =  sitk.ReadImage(fdata("training_001_ct.mha"), sitk.sitkFloat32)
moving_image = sitk.ReadImage(fdata("training_001_mr_T1.mha"), sitk.sitkFloat32) 
fixed_fiducial_points, moving_fiducial_points = ru.load_RIRE_ground_truth(fdata("ct_T1.standard"))

R, t = ru.absolute_orientation_m(fixed_fiducial_points, moving_fiducial_points)
reference_transform = sitk.Euler3DTransform()
reference_transform.SetMatrix(R.flatten())
reference_transform.SetTranslation(t)

# Generate a reference dataset from the reference transformation (corresponding points in the fixed and moving images).
fixed_points = ru.generate_random_pointset(image=fixed_image, num_points=1000)
moving_points = [reference_transform.TransformPoint(p) for p in fixed_points]    

interact(lambda image1_z, image2_z, image1, image2:ru.display_scalar_images(image1_z, image2_z, image1, image2), 
         image1_z=(0,fixed_image.GetSize()[2]-1), 
         image2_z=(0,moving_image.GetSize()[2]-1), 
         image1 = fixed(fixed_image), 
         image2=fixed(moving_image));

# Invest time and memory in exchange for future time savings
# Isotropic voxels with 1mm spacing.
new_spacing = [1.0]*moving_image.GetDimension()

# Create resampled image using new spacing and size.
original_size = moving_image.GetSize()
original_spacing = moving_image.GetSpacing()
resampled_image_size = [int(spacing/new_s*size) 
                        for spacing, size, new_s in zip(original_spacing, original_size, new_spacing)]  
resampled_moving_image = sitk.Image(resampled_image_size, moving_image.GetPixelID())
resampled_moving_image.SetSpacing(new_spacing)
resampled_moving_image.SetOrigin(moving_image.GetOrigin())
resampled_moving_image.SetDirection(moving_image.GetDirection())

# Resample original image using identity transform and the BSpline interpolator.
resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(resampled_moving_image)                      
resample.SetInterpolator(sitk.sitkBSpline)  
resample.SetTransform(sitk.Transform())
resampled_moving_image = resample.Execute(moving_image)

print('Original image size and spacing: {0} {1}'.format(original_size, original_spacing)) 
print('Resampled image size and spacing: {0} {1}'.format(resampled_moving_image.GetSize(), 
                                                         resampled_moving_image.GetSpacing()))
print('Memory ratio: 1 : {0}'.format((np.array(resampled_image_size)/np.array(original_size).astype(float)).prod()))

# Registration
# 1 initial alignment
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

# 2 Original Resolution
# The arguments to the timeit magic specify that this cell should only be run once. 

# We define this variable as global so that it is accessible outside of the cell (timeit wraps the code in the cell
# making all variables local, unless explicitly declared global).
global original_resolution_errors

final_transform, optimizer_termination = register_images(fixed_image, moving_image, initial_transform, sitk.sitkLinear)
final_errors_mean, final_errors_std, _, final_errors_max, original_resolution_errors = ru.registration_errors(final_transform, fixed_points, moving_points)

print(optimizer_termination)
print('After registration, errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean, final_errors_std, final_errors_max))

# 3 higher resolution
# The arguments to the timeit magic specify that this cell should only be run once. 

# We define this variable as global so that it is accessible outside of the cell (timeit wraps the code in the cell
# making all variables local, unless explicitly declared global).
global resampled_resolution_errors 

final_transform, optimizer_termination = register_images(fixed_image, resampled_moving_image, initial_transform, sitk.sitkNearestNeighbor)
final_errors_mean, final_errors_std, _, final_errors_max, resampled_resolution_errors = ru.registration_errors(final_transform, fixed_points, moving_points)

print(optimizer_termination)
print('After registration, errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean, final_errors_std, final_errors_max))

# 4 Compare the error distributions
plt.hist(original_resolution_errors, bins=20, alpha=0.5, label='original resolution', color='blue')
plt.hist(resampled_resolution_errors, bins=20, alpha=0.5, label='higher resolution', color='green')
plt.legend()
plt.title('TRE histogram');

