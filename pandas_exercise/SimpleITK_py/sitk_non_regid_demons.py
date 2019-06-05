#  non_regid_demons
from __future__ import print_function

import SimpleITK as sitk
import numpy as np

# If the environment variable SIMPLE_ITK_MEMORY_CONSTRAINED_ENVIRONMENT is set, this will override the ReadImage
# function so that it also resamples the image to a smaller size (testing environment is memory constrained).
%run setup_for_testing

import registration_utilities as ru
import registration_callbacks as rc

%matplotlib inline
import matplotlib.pyplot as plt

from ipywidgets import interact, fixed

#utility method that either downloads data from the Girder repository or
#if already downloaded returns the file name for reading from disk (cached data)
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

# utilities
%run popi_utilities_setup.py

#  loading data 
images = []
masks = []
points = []
for i in range(0,10):
    image_file_name = 'POPI/meta/{0}0-P.mhd'.format(i)
    mask_file_name = 'POPI/masks/{0}0-air-body-lungs.mhd'.format(i)
    points_file_name = 'POPI/landmarks/{0}0-Landmarks.pts'.format(i)
    images.append(sitk.ReadImage(fdata(image_file_name), sitk.sitkFloat32)) #read and cast to format required for registration
    masks.append(sitk.ReadImage(fdata(mask_file_name)))
    points.append(read_POPI_points(fdata(points_file_name)))
        
interact(display_coronal_with_overlay, temporal_slice=(0,len(images)-1), 
         coronal_slice = (0, images[0].GetSize()[1]-1), 
         images = fixed(images), masks = fixed(masks), 
         label=fixed(lung_label), window_min = fixed(-1024), window_max=fixed(976));


# Demons Registration
def demons_registration(fixed_image, moving_image, fixed_points = None, moving_points = None):
    
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))
    
    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0) 
    
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU
        
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])    

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)        
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
        
    return registration_method.Execute(fixed_image, moving_image)

# Uncomment the line above if you want to time the running of this cell.

# Select the fixed and moving images, valid entries are in [0,9]
fixed_image_index = 0
moving_image_index = 7


tx = demons_registration(fixed_image = images[fixed_image_index], 
                         moving_image = images[moving_image_index],
                         fixed_points = points[fixed_image_index], 
                         moving_points = points[moving_image_index]
                         )
initial_errors_mean, initial_errors_std, _, initial_errors_max, initial_errors = ru.registration_errors(sitk.Euler3DTransform(), points[fixed_image_index], points[moving_image_index])
final_errors_mean, final_errors_std, _, final_errors_max, final_errors = ru.registration_errors(tx, points[fixed_image_index], points[moving_image_index])

plt.hist(initial_errors, bins=20, alpha=0.5, label='before registration', color='blue')
plt.hist(final_errors, bins=20, alpha=0.5, label='after registration', color='green')
plt.legend()
plt.title('TRE histogram');
print('Initial alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(initial_errors_mean, initial_errors_std, initial_errors_max))
print('Final alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean, final_errors_std, final_errors_max))

'''
SimpleITK also includes a set of Demons filters which are independent of the ImageRegistrationMethod. These include:

DemonsRegistrationFilter
DiffeomorphicDemonsRegistrationFilter
FastSymmetricForcesDemonsRegistrationFilter
SymmetricForcesDemonsRegistrationFilter
'''

def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    """
    Args:
        image: The image we want to resample.
        shrink_factor(s): Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma(s): Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
    """
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors]*image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
    
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(shrink_factors,original_size)]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0, 
                         image.GetPixelID())


    
def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None, 
                      shrink_factors=None, smoothing_sigmas=None):
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))
    
    # Create initial displacement field at lowest resolution. 
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(), 
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])
 
    # Run the registration.            
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1], 
                                                                moving_images[-1], 
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.    
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
            initial_displacement_field = sitk.Resample (initial_displacement_field, f_image)
            initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)


# Define a simple callback which allows us to monitor the Demons filter's progress.
def iteration_callback(filter):
    print('\r{0}: {1:.2f}'.format(filter.GetElapsedIterations(), filter.GetMetric()), end='')

fixed_image_index = 0
moving_image_index = 7

# Select a Demons filter and configure it.
demons_filter =  sitk.FastSymmetricForcesDemonsRegistrationFilter()
demons_filter.SetNumberOfIterations(20)
# Regularization (update field - viscous, total field - elastic).
demons_filter.SetSmoothDisplacementField(True)
demons_filter.SetStandardDeviations(2.0)

# Add our simple callback to the registration filter.
demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

# Run the registration.
tx = multiscale_demons(registration_algorithm=demons_filter, 
                       fixed_image = images[fixed_image_index], 
                       moving_image = images[moving_image_index],
                       shrink_factors = [4,2],
                       smoothing_sigmas = [8,4])

# Compare the initial and final TREs.
initial_errors_mean, initial_errors_std, _, initial_errors_max, initial_errors = ru.registration_errors(sitk.Euler3DTransform(), points[fixed_image_index], points[moving_image_index])
final_errors_mean, final_errors_std, _, final_errors_max, final_errors = ru.registration_errors(tx, points[fixed_image_index], points[moving_image_index])

plt.hist(initial_errors, bins=20, alpha=0.5, label='before registration', color='blue')
plt.hist(final_errors, bins=20, alpha=0.5, label='after registration', color='green')
plt.legend()
plt.title('TRE histogram');
print('\nInitial alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(initial_errors_mean, initial_errors_std, initial_errors_max))
print('Final alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean, final_errors_std, final_errors_max))


# Transferring Segmentation
# load data
import glob
import pandas as pd
from gui import multi_image_display2D

# Fetch all of the data associated with this example.
data_directory = os.path.dirname(fdata("mr_slice_atlas/readme.txt"))
                                       
segmented_img = sitk.ReadImage(os.path.join(data_directory,'segmented_image.mha'))
new_img = sitk.ReadImage(os.path.join(data_directory,'new_image.mha'))

contours_list = []
for file_name in glob.glob(os.path.join(data_directory,'*.csv')):
    df = pd.read_csv(file_name)
    contours_list.append((list(df['X']), list(df['Y'])))

# Display the images and overlay the contours onto the segmented image.
fig,axes = multi_image_display2D([segmented_img, new_img])
for contour in contours_list:
    axes[0].plot(contour[0], contour[1], linewidth=5)

# Register and transfer the segmentation.
# Select a Demons filter and configure it.
demons_filter =  sitk.DiffeomorphicDemonsRegistrationFilter()
demons_filter.SetNumberOfIterations(20)
# Regularization (update field - viscous, total field - elastic).
demons_filter.SetSmoothDisplacementField(True)
demons_filter.SetStandardDeviations(0.8)

# create initial transform
initial_tfm = initial_transform = sitk.CenteredTransformInitializer(segmented_img, 
                                                                    new_img, 
                                                                    sitk.Euler2DTransform(), 
                                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
# Run the registration.
final_tfm = multiscale_demons(registration_algorithm=demons_filter, 
                              fixed_image = segmented_img,
                              moving_image = new_img,
                              initial_transform = initial_tfm,
                              shrink_factors = [6,4,2],
                              smoothing_sigmas = [6,4,2])

# Display the transformed segmentation.
fig,axes = multi_image_display2D([segmented_img, new_img])
for contour in contours_list:
    # Plot on segmented image.
    axes[0].plot(contour[0], contour[1], linewidth=5)
    # Transform the contour points from segmented image to new image (requires the use of points in physical space)
    transformed_contour = [new_img.TransformPhysicalPointToContinuousIndex(final_tfm.TransformPoint(segmented_img.TransformContinuousIndexToPhysicalPoint(p))) for p in zip(contour[0],contour[1])]
    x_coords, y_coords = zip(*transformed_contour)
    axes[1].plot(x_coords, y_coords, linewidth=5)