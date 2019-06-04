# non_regid_registration
from __future__ import print_function

import SimpleITK as sitk
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

# loading data
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

label_shape_statistics_filter = sitk.LabelShapeStatisticsImageFilter()

for i, mask in enumerate(masks):
    label_shape_statistics_filter.Execute(mask)
    print('Lung volume in image {0} is {1} liters.'.format(i,0.000001*label_shape_statistics_filter.GetPhysicalSize(lung_label)))


# free from deformation
def bspline_intra_modal_registration(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None):
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 
    grid_physical_spacing = [50.0, 50.0, 50.0] # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(initial_transform)
        
    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
    
    return registration_method.Execute(fixed_image, moving_image)

# Perform Registration
# Select the fixed and moving images, valid entries are in [0,9].
fixed_image_index = 0
moving_image_index = 7


tx = bspline_intra_modal_registration(fixed_image = images[fixed_image_index], 
                                      moving_image = images[moving_image_index],
                                      fixed_image_mask = (masks[fixed_image_index] == lung_label),
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

# Transfer the segmentation via the estimated transformation. Use Nearest Neighbor interpolation to retain the labels.
transformed_labels = sitk.Resample(masks[moving_image_index],
                                   images[fixed_image_index],
                                   tx, 
                                   sitk.sitkNearestNeighbor,
                                   0.0, 
                                   masks[moving_image_index].GetPixelID())

segmentations_before_and_after = [masks[moving_image_index], transformed_labels]
interact(display_coronal_with_label_maps_overlay, coronal_slice = (0, images[0].GetSize()[1]-1),
         mask_index=(0,len(segmentations_before_and_after)-1),
         image = fixed(images[fixed_image_index]), masks = fixed(segmentations_before_and_after), 
         label=fixed(lung_label), window_min = fixed(-1024), window_max=fixed(976));

# Compute the Dice coefficient and Hausdorff distance between the segmentations before, and after registration.
ground_truth = masks[fixed_image_index] == lung_label
before_registration = masks[moving_image_index] == lung_label
after_registration = transformed_labels == lung_label

label_overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
label_overlap_measures_filter.Execute(ground_truth, before_registration)
print("Dice coefficient before registration: {:.2f}".format(label_overlap_measures_filter.GetDiceCoefficient()))
label_overlap_measures_filter.Execute(ground_truth, after_registration)
print("Dice coefficient after registration: {:.2f}".format(label_overlap_measures_filter.GetDiceCoefficient()))

hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
hausdorff_distance_image_filter.Execute(ground_truth, before_registration)
print("Hausdorff distance before registration: {:.2f}".format(hausdorff_distance_image_filter.GetHausdorffDistance()))
hausdorff_distance_image_filter.Execute(ground_truth, after_registration)
print("Hausdorff distance after registration: {:.2f}".format(hausdorff_distance_image_filter.GetHausdorffDistance()))

# Multi-resolution control point grid
def bspline_intra_modal_registration2(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None):
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Determine the number of BSpline control points using the physical spacing we 
    # want for the finest resolution control grid. 
    grid_physical_spacing = [50.0, 50.0, 50.0] # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    # The starting mesh size will be 1/4 of the original, it will be refined by 
    # the multi-resolution framework.
    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]
    
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=True,
                                                     scaleFactors=[1,2,4])
    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    # Use the LBFGS2 instead of LBFGS. The latter cannot adapt to the changing control grid resolution.
    registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-2, numberOfIterations=100, deltaConvergenceTolerance=0.01)

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
    
    return registration_method.Execute(fixed_image, moving_image)

# Select the fixed and moving images, valid entries are in [0,9].
fixed_image_index = 0
moving_image_index = 7


tx = bspline_intra_modal_registration2(fixed_image = images[fixed_image_index], 
                                      moving_image = images[moving_image_index],
                                      fixed_image_mask = (masks[fixed_image_index] == lung_label),
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