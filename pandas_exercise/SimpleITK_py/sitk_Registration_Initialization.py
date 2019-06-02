#  Registration Initialization
import SimpleITK as sitk

# If the environment variable SIMPLE_ITK_MEMORY_CONSTRAINED_ENVIRONMENT is set, this will override the ReadImage
# function so that it also resamples the image to a smaller size (testing environment is memory constrained).
%run setup_for_testing

import os
import numpy as np

from ipywidgets import interact, fixed
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

%matplotlib notebook
import gui

# This is the registration configuration which we use in all cases. The only parameter that we vary 
# is the initial_transform. 
def multires_registration(fixed_image, moving_image, initial_transform):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return (final_transform, registration_method.GetMetricValue())

#  loading data
data_directory = os.path.dirname(fdata("CIRS057A_MR_CT_DICOM/readme.txt"))

fixed_series_ID = "1.2.840.113619.2.290.3.3233817346.783.1399004564.515"
moving_series_ID = "1.3.12.2.1107.5.2.18.41548.30000014030519285935000000933"

reader = sitk.ImageSeriesReader()
fixed_image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory, fixed_series_ID), sitk.sitkFloat32)
moving_image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory, moving_series_ID), sitk.sitkFloat32)

# To provide a reasonable display we need to window/level the images. By default we could have used the intensity
# ranges found in the images [SimpleITK's StatisticsImageFilter], but these are not the best values for viewing.
# Using an external viewer we identified the following settings.
ct_window_level = [1727,-320]
mr_window_level = [355,178]

gui.MultiImageDisplay(image_list = [fixed_image, moving_image],                   
                      title_list = ['fixed image', 'moving image'], figure_size=(8,4), window_level_list=[ct_window_level, mr_window_level]);

# Register using centered transform initializer (assumes orientation is similar)
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

final_transform,_ = multires_registration(fixed_image, moving_image, initial_transform)

gui.RegistrationPointDataAquisition(fixed_image, moving_image, figure_size=(8,4), 
                                    known_transformation=final_transform, 
                                    fixed_window_level=ct_window_level, moving_window_level=mr_window_level);

# Register using sampling of the parameter space
all_orientations = {'x=0, y=0, z=180': (0.0,0.0,np.pi),
                    'x=0, y=180, z=0': (0.0,np.pi, 0.0),
                    'x=0, y=180, z=180': (0.0,np.pi, np.pi)}    

# Registration framework setup.
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
registration_method.SetInterpolator(sitk.sitkLinear)

# Evaluate the similarity metric using the rotation parameter space sampling, translation remains the same for all.
initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed_image, 
                                                                            moving_image, 
                                                                            sitk.Euler3DTransform(), 
                                                                            sitk.CenteredTransformInitializerFilter.GEOMETRY))
registration_method.SetInitialTransform(initial_transform, inPlace=False)
best_orientation = (0.0,0.0,0.0)
best_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)

# Iterate over all other rotation parameter settings. 
for key, orientation in all_orientations.items():
    initial_transform.SetRotation(*orientation)
    registration_method.SetInitialTransform(initial_transform)
    current_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
    if current_similarity_value < best_similarity_value:
        best_similarity_value = current_similarity_value
        best_orientation = orientation
print('best orientation is: ' + str(best_orientation))

# 并行处理
from multiprocessing.pool import ThreadPool
from functools import partial

# This function evaluates the metric value in a thread safe manner
def evaluate_metric(current_rotation, tx, f_image, m_image):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    current_transform = sitk.Euler3DTransform(tx)
    current_transform.SetRotation(*current_rotation)
    registration_method.SetInitialTransform(current_transform)
    res = registration_method.MetricEvaluate(f_image, m_image)
    return res

p = ThreadPool(len(all_orientations)+1)
orientations_list = [(0,0,0)] + list(all_orientations.values())
all_metric_values = p.map(partial(evaluate_metric, 
                                  tx = initial_transform, 
                                  f_image = fixed_image,
                                  m_image = moving_image),
                          orientations_list)
best_orientation = orientations_list[np.argmin(all_metric_values)]
print('best orientation is: ' + str(best_orientation))

initial_transform.SetRotation(*best_orientation)
final_transform,_ = multires_registration(fixed_image, moving_image, initial_transform)

gui.RegistrationPointDataAquisition(fixed_image, moving_image, figure_size=(8,4), 
                                    known_transformation=final_transform, 
                                    fixed_window_level=ct_window_level, moving_window_level=mr_window_level);

#  Exhaustive optimizer
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
registration_method.SetInterpolator(sitk.sitkLinear)
# The order of parameters for the Euler3DTransform is [angle_x, angle_y, angle_z, t_x, t_y, t_z]. The parameter 
# sampling grid is centered on the initial_transform parameter values, that are all zero for the rotations. Given
# the number of steps and their length and optimizer scales we have:
# angle_x = 0
# angle_y = -pi, 0, pi
# angle_z = -pi, 0, pi
registration_method.SetOptimizerAsExhaustive(numberOfSteps=[0,1,1,0,0,0], stepLength = np.pi)
registration_method.SetOptimizerScales([1,1,1,1,1,1])

#Perform the registration in-place so that the initial_transform is modified.
registration_method.SetInitialTransform(initial_transform, inPlace=True)
registration_method.Execute(fixed_image, moving_image)

print('best initial transformation is: ' + str(initial_transform.GetParameters()))

final_transform, _ = multires_registration(fixed_image, moving_image, initial_transform)
gui.RegistrationPointDataAquisition(fixed_image, moving_image, figure_size=(8,4), 
                                    known_transformation=final_transform, 
                                    fixed_window_level=ct_window_level, moving_window_level=mr_window_level);

#  Exhaustive optimizer - an exploration-exploitation view
#
# Exploration step.
#
def start_observer():
    global metricvalue_parameters_list 
    metricvalue_parameters_list = []

def iteration_observer(registration_method):    
    metricvalue_parameters_list.append((registration_method.GetMetricValue(), registration_method.GetOptimizerPosition()))
    
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
registration_method.SetInterpolator(sitk.sitkLinear)
# The order of parameters for the Euler3DTransform is [angle_x, angle_y, angle_z, t_x, t_y, t_z]. The parameter 
# sampling grid is centered on the initial_transform parameter values, that are all zero for the rotations. Given
# the number of steps and their length and optimizer scales we have:
# angle_x = 0
# angle_y = -pi, 0, pi
# angle_z = -pi, 0, pi
registration_method.SetOptimizerAsExhaustive(numberOfSteps=[0,1,1,0,0,0], stepLength = np.pi)
registration_method.SetOptimizerScales([1,1,1,1,1,1])

#We don't really care if transformation is modified in place or not, we will select the k 
#best transformations from the parameters_metricvalue_list.
registration_method.SetInitialTransform(initial_transform, inPlace=True)

registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_observer(registration_method))
registration_method.AddCommand(sitk.sitkStartEvent, start_observer )
_ = registration_method.Execute(fixed_image, moving_image)

#
# Exploitation step.
#

#Sort our list from most to least promising solutions (low to high metric values). 
metricvalue_parameters_list.sort(key=lambda x: x[0]) 

# We exploit the k_most_promising parameter value settings.
k_most_promising = min(3, len(metricvalue_parameters_list))
final_results = []
for metricvalue, parameters in metricvalue_parameters_list[0:k_most_promising]:
    initial_transform.SetParameters(parameters)
    final_results.append(multires_registration(fixed_image, moving_image, initial_transform))

final_transform, _ = min(final_results, key=lambda x: x[1]) 
gui.RegistrationPointDataAquisition(fixed_image, moving_image, figure_size=(8,4), 
                                    known_transformation=final_transform, 
                                    fixed_window_level=ct_window_level, moving_window_level=mr_window_level);

# Register using manual initialization
point_acquisition_interface = gui.RegistrationPointDataAquisition(fixed_image, moving_image, figure_size=(8,4), 
                                                                  fixed_window_level=ct_window_level, 
                                                                  moving_window_level=mr_window_level);

# Get the manually specified points and compute the transformation.

fixed_image_points, moving_image_points = point_acquisition_interface.get_points()

# FOR TESTING: previously localized points
fixed_image_points = [(24.062587103074605, 14.594981536981521, -58.75), 
                      (6.178716135332678, 53.93949766601378, -58.75), 
                      (74.14383149714774, -69.04462737237648, -76.25), 
                      (109.74899278747029, -14.905272533666817, -76.25)]
moving_image_points = [(4.358707846364581, 60.46357110706131, -71.53120422363281), 
                       (24.09010295252645, 98.21840981673873, -71.53120422363281), 
                       (-52.11888008581127, -26.57984635768439, -58.53120422363281), 
                       (-87.46150681392184, 28.73904765153219, -58.53120422363281)]

fixed_image_points_flat = [c for p in fixed_image_points for c in p]        
moving_image_points_flat = [c for p in moving_image_points for c in p]
initial_transform = sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), 
                                                                fixed_image_points_flat, 
                                                                moving_image_points_flat)


print('manual initial transformation is: ' + str(initial_transform.GetParameters()))

final_transform = multires_registration(fixed_image, moving_image, initial_transform)
gui.RegistrationPointDataAquisition(fixed_image, moving_image, figure_size=(8,4), 
                                    known_transformation=final_transform, 
                                    fixed_window_level=ct_window_level, moving_window_level=mr_window_level);