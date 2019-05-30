# Introduction to SimpleITKv4 Registration - Continued
import SimpleITK as sitk

# If the environment variable SIMPLE_ITK_MEMORY_CONSTRAINED_ENVIRONMENT is set, this will override the ReadImage
# function so that it also resamples the image to a smaller size (testing environment is memory constrained).
%run setup_for_testing

# Utility method that either downloads data from the network or
# if already downloaded returns the file name for reading from disk (cached data).
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

# Always write output to a separate directory, we don't want to pollute the source directory. 
import os
OUTPUT_DIR = 'Output'

# GUI components (sliders, dropdown...).
from ipywidgets import interact, fixed
     
# Enable display of HTML.
from IPython.display import display, HTML 
    
# Plots will be inlined.
%matplotlib inline

# Callbacks for plotting registration progress.
import registration_callbacks

# Utility functions
def save_transform_and_image(transform, fixed_image, moving_image, outputfile_prefix):
    """
    Write the given transformation to file, resample the moving_image onto the fixed_images grid and save the
    result to file.
    
    Args:
        transform (SimpleITK Transform): transform that maps points from the fixed image coordinate system to the moving.
        fixed_image (SimpleITK Image): resample onto the spatial grid defined by this image.
        moving_image (SimpleITK Image): resample this image.
        outputfile_prefix (string): transform is written to outputfile_prefix.tfm and resampled image is written to 
                                    outputfile_prefix.mha.
    """                             
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.     
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(transform)
    sitk.WriteImage(resample.Execute(moving_image), outputfile_prefix+'.mha')
    sitk.WriteTransform(transform, outputfile_prefix+'.tfm')
    
def DICOM_series_dropdown_callback(fixed_image, moving_image, series_dictionary):
    """
    Callback from dropbox which selects the two series which will be used for registration.
    The callback prints out some information about each of the series from the meta-data dictionary.
    For a list of all meta-dictionary tags and their human readable names see DICOM standard part 6, 
    Data Dictionary (http://medical.nema.org/medical/dicom/current/output/pdf/part06.pdf)
    """
    # The callback will update these global variables with the user selection.
    global selected_series_fixed
    global selected_series_moving
    
    img_fixed = sitk.ReadImage(series_dictionary[fixed_image][0])
    img_moving = sitk.ReadImage(series_dictionary[moving_image][0])
    
    # There are many interesting tags in the DICOM data dictionary, display a selected few.
    tags_to_print = {'0010|0010': 'Patient name: ', 
                     '0008|0060' : 'Modality: ',
                     '0008|0021' : 'Series date: ',
                     '0008|0031' : 'Series time:',
                     '0008|0070' : 'Manufacturer: '}
    html_table = []
    html_table.append('<table><tr><td><b>Tag</b></td><td><b>Fixed Image</b></td><td><b>Moving Image</b></td></tr>')
    for tag in tags_to_print:
        fixed_tag = ''
        moving_tag = ''
        try:            
            fixed_tag = img_fixed.GetMetaData(tag)
        except: # ignore if the tag isn't in the dictionary
            pass
        try:            
            moving_tag = img_moving.GetMetaData(tag)
        except: # ignore if the tag isn't in the dictionary
            pass           
        html_table.append('<tr><td>' + tags_to_print[tag] + 
                          '</td><td>' + fixed_tag + 
                          '</td><td>' + moving_tag + '</td></tr>')
    html_table.append('</table>')
    display(HTML(''.join(html_table)))
    selected_series_fixed = fixed_image
    selected_series_moving = moving_image

#  load data
data_directory = os.path.dirname(fdata("CIRS057A_MR_CT_DICOM/readme.txt"))

# 'selected_series_moving/fixed' will be updated by the interact function.
selected_series_fixed = ''
selected_series_moving = ''

# Directory contains multiple DICOM studies/series, store the file names
# in dictionary with the key being the series ID.
reader = sitk.ImageSeriesReader()
series_file_names = {}
series_IDs = list(reader.GetGDCMSeriesIDs(data_directory)) #list of all series
            
if series_IDs: #check that we have at least one series
    for series in series_IDs:
        series_file_names[series] = reader.GetGDCMSeriesFileNames(data_directory, series)    
    interact(DICOM_series_dropdown_callback, fixed_image=series_IDs, moving_image =series_IDs, series_dictionary=fixed(series_file_names)); 
else:
    print('This is surprising, data directory does not contain any DICOM series.')

# Actually read the data based on the user's selection.
fixed_image = sitk.ReadImage(series_file_names[selected_series_fixed])
moving_image = sitk.ReadImage(series_file_names[selected_series_moving])

# Save images to file and view overlap using external viewer.
sitk.WriteImage(fixed_image, os.path.join(OUTPUT_DIR, "fixedImage.mha"))
sitk.WriteImage(moving_image, os.path.join(OUTPUT_DIR, "preAlignment.mha"))

# initial 
initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

# Save moving image after initial transform and view overlap using external viewer.
save_transform_and_image(initial_transform, fixed_image, moving_image, os.path.join(OUTPUT_DIR, "initialAlignment"))

print(initial_transform)

# final registration
# version1 
registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
# Scale the step size differently for each parameter, this is critical!!!
registration_method.SetOptimizerScalesFromPhysicalShift() 

registration_method.SetInitialTransform(initial_transform, inPlace=False)

registration_method.AddCommand(sitk.sitkStartEvent, registration_callbacks.metric_start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, registration_callbacks.metric_end_plot)
registration_method.AddCommand(sitk.sitkIterationEvent, 
                               lambda: registration_callbacks.metric_plot_values(registration_method))

final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                 sitk.Cast(moving_image, sitk.sitkFloat32))

print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

# Save moving image after registration and view overlap using external viewer.
save_transform_and_image(final_transform_v1, fixed_image, moving_image, os.path.join(OUTPUT_DIR, "finalAlignment-v1"))

print(final_transform_v1)

#  version1.1
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
registration_method.SetInterpolator(sitk.sitkLinear)
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Set the initial moving and optimized transforms.
optimized_transform = sitk.Euler3DTransform()    
registration_method.SetMovingInitialTransform(initial_transform)
registration_method.SetInitialTransform(optimized_transform)

registration_method.AddCommand(sitk.sitkStartEvent, registration_callbacks.metric_start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, registration_callbacks.metric_end_plot)
registration_method.AddCommand(sitk.sitkIterationEvent, 
                               lambda: registration_callbacks.metric_plot_values(registration_method))

registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                            sitk.Cast(moving_image, sitk.sitkFloat32))

# Need to compose the transformations after registration.
final_transform_v11 = sitk.Transform(optimized_transform)
final_transform_v11.AddTransform(initial_transform)

print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

# Save moving image after registration and view overlap using external viewer.
save_transform_and_image(final_transform_v11, fixed_image, moving_image, os.path.join(OUTPUT_DIR, "finalAlignment-v1.1"))

print(final_transform_v11)

#  version2
registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)
   
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100) #, estimateLearningRate=registration_method.EachIteration)
registration_method.SetOptimizerScalesFromPhysicalShift() 

final_transform = sitk.Euler3DTransform(initial_transform)
registration_method.SetInitialTransform(final_transform)
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

registration_method.AddCommand(sitk.sitkStartEvent, registration_callbacks.metric_start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, registration_callbacks.metric_end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, 
                               registration_callbacks.metric_update_multires_iterations) 
registration_method.AddCommand(sitk.sitkIterationEvent, 
                               lambda: registration_callbacks.metric_plot_values(registration_method))

registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                            sitk.Cast(moving_image, sitk.sitkFloat32))

print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

# Save moving image after registration and view overlap using external viewer.
save_transform_and_image(final_transform, fixed_image, moving_image, os.path.join(OUTPUT_DIR, 'finalAlignment-v2'))

print(final_transform)