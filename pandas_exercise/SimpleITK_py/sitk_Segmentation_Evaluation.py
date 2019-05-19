#  Segmentation Evaluation
import SimpleITK as sitk

import numpy as np

%run update_path_to_download_script
from downloaddata import fetch_data as fdata
%matplotlib inline
import matplotlib.pyplot as plt
import gui

from ipywidgets import interact, fixed

#  Utility method for display
def display_with_overlay(segmentation_number, slice_number, image, segs, window_min, window_max):
    """
    Display a CT slice with segmented contours overlaid onto it. The contours are the edges of 
    the labeled regions.
    """
    img = image[:,:,slice_number]
    msk = segs[segmentation_number][:,:,slice_number]
    overlay_img = sitk.LabelMapContourOverlay(sitk.Cast(msk, sitk.sitkLabelUInt8), 
                                              sitk.Cast(sitk.IntensityWindowing(img,
                                                                                windowMinimum=window_min, 
                                                                                windowMaximum=window_max), 
                                                        sitk.sitkUInt8), 
                                             opacity = 1, 
                                             contourThickness=[2,2])
    #We assume the original slice is isotropic, otherwise the display would be distorted 
    plt.imshow(sitk.GetArrayViewFromImage(overlay_img))
    plt.axis('off')
    plt.show()

##  Fetch the data
# Retrieve a single CT scan and three manual delineations of a liver tumor. Visual inspection of the data highlights the variability between experts.
image = sitk.ReadImage(fdata("liverTumorSegmentations/Patient01Homo.mha"))
segmentation_file_names = ["liverTumorSegmentations/Patient01Homo_Rad01.mha", 
                          "liverTumorSegmentations/Patient01Homo_Rad02.mha",
                          "liverTumorSegmentations/Patient01Homo_Rad03.mha"]
                          
segmentations = [sitk.ReadImage(fdata(file_name), sitk.sitkUInt8) for file_name in segmentation_file_names]
    
interact(display_with_overlay, segmentation_number=(0,len(segmentations)-1), 
         slice_number = (0, image.GetSize()[2]-1), image = fixed(image),
         segs = fixed(segmentations), window_min = fixed(-1024), window_max=fixed(976));

#  Derive a reference
# Use majority voting to obtain the reference segmentation. Note that this filter does not resolve ties. In case of 
# ties, it will assign max_label_value+1 or a user specified label value (labelForUndecidedPixels) to the result. 
# Before using the results of this filter you will have to check whether there were ties and modify the results to
# resolve the ties in a manner that makes sense for your task. The filter implicitly accommodates multiple labels.
labelForUndecidedPixels = 10
reference_segmentation_majority_vote = sitk.LabelVoting(segmentations, labelForUndecidedPixels)    

manual_plus_majority_vote = list(segmentations)  
# Append the reference segmentation to the list of manual segmentations
manual_plus_majority_vote.append(reference_segmentation_majority_vote)

interact(display_with_overlay, segmentation_number=(0,len(manual_plus_majority_vote)-1), 
         slice_number = (0, image.GetSize()[1]-1), image = fixed(image),
         segs = fixed(manual_plus_majority_vote), window_min = fixed(-1024), window_max=fixed(976));

# Use the STAPLE algorithm to obtain the reference segmentation. This implementation of the original algorithm
# combines a single label from multiple segmentations, the label is user specified. The result of the
# filter is the voxel's probability of belonging to the foreground. We then have to threshold the result to obtain
# a reference binary segmentation.
foregroundValue = 1
threshold = 0.95
reference_segmentation_STAPLE_probabilities = sitk.STAPLE(segmentations, foregroundValue) 
# We use the overloaded operator to perform thresholding, another option is to use the BinaryThreshold function.
reference_segmentation_STAPLE = reference_segmentation_STAPLE_probabilities > threshold

manual_plus_staple = list(segmentations)  
# Append the reference segmentation to the list of manual segmentations
manual_plus_staple.append(reference_segmentation_STAPLE)

interact(display_with_overlay, segmentation_number=(0,len(manual_plus_staple)-1), 
         slice_number = (0, image.GetSize()[1]-1), image = fixed(image),
         segs = fixed(manual_plus_staple), window_min = fixed(-1024), window_max=fixed(976));

# Evaluate segmentations using the reference
from enum import Enum

# Use enumerations to represent the various evaluation measures
class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)

class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)
    
# Select which reference we want to use (majority vote or STAPLE)    
reference_segmentation = reference_segmentation_STAPLE

# Empty numpy arrays to hold the results 
overlap_results = np.zeros((len(segmentations),len(OverlapMeasures.__members__.items())))  
surface_distance_results = np.zeros((len(segmentations),len(SurfaceDistanceMeasures.__members__.items())))  

# Compute the evaluation criteria

# Note that for the overlap measures filter, because we are dealing with a single label we 
# use the combined, all labels, evaluation measures without passing a specific label to the methods.
overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

# Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside 
# relationship, is irrelevant)
label = 1
reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
reference_surface = sitk.LabelContour(reference_segmentation)

statistics_image_filter = sitk.StatisticsImageFilter()
# Get the number of pixels in the reference surface by counting all pixels that are 1.
statistics_image_filter.Execute(reference_surface)
num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 

for i, seg in enumerate(segmentations):
    # Overlap measures
    overlap_measures_filter.Execute(reference_segmentation, seg)
    overlap_results[i,OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
    overlap_results[i,OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    overlap_results[i,OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
    overlap_results[i,OverlapMeasures.false_negative.value] = overlap_measures_filter.GetFalseNegativeError()
    overlap_results[i,OverlapMeasures.false_positive.value] = overlap_measures_filter.GetFalsePositiveError()
    # Hausdorff distance
    hausdorff_distance_filter.Execute(reference_segmentation, seg)
    
    surface_distance_results[i,SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(seg)
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In 
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two 
    # segmentations, though in our case it is. More on this below.
    surface_distance_results[i,SurfaceDistanceMeasures.mean_surface_distance.value] = np.mean(all_surface_distances)
    surface_distance_results[i,SurfaceDistanceMeasures.median_surface_distance.value] = np.median(all_surface_distances)
    surface_distance_results[i,SurfaceDistanceMeasures.std_surface_distance.value] = np.std(all_surface_distances)
    surface_distance_results[i,SurfaceDistanceMeasures.max_surface_distance.value] = np.max(all_surface_distances)
    
# Print the matrices
np.set_printoptions(precision=3)
print(overlap_results)
print(surface_distance_results)

# Improved output
import pandas as pd
from IPython.display import display, HTML 

# Graft our results matrix into pandas data frames 
overlap_results_df = pd.DataFrame(data=overlap_results, index = list(range(len(segmentations))), 
                                  columns=[name for name, _ in OverlapMeasures.__members__.items()]) 
surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index = list(range(len(segmentations))), 
                                  columns=[name for name, _ in SurfaceDistanceMeasures.__members__.items()]) 

# Display the data as HTML tables and graphs
display(HTML(overlap_results_df.to_html(float_format=lambda x: '%.3f' % x)))
display(HTML(surface_distance_results_df.to_html(float_format=lambda x: '%.3f' % x)))
overlap_results_df.plot(kind='bar').legend(bbox_to_anchor=(1.6,0.9))
surface_distance_results_df.plot(kind='bar').legend(bbox_to_anchor=(1.6,0.9))

# The formatting of the table using the default settings is less than ideal 
print(overlap_results_df.to_latex())

# We can improve on this by specifying the table's column format and the float format
print(overlap_results_df.to_latex(column_format='ccccccc', float_format=lambda x: '%.3f' % x))

### Segmentation Representation and the Hausdorff Distance
# Create our segmentations and display
image_size = [64,64]
circle_center = [30,30]
circle_radius = [20,20]

# A filled circle with radius R
seg = sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center) > 200
# A torus with inner radius r
reference_segmentation1 = seg - (sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center) > 240)
# A torus with inner radius r_2<r
reference_segmentation2 = seg - (sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center) > 250)

gui.multi_image_display2D([reference_segmentation1, reference_segmentation2, seg], 
                      ['reference 1', 'reference 2', 'segmentation'], figure_size=(12,4));

def surface_hausdorff_distance(reference_segmentation, seg):
    '''
    Compute symmetric surface distances and take the maximum.
    '''
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(reference_segmentation)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 

    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(seg)
    
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
    
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    all_surface_distances = seg2ref_distances + ref2seg_distances
    return np.max(all_surface_distances)

hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

# Use reference1, larger inner annulus radius, the surface based computation
# has a smaller difference. 
hausdorff_distance_filter.Execute(reference_segmentation1, seg)
print('HausdorffDistanceImageFilter result (reference1-segmentation): ' + 
      str(hausdorff_distance_filter.GetHausdorffDistance()))
print('Surface Hausdorff result (reference1-segmentation): ' + 
      str(surface_hausdorff_distance(reference_segmentation1,seg)))

# Use reference2, smaller inner annulus radius, the surface based computation
# has a larger difference.
hausdorff_distance_filter.Execute(reference_segmentation2, seg)
print('HausdorffDistanceImageFilter result (reference2-segmentation): ' + 
      str(hausdorff_distance_filter.GetHausdorffDistance()))
print('Surface Hausdorff result (reference2-segmentation): ' + 
      str(surface_hausdorff_distance(reference_segmentation2,seg)))