# Segmentation_Thresholding_and_Edge_Detection

import SimpleITK as sitk

%run update_path_to_download_script
from downloaddata import fetch_data as fdata

%matplotlib notebook
import gui
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg

from ipywidgets import interact, fixed

spherical_fiducials_image = sitk.ReadImage(fdata("spherical_fiducials.mha"))
roi_list = [((280,320), (65,90), (8, 30)),
            ((200,240), (65,100), (15, 40))]

roi_acquisition_interface = gui.ROIDataAquisition(spherical_fiducials_image)
roi_acquisition_interface.set_rois(roi_list)

specified_rois = roi_acquisition_interface.get_rois()
# select the one ROI we will work on
ROI_INDEX = 0

roi = specified_rois[ROI_INDEX]
mask_value = 255

mask = sitk.Image(spherical_fiducials_image.GetSize(), sitk.sitkUInt8)
mask.CopyInformation(spherical_fiducials_image)
for x in range(roi[0][0], roi[0][1]+1):
    for y in range(roi[1][0], roi[1][1]+1):        
        for z in range(roi[2][0], roi[2][1]+1):
            mask[x,y,z] = mask_value

#  Thresholding based approach
intensity_values = sitk.GetArrayViewFromImage(spherical_fiducials_image)
roi_intensity_values = intensity_values[roi[2][0]:roi[2][1],
                                        roi[1][0]:roi[1][1],
                                        roi[0][0]:roi[0][1]].flatten()
plt.figure()
plt.hist(roi_intensity_values, bins=100)
plt.title("Intensity Values in ROI")
plt.show()

# Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
# set to outside_value. The sphere's have higher intensity values than the background, so they are outside.

inside_value = 0
outside_value = 255
number_of_histogram_bins = 100
mask_output = True

labeled_result = sitk.OtsuThreshold(spherical_fiducials_image, mask, inside_value, outside_value, 
                                   number_of_histogram_bins, mask_output, mask_value)

# Estimate the sphere radius from the segmented image using the LabelShapeStatisticsImageFilter.
label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
label_shape_analysis.SetBackgroundValue(inside_value)
label_shape_analysis.Execute(labeled_result)
print("The sphere's location is: {0:.2f}, {1:.2f}, {2:.2f}".format(*(label_shape_analysis.GetCentroid(outside_value))))
print("The sphere's radius is: {0:.2f}mm".format(label_shape_analysis.GetEquivalentSphericalRadius(outside_value)))

# Visually evaluate the results of segmentation, just to make sure. Use the zoom tool, second from the right, to 
# inspect the segmentation.
gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(sitk.Cast(sitk.IntensityWindowing(spherical_fiducials_image, windowMinimum=-32767, windowMaximum=-29611),
                                      sitk.sitkUInt8), labeled_result, opacity=0.5)],                   
                      title_list = ['thresholding result'])

#-----------------------------------------------------------
##  Edge detection based approach
# Create a cropped version of the original image.
sub_image = spherical_fiducials_image[roi[0][0]:roi[0][1],
                                      roi[1][0]:roi[1][1],
                                      roi[2][0]:roi[2][1]]

# Edge detection on the sub_image with appropriate thresholds and smoothing.
edges = sitk.CannyEdgeDetection(sitk.Cast(sub_image, sitk.sitkFloat32), lowerThreshold=0.0, 
                                upperThreshold=200.0, variance = (5.0,5.0,5.0))

#  Get the 3D location of the edge points and fit a sphere to them.
edge_indexes = np.where(sitk.GetArrayViewFromImage(edges) == 1.0)

# Note the reversed order of access between SimpleITK and numpy (z,y,x)
physical_points = [edges.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) \
                   for z,y,x in zip(edge_indexes[0], edge_indexes[1], edge_indexes[2])]

# Setup and solve linear equation system.
A = np.ones((len(physical_points),4))
b = np.zeros(len(physical_points))

for row, point in enumerate(physical_points):
    A[row,0:3] = -2*np.array(point)
    b[row] = -linalg.norm(point)**2

res,_,_,_ = linalg.lstsq(A,b)

print("The sphere's location is: {0:.2f}, {1:.2f}, {2:.2f}".format(*res[0:3]))
print("The sphere's radius is: {0:.2f}mm".format(np.sqrt(linalg.norm(res[0:3])**2 - res[3])))

# Visually evaluate the results of edge detection, just to make sure. Note that because SimpleITK is working in the
# physical world (not pixels, but mm) we can easily transfer the edges localized in the cropped image to the original.
# Use the zoom tool, second from the right, for close inspection of the edge locations.

edge_label = sitk.Image(spherical_fiducials_image.GetSize(), sitk.sitkUInt16)
edge_label.CopyInformation(spherical_fiducials_image)
e_label = 255
for point in physical_points:
    edge_label[edge_label.TransformPhysicalPointToIndex(point)] = e_label

gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(sitk.Cast(sitk.IntensityWindowing(spherical_fiducials_image, windowMinimum=-32767, windowMaximum=-29611),
                                                                sitk.sitkUInt8), edge_label, opacity=0.5)],                   
                      title_list = ['edge detection result'])

