# To use interactive plots (mouse clicks, zooming, panning) we use the notebook back end. We want our graphs 
# to be embedded in the notebook, inline mode, this combination is defined by the magic "%matplotlib notebook".
%matplotlib notebook

import SimpleITK as sitk

%run update_path_to_download_script
from downloaddata import fetch_data as fdata
import gui

# Using an external viewer (ITK-SNAP or 3D Slicer) we identified a visually appealing window-level setting
T1_WINDOW_LEVEL = (1050,500)

##  Read Data and Select Seed Point
img_T1 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd"))
# Rescale the intensities and map them to [0,255], these are the default values for the output
# We will use this image to display the results of segmentation
img_T1_255 = sitk.Cast(sitk.IntensityWindowing(img_T1, 
                                               windowMinimum=T1_WINDOW_LEVEL[1]-T1_WINDOW_LEVEL[0]/2.0, 
                                               windowMaximum=T1_WINDOW_LEVEL[1]+T1_WINDOW_LEVEL[0]/2.0), 
                       sitk.sitkUInt8)

point_acquisition_interface = gui.PointDataAquisition(img_T1, window_level=(1050,500))

#preselected seed point in the left ventricle  
point_acquisition_interface.set_point_indexes([(132,142,96)])

initial_seed_point_indexes = point_acquisition_interface.get_point_indexes()

##  ConnectedThreshold
seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=170)
# Overlay the segmentation onto the T1 image
gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img_T1_255, seg_explicit_thresholds)],                   
                      title_list = ['connected threshold result'])

# ConfidenceConnected¶
seg_implicit_thresholds = sitk.ConfidenceConnected(img_T1, seedList=initial_seed_point_indexes,
                                                   numberOfIterations=0,
                                                   multiplier=2,
                                                   initialNeighborhoodRadius=1,
                                                   replaceValue=1)

gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img_T1_255, seg_implicit_thresholds)],                   
                      title_list = ['confidence connected result'])

# VectorConfidenceConnected 
img_T2 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT2.nrrd"))
img_multi = sitk.Compose(img_T1, img_T2)

seg_implicit_threshold_vector = sitk.VectorConfidenceConnected(img_multi, 
                                                               initial_seed_point_indexes, 
                                                               numberOfIterations=2, 
                                                               multiplier=4)

gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img_T1_255, seg_implicit_threshold_vector)],                   
                      title_list = ['vector confidence connected result'])

vectorRadius=(1,1,1)
kernel=sitk.sitkBall
seg_implicit_thresholds_clean = sitk.BinaryMorphologicalClosing(seg_implicit_thresholds, 
                                                                vectorRadius,
                                                                kernel)

gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img_T1_255, seg_implicit_thresholds), 
                                sitk.LabelOverlay(img_T1_255, seg_implicit_thresholds_clean)], 
                  shared_slider=True,
                  title_list = ['before morphological closing', 'after morphological closing'])
