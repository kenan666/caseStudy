#  Segmentation_Shape_Analysis
# 1、SimpleITK supports a large number of filters that facilitate classical segmentation algorithms (variety of thresholding algorithms, watersheds...).
# 2、Once your data is segmented SimpleITK enables you to efficiently post process the segmentation (e.g. label distinct objects, analyze object shapes).
import SimpleITK as sitk
import pandas as pd

%matplotlib notebook

import matplotlib.pyplot as plt
import gui
from math import ceil

%run update_path_to_download_script
from downloaddata import fetch_data as fdata

#  load data
#  Load the 3D volume and display it.
img = sitk.ReadImage(fdata("fib_sem_bacillus_subtilis.mha"))
gui.MultiImageDisplay(image_list = [img], figure_size=(8,4));

#  Segmentation
'''
To allow us to analyze the shape of whole bacteria we first need to segment them. We will do this in several steps:

Separate the bacteria from the embedding resin background.
Mark each potential bacterium with a unique label, to evaluate the segmentation.
Remove small components and fill small holes using binary morphology operators (opening and closing).
Use seed based watersheds to perform final segmentation.
Remove bacterium that are connected to the image boundary.
'''
#  Separate the bacteria from the background
plt.figure()
plt.hist(sitk.GetArrayViewFromImage(img).flatten(), bins=100)
plt.show()

threshold_filters = {'Otsu': sitk.OtsuThresholdImageFilter(),
                     'Triangle' : sitk.TriangleThresholdImageFilter(),
                     'Huang' : sitk.HuangThresholdImageFilter(),
                     'MaxEntropy' : sitk.MaximumEntropyThresholdImageFilter()}

filter_selection = 'Manual'
try:
  thresh_filter = threshold_filters[filter_selection]
  thresh_filter.SetInsideValue(0)
  thresh_filter.SetOutsideValue(1)
  thresh_img = thresh_filter.Execute(img)
  thresh_value = thresh_filter.GetThreshold()
except KeyError:
  thresh_value = 120
  thresh_img = img>thresh_value

print("Threshold used: " + str(thresh_value))    
gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img, thresh_img)],                   
                      title_list = ['Binary Segmentation'], figure_size=(8,4));

#  Mark each potential bacterium with unique label and evaluate
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(sitk.ConnectedComponent(thresh_img))

# Look at the distribution of sizes of connected components (bacteria).
label_sizes = [ stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 1]

plt.figure()
plt.hist(label_sizes,bins=200)
plt.title("Distribution of Object Sizes")
plt.xlabel("size in pixels")
plt.ylabel("number of objects")
plt.show()

#  Remove small islands and holes
cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(thresh_img, [10, 10, 10])
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [10, 10, 10])

gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img, cleaned_thresh_img)],                   
                      title_list = ['Cleaned Binary Segmentation'], figure_size=(8,4));

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(sitk.ConnectedComponent(cleaned_thresh_img))

# Look at the distribution of sizes of connected components (bacteria).
label_sizes = [ stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 1]

plt.figure()
plt.hist(label_sizes,bins=200)
plt.title("Distribution of Object Sizes")
plt.xlabel("size in pixels")
plt.ylabel("number of objects")
plt.show()

gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img, sitk.ConnectedComponent(cleaned_thresh_img))],                   
                      title_list = ['Cleaned Binary Segmentation'],figure_size=(8,4));

#  Seed based watershed segmentation
dist_img = sitk.SignedMaurerDistanceMap(cleaned_thresh_img != 0, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
radius = 10
# Seeds have a distance of "radius" or more to the object boundary, they are uniquely labelled.
seeds = sitk.ConnectedComponent(dist_img < -radius)
# Relabel the seed objects using consecutive object labels while removing all objects with less than 15 pixels.
seeds = sitk.RelabelComponent(seeds, minimumObjectSize=15)
# Run the watershed segmentation using the distance map and seeds.
ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds, markWatershedLine=True)
ws = sitk.Mask( ws, sitk.Cast(cleaned_thresh_img, ws.GetPixelID()))

gui.MultiImageDisplay(image_list = [dist_img,
                                    sitk.LabelOverlay(img, seeds),
                                    sitk.LabelOverlay(img, ws)],                   
                      title_list = ['Segmentation Distance',
                                    'Watershed Seeds',
                                    'Binary Watershed Labeling'],
                      shared_slider=True,
                      horizontal=False,
                      figure_size=(6,12));

#  Removal of objects touching the image boundary
# The image has a small black border which we account for here.
bgp = sitk.BinaryGrindPeak( (ws!=0)| (img==0))
non_border_seg = sitk.Mask( ws, bgp==0)
gui.MultiImageDisplay(image_list = [sitk.LabelOverlay(img, non_border_seg)],                   
                      title_list = ['Final Segmentation'],figure_size=(8,4));

#  Object Analysis
shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.ComputeOrientedBoundingBoxOn()
shape_stats.Execute(non_border_seg)

intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
intensity_stats.Execute(non_border_seg,img)

stats_list = [ (shape_stats.GetPhysicalSize(i),
               shape_stats.GetElongation(i),
               shape_stats.GetFlatness(i),
               shape_stats.GetOrientedBoundingBoxSize(i)[0],
               shape_stats.GetOrientedBoundingBoxSize(i)[2],
               intensity_stats.GetMean(i),
               intensity_stats.GetStandardDeviation(i),
               intensity_stats.GetSkewness(i)) for i in shape_stats.GetLabels()]
cols=["Volume (nm^3)",
      "Elongation",
      "Flatness",
      "Oriented Bounding Box Minimum Size(nm)",
      "Oriented Bounding Box Maximum Size(nm)",
     "Intensity Mean",
     "Intensity Standard Deviation",
     "Intensity Skewness"]

# Create the pandas data frame and display descriptive statistics.
stats = pd.DataFrame(data=stats_list, index=shape_stats.GetLabels(), columns=cols)
stats.describe()

fig, axes = plt.subplots(nrows=len(cols), ncols=2, figsize=(6,4*len(cols)))
axes[0,0].axis('off')

stats.loc[:,cols[0]].plot.hist(ax=axes[0,1], bins=25)
axes[0,1].set_xlabel(cols[0])
axes[0,1].xaxis.set_label_position("top")

for i in range(1,len(cols)):
    c = cols[i]
    bar = stats.loc[:,[c]].plot.hist(ax=axes[i,0], bins=20,orientation='horizontal',legend=False)
    bar.set_ylabel(stats.loc[:,[c]].columns.values[0])    
    scatter = stats.plot.scatter(ax=axes[i,1],y=c,x=cols[0])
    scatter.set_ylabel('')
    # Remove axis labels from all plots except the last (they all share the labels)
    if(i<len(cols)-1):
        bar.set_xlabel('')
        scatter.set_xlabel('')
# Adjust the spacing between plot columns and set the plots to have a tight
# layout inside the figure.
plt.subplots_adjust(wspace=0.4)
plt.tight_layout()

bacteria_labels = shape_stats.GetLabels()
bacteria_volumes = [shape_stats.GetPhysicalSize(label) for label in bacteria_labels] 
num_images = 5 # number of bacteria images we want to display

bacteria_labels_volume_sorted = [label for _,label in sorted(zip(bacteria_volumes, bacteria_labels))]

resampler = sitk.ResampleImageFilter()
aligned_image_spacing = [10,10,10] #in nanometers

for label in bacteria_labels_volume_sorted[0:num_images]:
    aligned_image_size = [ int(ceil(shape_stats.GetOrientedBoundingBoxSize(label)[i]/aligned_image_spacing[i])) for i in range(3) ]
    direction_mat = shape_stats.GetOrientedBoundingBoxDirection(label)
    aligned_image_direction = [direction_mat[0], direction_mat[3], direction_mat[6], 
                               direction_mat[1], direction_mat[4], direction_mat[7],
                               direction_mat[2], direction_mat[5], direction_mat[8] ] 
    resampler.SetOutputDirection(aligned_image_direction)
    resampler.SetOutputOrigin(shape_stats.GetOrientedBoundingBoxOrigin(label))
    resampler.SetOutputSpacing(aligned_image_spacing)
    resampler.SetSize(aligned_image_size)
    
    obb_img = resampler.Execute(img)
    # Change the image axes order so that we have a nice display.
    obb_img = sitk.PermuteAxes(obb_img,[2,1,0])
    gui.MultiImageDisplay(image_list = [obb_img],                   
                          title_list = ["OBB_{0}".format(label)])