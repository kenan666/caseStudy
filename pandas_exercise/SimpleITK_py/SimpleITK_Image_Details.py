# pixel  type  sitkUInt8,sitkInt16....
from __future__ import print_function
import SimpleITK as sitk

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import interact,fixed
import os 

OUTPUT_DIR = 'Output'

# Utility method that either downloads data from the Girder repository or
# if already downloaded returns the file name for reading from disk (cached data).
#%run update_path_to_download_script
from downloaddata import fetch_data as fdata

#  load image and display it
logo = sitk.ReadImage(fdata('sitkImage.jpg'))

plt.imshow(sitk.GetArrayViewFromImage(logo))
plt.axis('off')

# image construction

image_3D = sitk.Image(256,126,64,sitk.sitkInt16)
image_2D = sitk.Image(64,64,sitk.sitkFloat32)
image_2D = sitk.Image([32,32],sitk.sitkUInt32)
image_RGB = sitk.Image([128,64],sitk.sitkVectorInt8,3)

# basic image attribution

image_3D.SetOrigin((78.0, 76.0, 77.0))
image_3D.SetSpacing([0.5,0.5,3.0])

print(image_3D.GetOrigin())
print(image_3D.GetSize())
print(image_3D.GetSpacing())
print(image_3D.GetDirection())

print(image_3D.GetDimension())
print(image_3D.GetWidth())
print(image_3D.GetHeight())
print(image_3D.GetDepth())

print(image_2D.GetSize())
print(image_2D.GetDepth())

print(image_3D.GetPixelIDValue())
print(image_3D.GetPixelIDTypeAsString())
print(image_3D.GetNumberOfComponentsPerPixel())

print(image_RGB.GetDimension())
print(image_RGB.GetSize())
print(image_RGB.GetNumberOfComponentsPerPixel())

# accessing pixel and slicing
help(image_3D.GetPixel)

print(image_3D.GetPixel(0, 0, 0))
image_3D.SetPixel(0, 0, 0, 1)
print(image_3D.GetPixel(0, 0, 0))

'''
# This can also be done using Pythonic notation.
print(image_3D[0,0,1])
image_3D[0,0,1] = 2
print(image_3D[0,0,1])
'''

# Brute force sub-sampling 
logo_subsampled = logo[::2,::2]

# Get the sub-image containing the word Simple
simple = logo[0:115,:]

# Get the sub-image containing the word Simple and flip it
simple_flipped = logo[115:0:-1,:]

n = 4

plt.subplot(n,1,1)
plt.imshow(sitk.GetArrayViewFromImage(logo))
plt.axis('off');

plt.subplot(n,1,2)
plt.imshow(sitk.GetArrayViewFromImage(logo_subsampled))
plt.axis('off');

plt.subplot(n,1,3)
plt.imshow(sitk.GetArrayViewFromImage(simple))
plt.axis('off')

plt.subplot(n,1,4)
plt.imshow(sitk.GetArrayViewFromImage(simple_flipped))
plt.axis('off');

# conversion between numpy and simpleitk
'''
We have two options for converting from SimpleITK to numpy:

GetArrayFromImage(): returns a copy of the image data. 
You can then freely modify the data as it has no effect on the original SimpleITK image.

GetArrayViewFromImage(): returns a view on the image data which is useful for display in a memory efficient manner. 
You cannot modify the data and the view will be invalid if the original SimpleITK image is deleted.
'''
#  GetImageFromArray()

nda = sitk.GetArrayFromImage(image_3D)
print(image_3D.GetSize())
print(nda.shape)

nda = sitk.GetArrayFromImage(image_RGB)
print(image_RGB.GetSize())
print(nda.shape)


gabor_image = sitk.GaborSource(size=[64,64], frequency=.03)
# Getting a numpy array view on the image data doesn't copy the data
nda_view = sitk.GetArrayViewFromImage(gabor_image)
plt.imshow(nda_view, cmap=plt.cm.Greys_r)
plt.axis('off');

# Trying to assign a value to the array view will throw an exception
nda_view[0,0] = 255

#  form numpy to simpleitk


nda = np.zeros((10,20,3))

        #if this is supposed to be a 3D gray scale image [x=3, y=20, z=10]
img = sitk.GetImageFromArray(nda)
print(img.GetSize())

      #if this is supposed to be a 2D color image [x=20,y=10]
img = sitk.GetImageFromArray(nda, isVector=True)
print(img.GetSize())

'''
The following code cell illustrates a situation where your code is a combination of SimpleITK methods and custom Python code which works with intensity values 
or labels outside of SimpleITK. This is a reasonable approach when you implement an algorithm in Python and don't care about the physical spacing of things .
'''

def my_algorithm(image_as_numpy_array):
    # res is the image result of your algorithm, has the same grid size as the original image
    res = image_as_numpy_array
    return res

# Starting with SimpleITK
img = sitk.ReadImage(fdata('training_001_mr_T1.mha'))

# Custom Python code working on a numpy array.
npa_res = my_algorithm(sitk.GetArrayFromImage(img))

# Converting back to SimpleITK (assumes we didn't move the image in space as we copy the information from the original)
res_img = sitk.GetImageFromArray(npa_res)
res_img.CopyInformation(img)

# Continuing to work with SimpleITK images
res_img - img

# image operation
# SimpleITK supports basic arithmetic operations between images, taking into account their physical space.

img1 = sitk.Image(24,24, sitk.sitkUInt8)
img1[0,0] = 0

img2 = sitk.Image(img1.GetSize(), sitk.sitkUInt8)
img2.SetDirection([0,1,0.5,0.5])
img2.SetSpacing([0.5,0.8])
img2.SetOrigin([0.000001,0.000001])
img2[0,0] = 255

img3 = img1 + img2
print(img3[0,0])

# reading and writing
img = sitk.ReadImage(fdata('SimpleITK.jpg'))
print(img.GetPixelIDTypeAsString())

# write as PNG and BMP
sitk.WriteImage(img, os.path.join(OUTPUT_DIR, 'SimpleITK.png'))
sitk.WriteImage(img, os.path.join(OUTPUT_DIR, 'SimpleITK.bmp'))

# Read an image in JPEG format and cast the pixel type according to user selection.
# Several pixel types, some make sense in this case (vector types) and some are just show
# that the user's choice will force the pixel type even when it doesn't make sense
# (e.g. sitkVectorUInt16 or sitkUInt8).
pixel_types = { 'sitkUInt8': sitk.sitkUInt8,
                'sitkUInt16' : sitk.sitkUInt16,
                'sitkFloat64' : sitk.sitkFloat64,
                'sitkVectorUInt8' : sitk.sitkVectorUInt8,
                'sitkVectorUInt16' : sitk.sitkVectorUInt16,
                'sitkVectorFloat64' : sitk.sitkVectorFloat64}

def pixel_type_dropdown_callback(pixel_type, pixel_types_dict):
    #specify the file location and the pixel type we want
    img = sitk.ReadImage(fdata('SimpleITK.jpg'), pixel_types_dict[pixel_type])
    
    print(img.GetPixelIDTypeAsString())
    print(img[0,0])
    plt.imshow(sitk.GetArrayViewFromImage(img))
    plt.axis('off')
 
interact(pixel_type_dropdown_callback, pixel_type=list(pixel_types.keys()), pixel_types_dict=fixed(pixel_types));

# Read a DICOM series and write it as a single mha file
data_directory = os.path.dirname(fdata("CIRS057A_MR_CT_DICOM/readme.txt"))
series_ID = '1.2.840.113619.2.290.3.3233817346.783.1399004564.515'

# Get the list of files belonging to a specific series ID.
reader = sitk.ImageSeriesReader()
# Use the functional interface to read the image series.
original_image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory, series_ID))

# Write the image.
output_file_name_3D = os.path.join(OUTPUT_DIR, '3DImage.mha')
sitk.WriteImage(original_image, output_file_name_3D)

# Read it back again.
written_image = sitk.ReadImage(output_file_name_3D)

# Check that the original and written image are the same.
statistics_image_filter = sitk.StatisticsImageFilter()
statistics_image_filter.Execute(original_image - written_image)

# Check that the original and written files are the same
print('Max, Min differences are : {0}, {1}'.format(statistics_image_filter.GetMaximum(), statistics_image_filter.GetMinimum()))


# Write an image series as JPEG.
sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(written_image), sitk.sitkUInt8), 
                [os.path.join(OUTPUT_DIR, 'slice{0:03d}.jpg'.format(i)) for i in range(written_image.GetSize()[2])])

# Select a specific DICOM series from a directory and only then load user selection.
data_directory = os.path.dirname(fdata("CIRS057A_MR_CT_DICOM/readme.txt"))
# Global variable 'selected_series' is updated by the interact function
selected_series = ''
file_reader = sitk.ImageFileReader()
def DICOM_series_dropdown_callback(series_to_load, series_dictionary):
    global selected_series
               # ------------------------------------------------------------------------------------------------------------
               # Print some information about the series from the meta-data dictionary
               # DICOM standard part 6, Data Dictionary: http://medical.nema.org/medical/dicom/current/output/pdf/part06.pdf
               # ------------------------------------------------------------------------------------------------------------
    file_reader.SetFileName(series_dictionary[series_to_load][0])
    file_reader.ReadImageInformation()
    tags_to_print = {'0010|0010': 'Patient name: ', 
                     '0008|0060' : 'Modality: ',
                     '0008|0021' : 'Series date: ',
                     '0008|0080' : 'Institution name: ',
                     '0008|1050' : 'Performing physician\'s name: '}
    for tag in tags_to_print:
        try:
            print(tags_to_print[tag] + file_reader.GetMetaData(tag))
        except: # Ignore if the tag isn't in the dictionary
            pass
    selected_series = series_to_load                    

# Directory contains multiple DICOM studies/series, store
             # in dictionary with key being the series ID
reader = sitk.ImageSeriesReader()
series_file_names = {}
series_IDs = reader.GetGDCMSeriesIDs(data_directory)
            # Check that we have at least one series
if series_IDs:
    for series in series_IDs:
        series_file_names[series] = reader.GetGDCMSeriesFileNames(data_directory, series)
    
    interact(DICOM_series_dropdown_callback, series_to_load=list(series_IDs), series_dictionary=fixed(series_file_names)); 
else:
    print('Data directory does not contain any DICOM series.')

reader.SetFileNames(series_file_names[selected_series])
img = reader.Execute()
# Display the image slice from the middle of the stack, z axis
z = int(img.GetDepth()/2)
plt.imshow(sitk.GetArrayViewFromImage(img)[z,:,:], cmap=plt.cm.Greys_r)
plt.axis('off');

# Selecting a Specific Image IO

file_reader = sitk.ImageFileReader()

# Get a tuple listing all registered ImageIOs
image_ios_tuple = file_reader.GetRegisteredImageIOs()
print("The supported image IOs are: " + str(image_ios_tuple))

# Optionally, just print the reader and see which ImageIOs are registered
print('\n',file_reader)

# Specify the JPEGImageIO and read file
file_reader.SetImageIO('JPEGImageIO')
file_reader.SetFileName(fdata('SimpleITK.jpg'))
logo = file_reader.Execute()

# Unfortunately, now reading a non JPEG image will fail
try:
    file_reader.SetFileName(fdata('cthead1.png'))
    ct_head = file_reader.Execute()
except RuntimeError:
    print('Got a RuntimeError exception.')
    
# We can reset the file reader to its default behaviour so that it automatically 
# selects the ImageIO
file_reader.SetImageIO('')
ct_head = file_reader.Execute()

# Streaming Image IO
file_reader = sitk.ImageFileReader()
file_reader.SetFileName(fdata('vm_head_rgb.mha'))

file_reader.ReadImageInformation()
image_size = file_reader.GetSize()
start_index, extract_size = zip(*[(int(1.0/3.0*sz), int(1.0/3.0*sz)) for sz in file_reader.GetSize()])

file_reader.SetExtractIndex(start_index)
file_reader.SetExtractSize(extract_size)

sitk.Show(file_reader.Execute())

'''
The code show how to subtract two large images from each other with a smaller memory footprint than the direct approach, 
though the code is much more complex and slower than the direct approach:

sitk.ReadImage(image1_file_name) - sitk.ReadImage(image2_file_name)

Note: The code assume that the two images occupy the same spatial region (origin, spacing, direction cosine matrix).
'''

def streaming_subtract(image1_file_name, image2_file_name, parts):
    '''
    Subtract image1 from image2 using 'parts' number of sub-regions.
    '''
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(image1_file_name)
    file_reader.ReadImageInformation()
    image_size = file_reader.GetSize()

    # Create the result image, initially empty
    result_img = sitk.Image(file_reader.GetSize(), file_reader.GetPixelID(), file_reader.GetNumberOfComponents())
    result_img.SetSpacing(file_reader.GetSpacing())
    result_img.SetOrigin(file_reader.GetOrigin())
    result_img.SetDirection(file_reader.GetDirection())

    extract_size = list(file_reader.GetSize())
    extract_size[-1] = extract_size[-1]//parts
    current_index = [0]*file_reader.GetDimension()
    for i in range(parts):
        if i == (parts-1): # last region may be smaller than the standard extract region
            extract_size[-1] = image_size[-1] - current_index[-1]
        file_reader.SetFileName(image1_file_name)
        file_reader.SetExtractIndex(current_index)
        file_reader.SetExtractSize(extract_size)
        sub_image1 = file_reader.Execute()

        file_reader.SetFileName(image2_file_name)
        file_reader.SetExtractIndex(current_index)
        file_reader.SetExtractSize(extract_size)
        sub_image2 = file_reader.Execute()
        # Paste the result of subtracting the two subregions into their location in the result_img
        result_img = sitk.Paste(result_img, sub_image1 - sub_image2, extract_size, [0]*file_reader.GetDimension(), current_index)
        current_index[-1] += extract_size[-1]        
    return result_img

# If you have the patience and RAM you can try this with the vm_head_rgb.mha image.
image1_file_name = fdata('fib_sem_bacillus_subtilis.mha')
image2_file_name = fdata('fib_sem_bacillus_subtilis.mha')

result_img = streaming_subtract(image1_file_name, image2_file_name, parts=5)
del result_img

result_img = sitk.ReadImage(image1_file_name) - sitk.ReadImage(image2_file_name)
del result_img