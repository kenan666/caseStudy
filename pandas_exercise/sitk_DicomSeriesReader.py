'''
This example illustrates how to read a DICOM series into a 3D volume. 
Additional actions include printing some information, writing the image and possibly displaying it using 
the default display program via the SimpleITK Show function. 
The program makes several assumptions: the given directory contains at least one DICOM series, 
if there is more than one series the first series is read, and the default SimpleITK external viewer is installed.
'''


from __future__ import print_function

import SimpleITK as sitk
import sys,os

if len(sys.argv) < 3:
    print( "Usage: DicomSeriesReader <input_directory> <output_file>" )
    sys.exit ( 1 )

print("Reading Dicom directory :",sys.argv[1])
reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames(sys.argv[1])
reader.SetFileNames(dicom_names)

image = reader.Execute()

size = image.GetSize()
print("image size :",size[0],size[1],size[2])

print("writing image :",sys.argv[2])

if (not "SITK_NOSHOW" in os.environ):
    sitk.Show(image,'Dicom Series')
