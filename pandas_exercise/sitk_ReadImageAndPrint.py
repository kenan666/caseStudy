'''
This example illustrates how to read only an image’s information and
meta-data dictionary without loading the pixel content via the ImageFileReader.

Reading an entire image potentially is memory and time intensive operation when the image is large or many 
files must be read. The image information and meta-data dictionary can be read without the bulk data by 
using the ImageFilerReader’s object oriented interface, 
with use of the ImageFileReader::ReadImageInformation method.

While all file formats support loading image information such as size, pixel type, origin, 
and spacing many image types do not have a meta-data dictionary. 
The most common case for images with a dictionary is DICOM, but also the fields from TIFF, NIFTI, MetaIO 
and other file formats maybe loaded into the meta-data dictionary.

For efficiency, the default DICOM reader settings will only load public tags (even group numbers). 
In the example we explicitly set the reader to also load private tags (odd group numbers). 
For further information on DICOM data elements see the standard part 5, Data Structures and Encoding.

'''



import SimpleITK as sitk
import sys,os

if len(sys.argv) < 2:
    print( "Usage: DicomImagePrintTags <input_file>" )
    sys.exit ( 1 )

reader = sitk.ImageFileReader()

reader.SetFileName()
reader.LoadPrivateTagOn();

reader.ReadImageInformation();

for k in reader.GetMetaDataKeys():
    v = reader.GetMetaData(k)
    print("({0}) = = \"{1}\"".format(k,v))

print("Image Size : {0}".format(reader.GetSize()));
print("Image PixelType : {0}".format(sitk.GetPixelIDValueAsString(reader.GetPixelID())))
