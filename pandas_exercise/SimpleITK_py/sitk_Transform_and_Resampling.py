from __future__ import print_function
import SimpleITK as sitk
import numpy as np
%matplotlib inline
from matplotlib import pyplot as plt
from ipywidgets import interact, fixed

# Utility method that either downloads data from the Girder repository or
# if already downloaded returns the file name for reading from disk (cached data).
%run update_path_to_download_script
from downloaddata import fetch_data as fdata

'''
Creating and Manipulating Transforms
A number of different spatial transforms are available in SimpleITK.

The simplest is the Identity Transform. This transform simply returns input points unaltered.
'''
dimension = 2

print('*Identity Transform*')
identity = sitk.Transform(dimension, sitk.sitkIdentity)
print('Dimension: ' + str(identity.GetDimension()))

# Points are always defined in physical space
point = (1.0, 1.0)
def transform_point(transform, point):
    transformed_point = transform.TransformPoint(point)
    print('Point ' + str(point) + ' transformed is ' + str(transformed_point))

transform_point(identity, point)

'''
Transform are defined by two sets of parameters, the Parameters and FixedParameters. 
FixedParameters are not changed during the optimization process when performing registration. For the TranslationTransform, the Parameters are the values of the translation Offset.
'''
print('*Translation Transform*')
translation = sitk.TranslationTransform(dimension)

print('Parameters: ' + str(translation.GetParameters()))
print('Offset:     ' + str(translation.GetOffset()))
print('FixedParameters: ' + str(translation.GetFixedParameters()))
transform_point(translation, point)

print('')
translation.SetParameters((3.1, 4.4))
print('Parameters: ' + str(translation.GetParameters()))
transform_point(translation, point)

# The affine transform is capable of representing translations, rotations, shearing, and scaling.
print('*Affine Transform*')
affine = sitk.AffineTransform(dimension)

print('Parameters: ' + str(affine.GetParameters()))
print('FixedParameters: ' + str(affine.GetFixedParameters()))
transform_point(affine, point)

print('')
affine.SetTranslation((3.1, 4.4))
print('Parameters: ' + str(affine.GetParameters()))
transform_point(affine, point)

#  Applying Transforms to Images
def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()
        
    ysize = nda.shape[0]
    xsize = nda.shape[1]
      
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], 0, ysize*spacing[0])
    
    t = ax.imshow(nda,
            extent=extent,
            interpolation='hamming',
            cmap='gray',
            origin='lower')
    
    if(title):
        plt.title(title)

# create a grid image
grid = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
    size=(250, 250),
    sigma=(0.5, 0.5),
    gridSpacing=(5.0, 5.0),
    gridOffset=(0.0, 0.0),
    spacing=(0.2,0.2))
myshow(grid, 'Grid Input')

#  To apply the transform,  resampling first
def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

translation.SetOffset((3.1, 4.6))
transform_point(translation, point)
resampled = resample(grid, translation)
myshow(resampled, 'Resampled Translation')

translation.SetOffset(-1*np.array(translation.GetParameters()))
transform_point(translation, point)
resampled = resample(grid, translation)
myshow(resampled, 'Inverse Resampled')

# affine transformation
def affine_translate(transform, x_translation=3.1, y_translation=4.6):
    new_transform = sitk.AffineTransform(transform)
    new_transform.SetTranslation((x_translation, y_translation))
    resampled = resample(grid, new_transform)
    myshow(resampled, 'Translated')
    return new_transform
    
affine = sitk.AffineTransform(dimension)

interact(affine_translate, transform=fixed(affine), x_translation=(-5.0, 5.0), y_translation=(-5.0, 5.0))

'''
# scaling
def affine_scale(transform, x_scale=3.0, y_scale=0.7):
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
    matrix[0,0] = x_scale
    matrix[1,1] = y_scale
    new_transform.SetMatrix(matrix.ravel())
    resampled = resample(grid, new_transform)
    myshow(resampled, 'Scaled')
    print(matrix)
    return new_transform

affine = sitk.AffineTransform(dimension)

interact(affine_scale, transform=fixed(affine), x_scale=(0.2, 5.0), y_scale=(0.2, 5.0))

'''
'''
# rotation
def affine_rotate(transform, degrees=15.0):
    parameters = np.array(transform.GetParameters())
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
    radians = -np.pi * degrees / 180.
    rotation = np.array([[np.cos(radians), -np.sin(radians)],[np.sin(radians), np.cos(radians)]])
    new_matrix = np.dot(rotation, matrix)
    new_transform.SetMatrix(new_matrix.ravel())
    resampled = resample(grid, new_transform)
    print(new_matrix)
    myshow(resampled, 'Rotated')
    return new_transform
    
affine = sitk.AffineTransform(dimension)

interact(affine_rotate, transform=fixed(affine), degrees=(-90.0, 90.0))

'''
'''
# shearing
def affine_shear(transform, x_shear=0.3, y_shear=0.1):
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
    matrix[0,1] = -x_shear
    matrix[1,0] = -y_shear
    new_transform.SetMatrix(matrix.ravel())
    resampled = resample(grid, new_transform)
    myshow(resampled, 'Sheared')
    print(matrix)
    return new_transform

affine = sitk.AffineTransform(dimension)

interact(affine_shear, transform=fixed(affine), x_shear=(0.1, 2.0), y_shear=(0.1, 2.0))

'''
# composite transform
'''
It is possible to compose multiple transform together into a single transform object. 
With a composite transform, multiple resampling operations are prevented, so interpolation errors are not accumulated.
For example, an affine transformation that consists of a translation and rotation,
'''
translate = (8.0, 16.0)
rotate = 20.0

affine = sitk.AffineTransform(dimension)
affine = affine_translate(affine, translate[0], translate[1])
affine = affine_rotate(affine, rotate)

resampled = resample(grid, affine)
myshow(resampled, 'Single Transform')

# can also be represented with two Transform objects applied in sequence with a Composite Transform
composite = sitk.Transform(dimension, sitk.sitkComposite)
translation = sitk.TranslationTransform(dimension)
translation.SetOffset(-1*np.array(translate))
composite.AddTransform(translation)
affine = sitk.AffineTransform(dimension)
affine = affine_rotate(affine, rotate)

composite.AddTransform(translation)
composite = sitk.Transform(dimension, sitk.sitkComposite)
composite.AddTransform(affine)

resampled = resample(grid, composite)
myshow(resampled, 'Two Transforms')
# ---------------------------------------
composite = sitk.Transform(dimension, sitk.sitkComposite)
composite.AddTransform(affine)
composite.AddTransform(translation)

resampled = resample(grid, composite)
myshow(resampled, 'Composite transform in reverse order')

#  resampling

def resample_display(image, euler2d_transform, tx, ty, theta):
    euler2d_transform.SetTranslation((tx, ty))
    euler2d_transform.SetAngle(theta)
    
    resampled_image = sitk.Resample(image, euler2d_transform)
    plt.imshow(sitk.GetArrayFromImage(resampled_image))
    plt.axis('off')    
    plt.show()

logo = sitk.ReadImage(fdata('SimpleITK.jpg'))

euler2d = sitk.Euler2DTransform()
# Why do we set the center?
euler2d.SetCenter(logo.TransformContinuousIndexToPhysicalPoint(np.array(logo.GetSize())/2.0))
interact(resample_display, image=fixed(logo), euler2d_transform=fixed(euler2d), tx=(-128.0, 128.0,2.5), ty=(-64.0, 64.0), theta=(-np.pi/4.0,np.pi/4.0));

#  Defining the Resampling Grid

euler2d = sitk.Euler2DTransform()
# Why do we set the center?
euler2d.SetCenter(logo.TransformContinuousIndexToPhysicalPoint(np.array(logo.GetSize())/2.0))

tx = 64
ty = 32
euler2d.SetTranslation((tx, ty))

extreme_points = [logo.TransformIndexToPhysicalPoint((0,0)), 
                  logo.TransformIndexToPhysicalPoint((logo.GetWidth(),0)),
                  logo.TransformIndexToPhysicalPoint((logo.GetWidth(),logo.GetHeight())),
                  logo.TransformIndexToPhysicalPoint((0,logo.GetHeight()))]
inv_euler2d = euler2d.GetInverse()

extreme_points_transformed = [inv_euler2d.TransformPoint(pnt) for pnt in extreme_points]
min_x = min(extreme_points_transformed)[0]
min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
max_x = max(extreme_points_transformed)[0]
max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]

# Use the original spacing (arbitrary decision).
output_spacing = logo.GetSpacing()
# Identity cosine matrix (arbitrary decision).   
output_direction = [1.0, 0.0, 0.0, 1.0]
# Minimal x,y coordinates are the new origin.
output_origin = [min_x, min_y]
# Compute grid size based on the physical size and spacing.
output_size = [int((max_x-min_x)/output_spacing[0]), int((max_y-min_y)/output_spacing[1])]

resampled_image = sitk.Resample(logo, output_size, euler2d, sitk.sitkLinear, output_origin, output_spacing, output_direction)
plt.imshow(sitk.GetArrayViewFromImage(resampled_image))
plt.axis('off')    
plt.show()

# Resampling at a set of locations

img = logo

# Generate random samples inside the image, we will obtain the intensity/color values at these points.
num_samples = 10
physical_points = []
for pnt in zip(*[list(np.random.random(num_samples)*sz) for sz in img.GetSize()]):
    physical_points.append(img.TransformContinuousIndexToPhysicalPoint(pnt))

# Create an image of size [num_samples,1...1], actual size is dependent on the image dimensionality. The pixel
# type is irrelevant, as the image is just defining the interpolation grid (sitkUInt8 has minimal memory footprint).
interp_grid_img = sitk.Image([num_samples] + [1]*(img.GetDimension()-1), sitk.sitkUInt8)

# Define the displacement field transformation, maps the points in the interp_grid_img to the points in the actual
# image.
displacement_img = sitk.Image([num_samples] + [1]*(img.GetDimension()-1), sitk.sitkVectorFloat64, img.GetDimension())
for i, pnt in enumerate(physical_points):
     displacement_img[[i] + [0]*(img.GetDimension()-1)] = np.array(pnt) - np.array(interp_grid_img.TransformIndexToPhysicalPoint([i] + [0]*(img.GetDimension()-1)))

# Actually perform the resampling. The only relevant choice here is the interpolator. The default_output_pixel_value
# is set to 0.0, but the resampling should never use it because we expect all points to be inside the image and this
# value is only used if the point is outside the image extent.
interpolator_enum = sitk.sitkLinear
default_output_pixel_value = 0.0
output_pixel_type = sitk.sitkFloat32 if img.GetNumberOfComponentsPerPixel()==1 else sitk.sitkVectorFloat32
resampled_points = sitk.Resample(img, interp_grid_img, sitk.DisplacementFieldTransform(displacement_img), 
                                 interpolator_enum, default_output_pixel_value, output_pixel_type)

# Print the interpolated values per point
for i in range(resampled_points.GetWidth()):
      print(str(physical_points[i]) + ': ' + str(resampled_points[[i] + [0]*(img.GetDimension()-1)]) + '\n')

