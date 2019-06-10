# Data_Augmentation
#  Data Augmentation for Deep Learning
import SimpleITK as sitk
import numpy as np

import gui

#utility method that either downloads data from the Girder repository or
#if already downloaded returns the file name for reading from disk (cached data)

from downloaddata import fetch_data as fdata

OUTPUT_DIR = 'Output'

# tips
# The image we will resample (a grid).
grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16, size=(512,512), 
                             sigma=(0.1,0.1), gridSpacing=(20.0,20.0))
sitk.Show(grid_image, "original grid image")

# The spatial definition of the images we want to use in a deep learning framework (smaller than the original). 
new_size = [100, 100]
reference_image = sitk.Image(new_size, grid_image.GetPixelIDValue())
reference_image.SetOrigin(grid_image.GetOrigin())
reference_image.SetDirection(grid_image.GetDirection())
reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, grid_image.GetSize(), grid_image.GetSpacing())])

# Resample without any smoothing.
sitk.Show(sitk.Resample(grid_image, reference_image) , "resampled without smoothing")

# Resample after Gaussian smoothing.
sitk.Show(sitk.Resample(sitk.SmoothingRecursiveGaussian(grid_image, 2.0), reference_image), "resampled with smoothing")

#  load data
data = [sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd")),
        sitk.ReadImage(fdata("vm_head_mri.mha")),
        sitk.ReadImage(fdata("head_mr_oriented.mha"))]
# Comment out the following line if you want to work in 3D. Note that in 3D some of the notebook visualizations are 
# disabled. 
data = [data[0][:,160,:], data[1][:,:,17], data[2][:,:,0]]
def disp_images(images, fig_size, wl_list=None):
    if images[0].GetDimension()==2:
      gui.multi_image_display2D(image_list=images, figure_size=fig_size, window_level_list=wl_list)
    else:
      gui.MultiImageDisplay(image_list=images, figure_size=fig_size, window_level_list=wl_list)
    
disp_images(data, fig_size=(6,2))

def threshold_based_crop(image):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.                                 
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    

modified_data = [threshold_based_crop(img) for img in data]

disp_images(modified_data, fig_size=(6,2))
data = modified_data

# Augmentation using spatial transformations
# utility methods 
def parameter_space_regular_grid_sampling(*transformation_parameters):
    '''
    Create a list representing a regular sampling of the parameter space.     
    Args:
        *transformation_paramters : two or more numpy ndarrays representing parameter values. The order 
                                    of the arrays should match the ordering of the SimpleITK transformation 
                                    parameterization (e.g. Similarity2DTransform: scaling, rotation, tx, ty)
    Return:
        List of lists representing the regular grid sampling.
        
    Examples:
        #parameterization for 2D translation transform (tx,ty): [[1.0,1.0], [1.5,1.0], [2.0,1.0]]
        >>>> parameter_space_regular_grid_sampling(np.linspace(1.0,2.0,3), np.linspace(1.0,1.0,1))        
    '''
    return [[np.asscalar(p) for p in parameter_values] 
            for parameter_values in np.nditer(np.meshgrid(*transformation_parameters))]

def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an 
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0],parameter_values[1], parameter_values[2])) + 
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]
    

def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     

    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx

    # Compute quaternion: 
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom;
        qv[1] = (r[0,2] - r[2,0])/denom;
        qv[2] = (r[1,0] - r[0,1])/denom;
    return qv

# Create reference domain
dimension = data[0].GetDimension()

# Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
reference_physical_size = np.zeros(dimension)
for img in data:
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

# Create the reference image with a zero origin, identity direction cosine matrix and dimension     
reference_origin = np.zeros(dimension)
reference_direction = np.identity(dimension).flatten()

# Select arbitrary number of pixels per dimension, smallest size that yields desired results 
# or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will 
# often result in non-isotropic pixel spacing.
reference_size = [128]*dimension 
reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

# Another possibility is that you want isotropic pixels, then you can specify the image size for one of
# the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
# spacing set accordingly. 
# Uncomment the following lines to use this strategy.
#reference_size_x = 128
#reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
#reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

reference_image = sitk.Image(reference_size, data[0].GetPixelIDValue())
reference_image.SetOrigin(reference_origin)
reference_image.SetSpacing(reference_spacing)
reference_image.SetDirection(reference_direction)

# Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as 
# this takes into account size, spacing and direction cosines. For the vast majority of images the direction 
# cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the 
# spacing will not yield the correct coordinates resulting in a long debugging session. 
reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

# data generation
def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters,
                    output_prefix, output_suffix,
                    interpolator = sitk.sitkLinear, default_intensity_value = 0.0):
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system 
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''
    all_images = [] 
    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)        
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                  interpolator, default_intensity_value)
        sitk.WriteImage(aug_image, output_prefix + '_' + 
                        '_'.join(str(param) for param in current_parameters) +'_.' + output_suffix)
         
        all_images.append(aug_image) 
    return all_images 

aug_transform = sitk.Similarity2DTransform() if dimension==2 else sitk.Similarity3DTransform()

all_images = []

for index,img in enumerate(data):
    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # Set the augmenting transform's center so that rotation is around the image center.
    aug_transform.SetCenter(reference_center)
    
    if dimension == 2:
        # The parameters are scale (+-10%), rotation angle (+-10 degrees), x translation, y translation
        transformation_parameters_list = parameter_space_regular_grid_sampling(np.linspace(0.9,1.1,3),
                                                                               np.linspace(-np.pi/18.0,np.pi/18.0,3),
                                                                               np.linspace(-10,10,3),
                                                                               np.linspace(-10,10,3))
    else:    
        transformation_parameters_list = similarity3D_parameter_space_regular_sampling(np.linspace(-np.pi/18.0,np.pi/18.0,3),
                                                                                       np.linspace(-np.pi/18.0,np.pi/18.0,3),
                                                                                       np.linspace(-np.pi/18.0,np.pi/18.0,3),
                                                                                       np.linspace(-10,10,3),
                                                                                       np.linspace(-10,10,3),
                                                                                       np.linspace(-10,10,3),
                                                                                       np.linspace(0.9,1.1,3))
    generated_images = augment_images_spatial(img, reference_image, centered_transform, 
                                       aug_transform, transformation_parameters_list, 
                                       os.path.join(OUTPUT_DIR, 'spatial_aug'+str(index)), 'mha')
    
    if dimension==2: # in 2D we join all of the images into a 3D volume which we use for display.
        all_images.append(sitk.JoinSeries(generated_images))
# If working in 2D, display the resulting set of images.    
if dimension==2:
    gui.MultiImageDisplay(image_list=all_images, shared_slider=True, figure_size=(6,2))

# sitk work
flipped_images = []
for index,img in enumerate(data):
    # Compute the transformation which maps between the reference and current image (same as done above).
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    
    flipped_transform = sitk.AffineTransform(dimension)    
    flipped_transform.SetCenter(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    if dimension==2: # matrices in SimpleITK specified in row major order
        flipped_transform.SetMatrix([1,0,0,-1])
    else:
        flipped_transform.SetMatrix([1,0,0,0,-1,0,0,0,1])
    centered_transform.AddTransform(flipped_transform)
    
    # Resample onto the reference image 
    flipped_images.append(sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0))
# Uncomment the following line to display the images (we don't want to time this)
#disp_images(flipped_images, fig_size=(6,2))

# Approach 2, flipping after resampling

flipped_images = []
for index,img in enumerate(data):
    # Compute the transformation which maps between the reference and current image (same as done above).
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    # Resample onto the reference image 
    resampled_img = sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    # We flip on the y axis (x, z are done similarly)
    if dimension==2:
        flipped_images.append(resampled_img[:,::-1])
    else:
        flipped_images.append(resampled_img[:,::-1,:])
# Uncomment the following line to display the images (we don't want to time this)        
#disp_images(flipped_images, fig_size=(6,2))

# Radial Distortion
def radial_distort(image, k1, k2, k3, distortion_center=None):
    c = distortion_center
    if not c: # The default distortion center coincides with the image center
        c = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    
    # Compute the vector image (p_d - p_c) 
    delta_image = sitk.PhysicalPointSource( sitk.sitkVectorFloat64, image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())
    delta_image_list = [sitk.VectorIndexSelectionCast(delta_image,i) - c[i] for i in range(len(c))]
    
    # Compute the radial distortion expression
    r2_image = sitk.NaryAdd([img**2 for img in delta_image_list])
    r4_image = r2_image**2
    r6_image = r2_image*r4_image
    disp_image = k1*r2_image + k2*r4_image + k3*r6_image
    displacement_image = sitk.Compose([disp_image*img for img in delta_image_list])
    
    displacement_field_transform = sitk.DisplacementFieldTransform(displacement_image)
    return sitk.Resample(image, image, displacement_field_transform)

k1 = 0.00001
k2 = 0.0000000000001
k3 = 0.0000000000001
original_image = data[0]
distorted_image = radial_distort(original_image, k1, k2, k3)
# Use a grid image to highlight the distortion.
grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16, size=original_image.GetSize(), 
                             sigma=[0.1]*dimension, gridSpacing=[20.0]*dimension)
grid_image.CopyInformation(original_image)
distorted_grid = radial_distort(grid_image, k1, k2, k3)
disp_images([original_image, distorted_image, distorted_grid], fig_size=(6,2))

# Augmentation using intensity modifications
def augment_images_intensity(image_list, output_prefix, output_suffix):
    '''
    Generate intensity modified images from the originals.
    Args:
        image_list (iterable containing SimpleITK images): The images which we whose intensities we modify.
        output_prefix (string): output file name prefix (file name: output_prefixi_FilterName.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefixi_FilterName.output_suffix).
    '''

    # Create a list of intensity modifying filters, which we apply to the given images
    filter_list = []
    
    # Smoothing filters
    
    filter_list.append(sitk.SmoothingRecursiveGaussianImageFilter())
    filter_list[-1].SetSigma(2.0)
    
    filter_list.append(sitk.DiscreteGaussianImageFilter())
    filter_list[-1].SetVariance(4.0)
    
    filter_list.append(sitk.BilateralImageFilter())
    filter_list[-1].SetDomainSigma(4.0)
    filter_list[-1].SetRangeSigma(8.0)
    
    filter_list.append(sitk.MedianImageFilter())
    filter_list[-1].SetRadius(8)
    
    # Noise filters using default settings
    
    # Filter control via SetMean, SetStandardDeviation.
    filter_list.append(sitk.AdditiveGaussianNoiseImageFilter())

    # Filter control via SetProbability
    filter_list.append(sitk.SaltAndPepperNoiseImageFilter())
    
    # Filter control via SetScale
    filter_list.append(sitk.ShotNoiseImageFilter())
    
    # Filter control via SetStandardDeviation
    filter_list.append(sitk.SpeckleNoiseImageFilter())

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(1.0)
    filter_list[-1].SetBeta(0.0)

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(0.0)
    filter_list[-1].SetBeta(1.0)
    
    aug_image_lists = [] # Used only for display purposes in this notebook.
    for i,img in enumerate(image_list):
        aug_image_lists.append([f.Execute(img) for f in filter_list])            
        for aug_image,f in zip(aug_image_lists[-1], filter_list):
            sitk.WriteImage(aug_image, output_prefix + str(i) + '_' +
                            f.GetName() + '.' + output_suffix)
    return aug_image_lists

intensity_augmented_images = augment_images_intensity(data, os.path.join(OUTPUT_DIR, 'intensity_aug'), 'mha')

          # in 2D we join all of the images into a 3D volume which we use for display.
if dimension==2:    
    def list2_float_volume(image_list) :
        return sitk.JoinSeries([sitk.Cast(img, sitk.sitkFloat32) for img in image_list])
        
    all_images = [list2_float_volume(imgs) for imgs in intensity_augmented_images]
    
    # Compute reasonable window-level values for display (just use the range of intensity values
    # from the original data).
    original_window_level = []
    statistics_image_filter = sitk.StatisticsImageFilter()
    for img in data:
        statistics_image_filter.Execute(img)
        max_intensity = statistics_image_filter.GetMaximum()
        min_intensity = statistics_image_filter.GetMinimum()
        original_window_level.append((max_intensity-min_intensity, (max_intensity+min_intensity)/2.0))
    gui.MultiImageDisplay(image_list=all_images, shared_slider=True, figure_size=(6,2), window_level_list=original_window_level)

def mult_and_add_intensity_fields(original_image):
    '''
    Modify the intensities using multiplicative and additive Gaussian bias fields.
    '''
    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is half the image's physical size and mean is the center of the image. 
    g_mult = sitk.GaussianSource(original_image.GetPixelIDValue(),
                             original_image.GetSize(),
                             [(sz-1)*spc/2.0 for sz, spc in zip(original_image.GetSize(), original_image.GetSpacing())],
                             original_image.TransformContinuousIndexToPhysicalPoint(np.array(original_image.GetSize())/2.0),
                             255,
                             original_image.GetOrigin(),
                             original_image.GetSpacing(),
                             original_image.GetDirection())

    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is 1/8 the image's physical size and mean is at 1/16 of the size 
    g_add = sitk.GaussianSource(original_image.GetPixelIDValue(),
                             original_image.GetSize(),
               [(sz-1)*spc/8.0 for sz, spc in zip(original_image.GetSize(), original_image.GetSpacing())],
               original_image.TransformContinuousIndexToPhysicalPoint(np.array(original_image.GetSize())/16.0),
               255,
               original_image.GetOrigin(),
               original_image.GetSpacing(),
               original_image.GetDirection())
    
    return g_mult*original_image+g_add

disp_images([mult_and_add_intensity_fields(img) for img in data], fig_size=(6,2))