# Improving Registration via Registration:Semiautomatic Landmark Localization
#  location 
# 
import numpy as np
import SimpleITK as sitk
import registration_utilities as ru
%run update_path_to_download_script
from downloaddata import fetch_data as fdata
import gui

# load data
fixed_image =  sitk.ReadImage(fdata("training_001_ct.mha"), sitk.sitkFloat32)
moving_image = sitk.ReadImage(fdata("training_001_mr_T1.mha"), sitk.sitkFloat32) 
fixed_fiducial_points, moving_fiducial_points = ru.load_RIRE_ground_truth(fdata("ct_T1.standard"))

# In the original data both images have the same orientation (patient in supine), the approach should also work when 
# images have different orientation. In the extreme they have a 180^o rotation between them.

rotate = True

if rotate:
    rotation_center = moving_image.TransformContinuousIndexToPhysicalPoint([(index-1)/2.0 for index in moving_image.GetSize()])    
    transform_moving = sitk.Euler3DTransform(rotation_center, 0, 0, np.pi, (0,0,0))
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(moving_image)
    resample.SetInterpolator(sitk.sitkLinear)    
    resample.SetTransform(transform_moving)
    moving_image = resample.Execute(moving_image)
    for i,p in enumerate(moving_fiducial_points):
        moving_fiducial_points[i] = transform_moving.TransformPoint(p)

        
# Compute the rigid transformation defined by the two point sets. Flatten the tuple lists 
# representing the points. The LandmarkBasedTransformInitializer expects the point coordinates 
# in one flat list [x1, y1, z1, x2, y2, z2...].
fixed_fiducial_points_flat = [c for p in fixed_fiducial_points for c in p]        
moving_fiducial_points_flat = [c for p in moving_fiducial_points for c in p]
reference_transform = sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), 
                                                             fixed_fiducial_points_flat, 
                                                             moving_fiducial_points_flat)

# Generate a reference dataset from the reference transformation 
# (corresponding points in the fixed and moving images).
fixed_points = ru.generate_random_pointset(image=fixed_image, num_points=100)
moving_points = [reference_transform.TransformPoint(p) for p in fixed_points]    

# Compute the TRE prior to registration.
pre_errors_mean, pre_errors_std, _, pre_errors_max, pre_errors = ru.registration_errors(sitk.Euler3DTransform(), fixed_points, moving_points, display_errors=True)
print('Before registration, errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(pre_errors_mean, pre_errors_std, pre_errors_max))

# Manual Landmark Localization
point_acquisition_interface = gui.RegistrationPointDataAquisition(fixed_image, moving_image, fixed_window_level=(215,50))

# Registration (manual landmark localization)
#fixed_image_points, moving_image_points = point_acquisition_interface.get_points()
fixed_image_points = [(156.48434676356158, 201.92274575468412, 68.0), 
                      (194.25413436597393, 98.55771047484492, 32.0),
                      (128.94523819661913, 96.18284152323203, 32.0)]
moving_image_points = [(141.46826904042848, 156.97653126727528, 48.0),
                       (113.70102381552435, 251.76553994455645, 8.0),
                       (180.69457220262115, 251.76553994455645, 8.0)]

fixed_image_points_flat = [c for p in fixed_image_points for c in p]        
moving_image_points_flat = [c for p in moving_image_points for c in p]
manual_localized_transformation = sitk.VersorRigid3DTransform(sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), 
                                                                         fixed_image_points_flat, 
                                                                         moving_image_points_flat))

manual_errors_mean, manual_errors_std, manual_errors_min, manual_errors_max,_ = \
    ru.registration_errors(manual_localized_transformation,
                           fixed_points, 
                           moving_points, 
                           display_errors=True)
print('After registration (manual point localization), errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(manual_errors_mean, manual_errors_std, manual_errors_max))

gui.RegistrationPointDataAquisition(fixed_image, moving_image, fixed_window_level=(215,50), known_transformation=manual_localized_transformation)

#  semiautomatic landmark localization
updated_moving_image_points = moving_image_points

# Registration (semiautomatic landmark localization)
updated_moving_image_points_flat = [c for p in updated_moving_image_points for c in p]        
semi_automatic_transform = sitk.VersorRigid3DTransform(sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), 
                                                                         fixed_image_points_flat, 
                                                                         updated_moving_image_points_flat))

semi_automatic_errors_mean, semi_automatic_errors_std, _, semi_automatic_errors_max,_ = ru.registration_errors(semi_automatic_transform,
                                                                                       fixed_points, 
                                                                                       moving_points, 
                                                                                       display_errors=True,
                                                                                       min_err=manual_errors_min,
                                                                                       max_err = manual_errors_max)
print('After registration (semiautomatic point localization), errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(semi_automatic_errors_mean, 
                                                                                                                                          semi_automatic_errors_std,
                                                                                                                                          semi_automatic_errors_max))

gui.RegistrationPointDataAquisition(fixed_image, moving_image, fixed_window_level=(215,50), known_transformation=semi_automatic_transform)

