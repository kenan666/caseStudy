#  Registration Errors
import SimpleITK as sitk
import numpy as np
import copy

from gui import PairedPointDataManipulation, display_errors
import matplotlib.pyplot as plt
from registration_utilities import registration_errors

manipulation_interface = PairedPointDataManipulation(sitk.Euler2DTransform())

# Sensitivity to outliers
ideal_fixed_fiducials = [[23.768817532447077, 60.082971482049849], [29.736559467930949, 68.740980140058511],
                   [37.639785274382561, 68.524529923608299], [41.994623984059984, 59.000720399798773]]
ideal_fixed_targets = [[32.317204629221266, 60.732322131400501], [29.413978822769653, 56.403317802396167]]
ideal_moving_fiducials = [[76.77857043206542, 30.557710579173616], [86.1401622129338, 25.76859196933914],
                    [86.95501792478755, 17.904506579872375], [78.07960498849866, 12.346214284259808]]
ideal_moving_targets = [[78.53588814928511, 22.166738486331596], [73.86559697098288, 24.481339720595585]]

# Registration with perfect data (no noise or outliers) 
fixed_fiducials = copy.deepcopy(ideal_fixed_fiducials)
fixed_targets = copy.deepcopy(ideal_fixed_targets)
moving_fiducials = copy.deepcopy(ideal_moving_fiducials)
moving_targets = copy.deepcopy(ideal_moving_targets)

# Flatten the point lists, SimpleITK expects a single list/tuple with coordinates (x1,y1,...xn,yn)
fixed_fiducials_flat = [c for p in fixed_fiducials for c in p]
moving_fiducials_flat = [c for p in moving_fiducials for c in p]

transform = sitk.LandmarkBasedTransformInitializer(sitk.Euler2DTransform(), fixed_fiducials_flat, moving_fiducials_flat)

FRE_information = registration_errors(transform, fixed_fiducials, moving_fiducials)
TRE_information = registration_errors(transform, fixed_targets, moving_targets)
FLE_values = [0.0]*len(moving_fiducials)
FLE_information =  (np.mean(FLE_values), np.std(FLE_values), np.min(FLE_values), np.max(FLE_values), FLE_values) 
display_errors(fixed_fiducials, fixed_targets, FLE_information, FRE_information, TRE_information, title="Ideal Input")

# Change fourth fiducial to an outlier and register
outlier_fiducial = [88.07960498849866, 22.34621428425981]
FLE_values[3] = np.sqrt((outlier_fiducial[0] - moving_fiducials[3][0])**2 + 
                        (outlier_fiducial[1] - moving_fiducials[3][1])**2)
moving_fiducials[3][0] = 88.07960498849866
moving_fiducials[3][1] = 22.34621428425981

moving_fiducials_flat = [c for p in moving_fiducials for c in p]

transform = sitk.LandmarkBasedTransformInitializer(sitk.Euler2DTransform(), fixed_fiducials_flat, moving_fiducials_flat)

FRE_information = registration_errors(transform, fixed_fiducials, moving_fiducials)
TRE_information = registration_errors(transform, fixed_targets, moving_targets)
FLE_information =  (np.mean(FLE_values), np.std(FLE_values), np.min(FLE_values), np.max(FLE_values), FLE_values) 
display_errors(fixed_fiducials, fixed_targets, FLE_information, FRE_information, TRE_information, title="Single Outlier")

#  FRE is not a surrogate for TRE
# Registration with same bias added to all points
fixed_fiducials = copy.deepcopy(ideal_fixed_fiducials)
fixed_targets = copy.deepcopy(ideal_fixed_targets)
moving_fiducials = copy.deepcopy(ideal_moving_fiducials)
bias_vector = [4.5, 4.5]
bias_fle = np.sqrt(bias_vector[0]**2 + bias_vector[1]**2)
for fiducial in moving_fiducials:
    fiducial[0] +=bias_vector[0]
    fiducial[1] +=bias_vector[1]
FLE_values = [bias_fle]*len(moving_fiducials)
moving_targets = copy.deepcopy(ideal_moving_targets)

# Flatten the point lists, SimpleITK expects a single list/tuple with coordinates (x1,y1,...xn,yn)
fixed_fiducials_flat = [c for p in fixed_fiducials for c in p]
moving_fiducials_flat = [c for p in moving_fiducials for c in p]

transform = sitk.LandmarkBasedTransformInitializer(sitk.Euler2DTransform(), fixed_fiducials_flat, moving_fiducials_flat)

FRE_information = registration_errors(transform, fixed_fiducials, moving_fiducials)
TRE_information = registration_errors(transform, fixed_targets, moving_targets)
FLE_information =  (np.mean(FLE_values), np.std(FLE_values), np.min(FLE_values), np.max(FLE_values), FLE_values) 
display_errors(fixed_fiducials, fixed_targets, FLE_information, FRE_information, TRE_information, title="FRE<TRE")

# Registration with bias in one direction for half the fiducials and in the opposite direction for the other half
moving_fiducials = copy.deepcopy(ideal_moving_fiducials)
pol = 1
for fiducial in moving_fiducials:
    fiducial[0] +=bias_vector[0]*pol
    fiducial[1] +=bias_vector[1]*pol
    pol*=-1.0
FLE_values = [bias_fle]*len(moving_fiducials)
moving_targets = copy.deepcopy(ideal_moving_targets)

# Flatten the point lists, SimpleITK expects a single list/tuple with coordinates (x1,y1,...xn,yn)
fixed_fiducials_flat = [c for p in fixed_fiducials for c in p]
moving_fiducials_flat = [c for p in moving_fiducials for c in p]

transform = sitk.LandmarkBasedTransformInitializer(sitk.Euler2DTransform(), fixed_fiducials_flat, moving_fiducials_flat)

FRE_information = registration_errors(transform, fixed_fiducials, moving_fiducials)
TRE_information = registration_errors(transform, fixed_targets, moving_targets)
FLE_information =  (np.mean(FLE_values), np.std(FLE_values), np.min(FLE_values), np.max(FLE_values), FLE_values) 
display_errors(fixed_fiducials, fixed_targets, FLE_information, FRE_information, TRE_information, title="FRE>TRE")

#  Fiducial Configuration
fiducials = [[31.026882048576109, 65.696247315510021], [34.252688500189009, 70.674602293864993], 
             [41.349462693737394, 71.756853376116084], [47.801075596963202, 68.510100129362826], 
             [52.47849495180192, 63.315294934557635]]
targets = [[38.123656242124497, 64.397546016808718], [43.768817532447073, 63.748195367458059], 
           [26.833333661479333, 8.7698403891030861], [33.768817532447073, 8.120489739752438]]
manipulation_interface = PairedPointDataManipulation(sitk.Euler2DTransform())
manipulation_interface.set_fiducials(fiducials)
manipulation_interface.set_targets(targets)

#  FRE-TRE, and Occam's razor
fiducials = [[31.026882048576109, 65.696247315510021], 
             [41.349462693737394, 71.756853376116084], 
             [52.47849495180192, 63.315294934557635]]

targets = [[38.123656242124497, 64.397546016808718], [43.768817532447073, 63.748195367458059]]
manipulation_interface = PairedPointDataManipulation(sitk.Euler2DTransform())
#manipulation_interface = PairedPointDataManipulation(sitk.AffineTransform(2))
manipulation_interface.set_fiducials(fiducials)
manipulation_interface.set_targets(targets)