from __future__ import print_function
import importlib
from distutils.version import LooseVersion

# check that all packages are installed (see requirements.txt file)
required_packages = {'jupyter', 
                     'numpy',
                     'matplotlib',
                     'ipywidgets',
                     'scipy',
                     'pandas',
                     'SimpleITK'
                    }

problem_packages = list()

for package in required_packages:
    try:
        p = importlib.import_module(package)
    except ImportError:
        problem_packages.append(package)

if len(problem_packages) is 0 :
    print('ALL is well')
else:
    print('The following packages are required but not installed: ' + ', '.join(problem_packages))

import SimpleITK as sitk

from downloaddata import fetch_data,fetch_data_all

from ipywidgets import interact

print(sitk.Version)

# Uncomment the line below to change the default external viewer to your viewer of choice and test that it works.
#%env SITK_SHOW_COMMAND /Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP 

# Retrieve an image from the network, read it and display using the external viewer. 
# The show method will also set the display window's title and by setting debugOn to True, 
# will also print information with respect to the command it is attempting to invoke.
# NOTE: The debug information is printed to the terminal from which you launched the notebook
#       server.
sitk.Show(sitk.ReadImage(fetch_data("SimpleITK.jpg")), "SimpleITK Logo", debugOn=True)

interact(lambda x : x, x = (0,10))

fetch_data_all(os.path.join('..','Data'),os.path.join('..','Data','mainfest.json'))