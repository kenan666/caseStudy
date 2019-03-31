'''
Overview
SimpleITK has the ability to add commands or callbacks as observers of events that may 
occur during data processing. This feature can be used to add progress reporting to a 
console, to monitor the process of optimization, to abort a process, or to improve the 
integration of SimpleITK into Graphical User Interface event queues.

Events
Events are a simple enumerated type in SimpleITK, represented by the EventEnum type. 
More information about each event type can be found in the documentation for the enum. 
All SimpleITK filters, including the reading and writing ones, are derived from the 
ProcessObject class which has support for events. SimpleITK utilizes the native ITK 
event system but has simpler events and methods to add an observer or commands. 
The goal is to provide a simpler interface more suitable for scripting languages.

Commands
The command design pattern is used to allow user code to be executed when an event 
occurs. It is encapsulated in the Command class. The Command class provides a virtual 
Execute method to be overridden in derived classes. Additionally, SimpleITK provides 
internal reference tracking between the ProcessObject and the Command. This reference
 tracking allows an object to be created on the stack or dynamically allocated, 
 without additional burden.
'''

'''
Command Directors for Wrapped Languages
SimpleITK uses SWIG’s director feature to enable wrapped languages to derive classes 
from the Command class. Thus a user may override the Command class’s Execute method 
for custom call-backs. The following languages support deriving classes from the Command class:
'''
import SimpleITK as sitk

class MyCommand(sitk.Command):
    def __init__(self,po):
        #  required
        super(MyCommand,self).__init__()
        self.processObject = po

    def Execute(self):
        print("{0} Progress: {1:1.2f}".format(self.processObject.GetName(),self.processObject.GetProgress()))

'''
Command Functions and Lambdas for Wrapped Languages
Not all scripting languages are naturally object oriented, and it is often easier to 
simply define a callback inline with a lambda function. The following language supports 
inline function definitions for functions for the ProcessObject::AddCommand method:
'''
gaussian.AddCommand(sitk.sitkStartEvent,lambda:print("startEvent"))
gaussian.AddCommand(sitk.sitkEndEvent,lambda:print('endEvent'))

# ---------------------------------------------------------------------------
import SimpleITK as sitk
import sys,os

if len ( sys.argv ) < 4:
    print( "Usage: "+sys.argv[0]+ " <input> <variance> <output>" )
    sys.exit ( 1 )

##! [python director command]

class MyCommand(sitk.Command):
    def __init__(self, po):
        # required
        super(MyCommand,self).__init__()
        self.processObject = po

    def Execute(self):
        print("{0} Progress: {1:1.2f}".format(self.processObject.GetName(),self.processObject.GetProgress()))

##! [python director command]

reader = sitk.ImageFileReader()
reader.SetFileName(sys.argv[1])
image = reader.Execute()

pixelID = image.GetPixelID()

gaussian = sitk.DiscreteGaussianImageFilter()
gaussian.SetVariance(float(sys.argv[2]))

##! [python lambda command]

gaussian.AddCommand(sitk.sitkStartEvent, lambda: print("StartEvent"))
gaussian.AddCommand(sitk.sitkEndEvent, lambda: print("EndEvent"))

##! [python lambda command]

cmd = MyCommand(gaussian)
gaussian.AddCommand(sitk.sitkProgressEvent,cmd)

image = gaussian.Execute(image)

caster = sitk.CastImageFilter()
caster.SetOutputPixelType(pixelID)
image = caster.Execute(image)

writer = sitk.ImageFileWriter()
writer.SetFileName(sys.argv[3])
writer.Execute(image)

if (not "SITK_NOSHOW" in os.environ):
    sitk.Show(image,"simple Gaussian")
    