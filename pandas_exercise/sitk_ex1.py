import SimpleITK as sitk

# create an image

pixelType = sitk.sitkUInt8
imageSize = [128,128]
image = sitk.Image(imageSize,pixelType)

#  create a face image

faceSize = [64,64]
faceCenter = [64,64]
face = sitk.GaussianSource(pixelType,imageSize,faceSize,faceCenter)

# create eye images

eyeSize = [5,5]
eye1Center = [48,48]
eye2Center = [80,48]
eye1 = sitk.GaussianSource(pixelType,imageSize,eyeSize,eye1Center,150)
eye2 = sitk.GaussianSource(pixelType,imageSize,eyeSize,eye2Center,150)

# Apply the eyes to face

face = face - eye1 - eye2
face = sitk.BinaryThreshold(face,200,255,255)

# create the mouth

mouthRadii = [30,20]
mouthCenter = [64,76]
mouth       = 255 - sitk.BinaryThreshold( sitk.GaussianSource(pixelType, imageSize, mouthRadii, mouthCenter),
                                          200, 255, 255 )
# paste the mouth into face

mouthSize = [64,18]
mouthLoc = [32,76]
face = sitk.Paste(face,mouth,mouthSize,mouthLoc,mouthLoc)

#  apply the face to the original image 

image = image + face

# display the results
sitk.Show(image, title = 'hello  world!', debugOn = True)