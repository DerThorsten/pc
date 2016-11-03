import numpy
import vigra
import h5py
import fastfilters

f = "/home/tbeier/src/pixel_classification/data/hhess_supersmall/raw_predictions.h5"

dset = h5py.File(f)['data']

membraneP = dset[:,:,0:40,1]

binary = numpy.zeros(membraneP.shape,dtype='uint8')
binary[membraneP>0.5] = 1










f = BinaryMorphologyFeatures()

with vigra.Timer("f"):
    features = f(binary)


for i in range(features.shape[3]):
    
    fC = features[:,:,:,i]
    
    vigra.imshow(fC[:,:,10])
    vigra.show()
