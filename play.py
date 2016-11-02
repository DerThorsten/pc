import numpy
import vigra
import h5py


f = "/home/tbeier/src/pixel_classification/data/hhess_supersmall/raw_predictions.h5"

dset = h5py.File(f)['data']

membraneP = dset[:,:,0:40,0]

binary = numpy.zeros(membraneP.shape,dtype='uint8')
binary[membraneP>0.5] = 1

res = vigra.filters.multiBinaryDilation(binary, 10).astype('uint32')

res = vigra.filters.multiBinaryErosion(binary, 4)
res = vigra.filters.multiBinaryDilation(res, 4)

res = vigra.filters.gaussianSmoothing(res.astype('float32'),1.0)


vigra.imshow(res[:,:,10])
vigra.show()