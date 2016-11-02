import numpy
import vigra
import h5py
import fastfilters

f = "/home/tbeier/src/pixel_classification/data/hhess_supersmall/raw_predictions.h5"

dset = h5py.File(f)['data']

membraneP = dset[:,:,0:40,1]

binary = numpy.zeros(membraneP.shape,dtype='uint8')
binary[membraneP>0.5] = 1






class FeatureExtractorBase(object):

    def halo(self):
        raise NotImplementedError("halo is not implemented")





class BinaryMorphologyFeatures(FeatureExtractorBase):
    def __init__(self, thresholds=(0.5, ),  radii=(1,2,4,6,8),#2,4,8,16),
        useRadiiDilation=  (1,1,1,1,1),
        useRadiiErosion =  (1,1,0,0,0),
        useRadiiClosing =  (1,1,1,1,1),
        useRadiiOpening =  (1,1,0,0,0),
        postSmoothScale = None
    ):
        self.thresholds = thresholds
        self.radii = radii
        self.halo = max(self.radii)

        self.useRadiiDilation= useRadiiDilation
        self.useRadiiErosion= useRadiiErosion
        self.useRadiiClosing= useRadiiClosing
        self.useRadiiOpening= useRadiiOpening

        self.postSmoothScale = postSmoothScale
            
    def halo():
        return (self.halo,)*3

    def __call__(self, data):

        allFeat = []

        for t in self.thresholds:

            binary = numpy.zeros(data.squeeze().shape,dtype='uint8')
            binary[data > t] = 1


            dilation = None
            erosion  = None

            for ir, r in enumerate(self.radii):

                if self.useRadiiDilation[ir]:
                    dilation = vigra.filters.multiBinaryDilation(binary, r)
                    allFeat.append(dilation[:,:,:,None])

                if self.useRadiiErosion[ir]:
                    erosion  = vigra.filters.multiBinaryErosion(binary, r)
                    allFeat.append(erosion[:,:,:,None])

                if self.useRadiiClosing[ir]:
                    if dilation is not None:
                        closing  = vigra.filters.multiBinaryErosion(dilation, r)
                    else:
                        closing  = vigra.filters.multiBinaryClosing(binary, r)
                    allFeat.append(closing[:,:,:,None])

                if self.useRadiiOpening[ir]:
                    if erosion is not None:
                        opening  = vigra.filters.multiBinaryDilation(erosion, r)
                    else:
                        opening  = vigra.filters.multiBinaryOpening(binary, r)
                    allFeat.append(opening[:,:,:,None])

        
        if self.postSmoothScale is not None:
            for i,feat in enumerate(allFeat):
                feat = fastfilters.gaussianSmoothing(feat.squeeze().astype('float32'), self.postSmoothScale)
                allFeat[i] = feat[:,:,:,None]
                
        
        return numpy.concatenate(allFeat, axis=3)



f = BinaryMorphologyFeatures()

with vigra.Timer("f"):
    features = f(binary)


for i in range(features.shape[3]):
    
    fC = features[:,:,:,i]
    
    vigra.imshow(fC[:,:,10])
    vigra.show()
