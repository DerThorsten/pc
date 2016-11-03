import fastfilters
import numpy
import math
def addHalo(shape, blockBegin, blockEnd, halo):

    withHaloBlockBegin = (
        max(blockBegin[0] - halo[0],0)   , 
        max(blockBegin[1] - halo[1],0)   ,
        max(blockBegin[2] - halo[2],0)
    )

    withHaloBlockEnd = (
        min(blockEnd[0] + halo[0],shape[0])   , 
        min(blockEnd[1] + halo[1],shape[1])   ,
        min(blockEnd[2] + halo[2],shape[2])
    )


    inBlockBegin = (
        blockBegin[0] -  withHaloBlockBegin[0],
        blockBegin[1] -  withHaloBlockBegin[1],
        blockBegin[2] -  withHaloBlockBegin[2]
    )

    inBlockEnd = (
        inBlockBegin[0] +  (blockEnd[0] - blockBegin[0]),
        inBlockBegin[1] +  (blockEnd[1] - blockBegin[1]),
        inBlockBegin[2] +  (blockEnd[2] - blockBegin[2])
    )

    return  withHaloBlockBegin, withHaloBlockEnd, inBlockBegin, inBlockEnd



def getScale(target, presmoothed):
    return math.sqrt(target**2 - presmoothed**2)





class FeatureExtractorBase(object):

    def halo(self):
        raise NotImplementedError("halo is not implemented")



class RawFeatureExtractor(FeatureExtractorBase):

    def __init__(self):
        self.shape = None
        self.__halo = (50,50,50)

    def halo(self):
        return self.__halo

    def __call__(self, h5Dset, blockBegin,blockEnd, whereLabels = None):
        
        gBegin, gEnd ,lBegin, lEnd = addHalo(self.shape, blockBegin, blockEnd, self.halo())


        blockShape1 = tuple([e-b for e,b in zip(lEnd, lBegin)])
        blockShape2 = tuple([e-b for e,b in zip(blockEnd, blockBegin)])
        


        # fetch raw data
        data = h5Dset[gBegin[0]:gEnd[0], gBegin[1]:gEnd[1], gBegin[2]:gEnd[2]].squeeze() 



        allFeat = []
        sigmas = (1.0, 2.0, 4.0, 8.0)


        # pre-smoothed
        sigmaPre = sigmas[0]/2.0
        preS = fastfilters.gaussianSmoothing(data, sigmaPre)

        for sigma in sigmas:

            neededScale = getScale(target=sigma, presmoothed=sigmaPre)

            preS = fastfilters.gaussianSmoothing(preS, neededScale)
            sigmaPre = sigma

            allFeat.append(preS[:,:,:,None])
            allFeat.append(fastfilters.laplacianOfGaussian(data, neededScale)[:,:,:,None])
            allFeat.append(fastfilters.gaussianGradientMagnitude(data, neededScale)[:,:,:,None])
            allFeat.append(fastfilters.gaussianGradientMagnitude(data, neededScale)[:,:,:,None])
            allFeat.append(fastfilters.hessianOfGaussianEigenvalues(data, neededScale)[:,:,:,:])
            allFeat.append(fastfilters.structureTensorEigenvalues(data, neededScale, sigma*2.0)[:,:,:,:])
        
        
        allFeat = numpy.concatenate(allFeat,axis=3)

        if whereLabels is not None:

            return allFeat[
                whereLabels[0,:] + lBegin[0],
                whereLabels[1,:] + lBegin[1],
                whereLabels[2,:] + lBegin[2],
                :
            ]
        else:
            return allFeat[
                lBegin[0]:lEnd[0], 
                lBegin[1]:lEnd[1], 
                lBegin[2]:lEnd[2],
                :
            ]


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





if __name__ == "__main__":
    pass