import fastfilters
import numpy

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

class RawFeatureExtractor(object):

    def __init__(self):
        self.shape = None
        self.halo = (50,50,50)


    def __call__(self, h5Dset, blockBegin,blockEnd, whereLabels = None):
        
        gBegin, gEnd ,lBegin, lEnd = addHalo(self.shape, blockBegin, blockEnd, self.halo)


        blockShape1 = tuple([e-b for e,b in zip(lEnd, lBegin)])
        blockShape2 = tuple([e-b for e,b in zip(blockEnd, blockBegin)])
        


        # fetch raw data
        data = h5Dset[gBegin[0]:gEnd[0], gBegin[1]:gEnd[1], gBegin[2]:gEnd[2]].squeeze() 



        allFeat = []
        sigmas = (1.0, 2.0, 4.0, 8.0)

        if whereLabels is not None:

            whereLabelsL = whereLabels.copy()
            whereLabelsL[0,:] += lBegin[0]
            whereLabelsL[1,:] += lBegin[1]
            whereLabelsL[2,:] += lBegin[2]
            wx,wy,wz = whereLabelsL[0,:], whereLabelsL[1,:], whereLabelsL[2,:]

            for sigma in sigmas:
                allFeat.append(fastfilters.gaussianSmoothing(data, sigma)[wx,wy,wz,None])
                allFeat.append(fastfilters.laplacianOfGaussian(data, sigma)[wx,wy,wz,None])
                allFeat.append(fastfilters.gaussianGradientMagnitude(data, sigma)[wx,wy,wz,None])
                allFeat.append(fastfilters.gaussianGradientMagnitude(data, sigma)[wx,wy,wz,None])
                allFeat.append(fastfilters.hessianOfGaussianEigenvalues(data, sigma)[wx,wy,wz,:])
                allFeat.append(fastfilters.structureTensorEigenvalues(data, sigma, sigma*2.0)[wx,wy,wz,:])
            
            allFeat = numpy.concatenate(allFeat,axis=1)
        else:
            for sigma in sigmas:
                allFeat.append(fastfilters.gaussianSmoothing(data, sigma)[:,:,:,None])
                allFeat.append(fastfilters.laplacianOfGaussian(data, sigma)[:,:,:,None])
                allFeat.append(fastfilters.gaussianGradientMagnitude(data, sigma)[:,:,:,None])
                allFeat.append(fastfilters.gaussianGradientMagnitude(data, sigma)[:,:,:,None])
                allFeat.append(fastfilters.hessianOfGaussianEigenvalues(data, sigma)[:,:,:,:])
                allFeat.append(fastfilters.structureTensorEigenvalues(data, sigma, sigma*2.0)[:,:,:,:])
            
            allFeat = numpy.concatenate(allFeat,axis=3)


            allFeat = allFeat[
                lBegin[0]:lEnd[0], 
                lBegin[1]:lEnd[1], 
                lBegin[2]:lEnd[2],
                :
            ]

        
        return allFeat




class PmapFeatures(object):
    