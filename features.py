import fastfilters
import numpy
import math
import vigra
from tools import addHalo, getSlicing




def getScale(target, presmoothed):
    return math.sqrt(target**2 - presmoothed**2)


class FeatureExtractorBase(object):

    def halo(self):
        raise NotImplementedError("halo is not implemented")


def extractChannels(data,usedChannels=(0,)):
    
    if data.ndim == 3:
        d =data[:,:,:,None]
    elif data.ndim == 4:
        d = data

    d = d[:,:,:,usedChannels]

    if d.ndim == 3:
        d =d[:,:,:,None]
    elif d.ndim == 4:
        d = d

    return d




class ConvolutionFeatures(FeatureExtractorBase):

    def __init__(self, 
        sigmas = (1.0, 2.0, 4.0, 8.0),
        usedChannels= (0,)
    ):
        self.shape = None
        assert sorted(sigmas) == sigmas
        self.sigmas  = sigmas
        self.usedChannels = usedChannels
        maxSigma = max(sigmas)
        maxOrder = 2
        r = int(round(3.0*maxSigma* + 0.5*2))

        # r of structure tensor
        rD = int(round(3.0*maxSigma*0.3 + 0.5))
        rG = int(round(3.0*maxSigma*0.7 ))
        rSt = rD+rG

        self.__halo = (max(r,rSt),)*3

    def halo(self):
        return self.__halo

    def numberOfFeatures(self):
        nEvFeat = 2
        nScalarFeat =  3
        nChannels = len(self.usedChannels)
        #print("nChannels",nChannels)
        #print("nSigmas",len(self.sigmas))
        return  len(self.sigmas) * (nEvFeat*3 + nScalarFeat) * nChannels


    def __call__(self, dataIn, slicing, featureArray):
        
        fIndex = 0
        dataIn = numpy.require(dataIn,'float32').squeeze()

        dataWithChannel = extractChannels(dataIn, self.usedChannels)

        slicingEv = slicing + [slice(0,3)]

        for c in range(dataWithChannel.shape[3]):

            data = dataWithChannel[:,:,:,c]

            # pre-smoothed
            sigmaPre = self.sigmas[0]/2.0
            preS = fastfilters.gaussianSmoothing(data, sigmaPre)

            for sigma in self.sigmas:

                neededScale = getScale(target=sigma, presmoothed=sigmaPre)
                preS = fastfilters.gaussianSmoothing(preS, neededScale)
                sigmaPre = sigma


                featureArray[:,:,:,fIndex] = preS[slicing]
                fIndex += 1

                featureArray[:,:,:,fIndex] = fastfilters.laplacianOfGaussian(preS, neededScale)[slicing]
                fIndex += 1

                featureArray[:,:,:,fIndex] = fastfilters.gaussianGradientMagnitude(preS, neededScale)[slicing]
                fIndex += 1


                featureArray[:,:,:,fIndex:fIndex+3] = fastfilters.hessianOfGaussianEigenvalues(preS, neededScale)[slicingEv]
                fIndex += 3

                
                #print("array shape",featureArray[:,:,:,fIndex:fIndex+3].shape)
                feat = fastfilters.structureTensorEigenvalues(preS, float(sigma)*0.3, float(sigma)*0.7)[slicingEv]
                #print("feat  shape",feat.shape)
                featureArray[:,:,:,fIndex:fIndex+3] = feat
                fIndex += 3

        assert fIndex == self.numberOfFeatures()




class BinaryMorphologyFeatures(FeatureExtractorBase):
    def __init__(self, 
        channel,
        thresholds=(0.5, ),  
        radii=(1,2,4,6,8),#2,4,8,16),
        useRadiiDilation=  (1,1,1,1,1),
        useRadiiErosion =  (1,1,0,0,0),
        useRadiiClosing =  (1,1,1,1,1),
        useRadiiOpening =  (1,1,0,0,0),
        postSmoothScale = None
    ):
        self.channel = channel
        self.thresholds = thresholds
        self.radii = radii
        self.__halo = max(self.radii)

        self.useRadiiDilation= useRadiiDilation
        self.useRadiiErosion= useRadiiErosion
        self.useRadiiClosing= useRadiiClosing
        self.useRadiiOpening= useRadiiOpening

        self.postSmoothScale = postSmoothScale
            
    def halo(self):
        return (self.__halo,)*3

    def numberOfFeatures(self):
        perRadius = sum(self.useRadiiDilation)
        perRadius += sum(self.useRadiiErosion)
        perRadius += sum(self.useRadiiClosing)
        perRadius += sum(self.useRadiiOpening)

        return  perRadius*len(self.thresholds)

    def __call__(self, dataIn, slicing, featureArray):

        if dataIn.ndim == 4:
            data = dataIn[:,:,:, self.channel]
        else:
            data = dataIn

        allFeat = []

        fIndex = 0 


        for t in self.thresholds:

            binary = numpy.zeros(data.squeeze().shape,dtype='uint8')
            binary[data > t] = 1


            dilation = None
            erosion  = None

            for ir, r in enumerate(self.radii):

                if self.useRadiiDilation[ir]:
                    dilation = vigra.filters.multiBinaryDilation(binary, r).squeeze()
                    featureArray[:,:,:, fIndex] = dilation[slicing]
                    fIndex += 1
                    

                if self.useRadiiErosion[ir]:
                    erosion  = vigra.filters.multiBinaryErosion(binary, r).squeeze()
                    featureArray[:,:,:, fIndex] = erosion[slicing]
                    fIndex += 1

                if self.useRadiiClosing[ir]:
                    if dilation is not None:
                        closing  = vigra.filters.multiBinaryErosion(dilation, r).squeeze()
                    else:
                        closing  = vigra.filters.multiBinaryClosing(binary, r).squeeze()
                    featureArray[:,:,:, fIndex] = closing[slicing]
                    fIndex += 1

                if self.useRadiiOpening[ir]:
                    if erosion is not None:
                        opening  = vigra.filters.multiBinaryDilation(erosion, r).squeeze()
                    else:
                        opening  = vigra.filters.multiBinaryOpening(binary, r).squeeze()

                    featureArray[:,:,:, fIndex] = opening[slicing]
                    fIndex += 1
        assert fIndex == self.numberOfFeatures()

        

# registered features
registerdFeatureOperators = {
    "ConvolutionFeatures" : ConvolutionFeatures,
    "BinaryMorphologyFeatures" : BinaryMorphologyFeatures
}




if __name__ == "__main__":
    import pylab
    import h5py

    f = "/home/tbeier/src/pc/data/hhess_supersmall/raw_predictions.h5"

    dset = h5py.File("/home/tbeier/src/pc/data/hhess_supersmall/raw_predictions.h5",'r')['data']
    data = dset[0:200,0:200,:,0].squeeze()



    fOp = BinaryMorphologyFeatures(channel=0,thresholds=[128.0])

    slicing = [slice(0,s) for s in data.shape]
    outShape = data.shape + (fOp.numberOfFeatures(),)

    fArray = numpy.zeros(outShape)

    fOp(data, slicing, fArray)

    for x in range(fArray.shape[3]):
        print(x)
        p = int(fArray.shape[2]/2)
        fImg = fArray[:,:,p,x]

        print(data.shape)
        rImg = data[:,:,p]

        f = pylab.figure()
        f.add_subplot(2, 1, 1)
        pylab.imshow(fImg,cmap='gray')
        f.add_subplot(2, 1, 2)
        pylab.imshow(rImg,cmap='gray')

        pylab.show()