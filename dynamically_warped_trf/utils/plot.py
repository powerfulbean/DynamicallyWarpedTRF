import torch
from matplotlib import pyplot as plt
from nntrf.models import TwoMixedTRF

class PlotInterm:
    
    def __init__(self,srate):
        self.srate = srate
    
    def plotAllTRFWeights(self,trfModel,title = ''):
        times = trfModel.lagTimes
        figures = []
        for i in range(trfModel.weights.shape[1]):
            fig2 = plt.figure()
            plt.plot(times,trfModel.weights[:,i,:].T)
            plt.title(title)
            figures.append(fig2)
        return figures

    def plotTransformTRF(self,nonLinTRF):
        figures = []
        fig = plt.figure()
        TRFs = nonLinTRF.lastState['TRFs'] # ((nLen, nWin, outDim))
        TRFs = TRFs.detach().cpu().numpy()
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for idx,TRF in enumerate(TRFs):
            if TRF.shape[1] == 128:
                tarTRF = TRF[:,18]
            else:
                tarTRF = TRF[:,0:1]
            plt.plot(nonLinTRF.lagTimes,tarTRF,color = cycle[idx % len(cycle)])
            # break
        figures.append(fig)
        plt.title(nonLinTRF.lastState['info'])

        times = nonLinTRF.lagTimes

        if getattr(nonLinTRF,'convKernels',None) is not None:
            if isinstance(nonLinTRF.convKernels,torch.nn.ParameterDict):
                for k,v in nonLinTRF.convKernels.items():
                    fig = plt.figure()
                    plt.plot(times,v.detach().cpu().T) #
                    plt.title(f'{k}_rawKernel')
                    figures.append(fig)
            else:
                fig = plt.figure()
                plt.plot(times,nonLinTRF.convKernels.detach().cpu().T) #
                plt.title(f'rawKernel')
                figures.append(fig)
        return figures


    def plotTransformTRFLinear(self,nonLinTRF):
        times = nonLinTRF.lagTimes
        twoStageFlag = False
        if getattr(nonLinTRF,'LinearKernels',None) is not None:
            oLinear = nonLinTRF.LinearKernels
            twoStageFlag = True
        else:
            oLinear = nonLinTRF.oTransform.oLinear
        figures = []
        if isinstance(oLinear,torch.nn.ModuleDict):
            for k in oLinear:
                if getattr(oLinear[k],'oLinear',None) is not None:
                    weight = oLinear[k].oLinear.oLinear.weight.cpu().detach().numpy()
                else:
                    weight = oLinear[k].weight.cpu().detach().numpy()
                
                if not twoStageFlag:
                    inDim = weight.shape[-1]
                    for i in range(inDim):
                        fig = plt.figure()
                        curWeight = weight[...,i]
                        curWeight = curWeight.reshape(len(times),-1)
                        plt.plot(times,curWeight) #
                        plt.title(f'{k}')
                        figures.append(fig)
                else:
                    inDim = weight.shape[0]
                    for i in range(inDim):
                        fig = plt.figure()
                        curWeight = weight[i]
                        plt.plot(times,curWeight) #
                        plt.title(f'{k}')
                        figures.append(fig)
        else:
            fig = plt.figure()
            weight = oLinear.weight.cpu().detach().numpy()
            inDim = weight.shape[-1]
            for i in range(inDim):
                weight = weight[...,i]
                weight = weight.reshape(len(times),-1)
                plt.plot(times,weight) #
                plt.title(f'{k}')
                figures.append(fig) 
            figures.append(fig)
        return figures

    def __call__(self,oriModel):
        oriModel.eval()
        figures = []

        #plot the nonlinear
        nonLinTRF = oriModel.oNonLinTRF
        if 'TRFs' in nonLinTRF.lastState:
            curFigs1 = self.plotTransformTRF(nonLinTRF)
            curFigs = self.plotTransformTRFLinear(nonLinTRF) 
            curFigs.extend(curFigs1)
            figures.extend(curFigs)
        
        linTRF = oriModel.oTRF
        figs = self.plotAllTRFWeights(linTRF)
        figures.extend(figs)
        
        return figures