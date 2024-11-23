from StimRespFlow.DataProcessing.DeepLearning.Trainer import CTrainerFunc
import torch
import numpy as np
from nntrf.models import CNNTRF
from torch.nn.functional import pad
from matplotlib import pyplot as plt

class CTrainForwardFunc(CTrainerFunc):
    
    def func(self, engine, batch):
        self.trainer.optimizer.zero_grad()
        self.model.train()
        x,y,tIntvl,otherStim,recordInfo = self.model.parseBatch(batch)
        assert len(set(recordInfo)) == 1
        # global idx
        # idx += 1
        # if idx == 10:
        # # if x[0][0][-1] == 0:
        #     print(idx)
        #     torch.save({x,y,tIntvl},f'inputTensor_{idx}.pt')
        #     torch.save(self.model.state_dict(),f'model_{idx}.pt')
        pred,y = self.model(x,y,tIntvl,otherStim,info = recordInfo)
        loss= self.trainer.criterion(pred, y)
        # print('???',pred.shape,y.shape)
        loss.backward()
        self.trainer.optimizer.step()
        return x,y,pred,loss
        
class CEvalForwardFunc(CTrainerFunc):
    def func(self, engine, batch):
        self.model.eval()
        x,y,tIntvl,otherStim,recordInfo = self.model.parseBatch(batch)
        assert len(set(recordInfo)) == 1
        # print(tIntvl[:,:,-1],y.shape)
        pred,y = self.model(x,y,tIntvl,otherStim,info = recordInfo)
        return x,y,pred

class CEvalForwardFunc_semResidual(CTrainerFunc):
    def func(self, engine, batch):
        self.model.eval()
        x,y,tIntvl,otherStim,recordInfo = self.model.parseBatch(batch)
        assert len(set(recordInfo)) == 1
        # print(tIntvl[:,:,-1],y.shape)
        pred,y = self.model(x,y,tIntvl,otherStim,info = recordInfo, ifSplitOtherPred = True)
        nonlinPred,controlPred = pred
        return x, y - controlPred, nonlinPred


def seqLast_pad_zero(seq):
    maxLen = max([i.shape[-1] for i in seq])
    output = []
    for i in seq:
        output.append(pad(i,(0,maxLen - i.shape[-1])))
    return torch.stack(output,0)



def collate_fn_CMixedTRF(samples):
    stimKeys = samples[0][0].keys()
    stims = {k:[] for k in stimKeys}
    resps = []
    infos = {k:[] for k in samples[0][2].keys()}
    for smpl in samples:
        s,r,info = smpl
        for k in stimKeys:
            stims[k].append(s[k])
        resps.append(r)
        for k in infos:
            infos[k].append(info[k])
    resps = seqLast_pad_zero(resps)
    for k,v in stims.items():
        if isinstance(v[0], torch.Tensor) and k != 'tIntvl':
            stims[k] = seqLast_pad_zero(v)
    #pad1 for stim key except 'vector' and 'tIntvl'
    #pad2 for 'vector' only,check if len(transforms) == len(tIntvl) in oneofbatch
    return stims,resps,infos

#device = torch.device('cuda')
# oTRFs = torch.nn.ModuleDict()
# oTRF = CCNNTRF(2, 128, 0, 700, 64)
# oTRFs['NS'] = oTRF
# oTRFs.eval()
# oNonLinTRF = CFTRF('CLSTMSeqContexterNoNorm',1,128,0,700,64)
def buildMixedRF(generalConfig, linConfig, nonLinConfig):
    multiHeadKeys = nonLinConfig['multiHeadKeys']
    device = nonLinConfig['device']
    oTRFs = torch.nn.ModuleDict()
    for key in multiHeadKeys:
        oTRF = CNNTRF(**linConfig,**generalConfig)
        oTRFs[key] = oTRF
    oNonLinTRF = CFTRF(**nonLinConfig,**generalConfig)
    oMixedRF = CMixedRF(device = device,
                         oTRF = oTRFs,oNonLinTRF = oNonLinTRF)
    oMixedRF = oMixedRF.to(device)
    return oMixedRF,{'generalConfig':generalConfig, 
                      'linConfig':linConfig, 
                      'nonLinConfig':nonLinConfig}

def from_pretrainedMixedRF(configs,state_dict,cpu = False):
    oMixedRF,_ = buildMixedRF(**configs)
    if isinstance(state_dict,str):
        if cpu:
            oMixedRF.load_state_dict(torch.load(state_dict,map_location=torch.device('cpu'))['state_dict'])
        else:
            oMixedRF.load_state_dict(torch.load(state_dict)['state_dict'])
    else:
        oMixedRF.load_state_dict(state_dict)
    return oMixedRF


class CMixedRF(torch.nn.Module):
        
    def __init__(self,device,oTRF,oNonLinTRF,ifWeighted = True,
                 ifZscore = False):
        super().__init__()
        self.oTRF = oTRF
        self.oNonLinTRF = oNonLinTRF
        self.weights = torch.nn.Parameter(torch.ones(2))
        self.device = device
        self.ifWeighted = ifWeighted
        self.ifZscore = ifZscore
        self.configToSave = ['device','ifWeighted','ifZscore']
          

    def parseBatch(self,batch):
        # x,y,tIntvl,otherStim,recordInfo = batch
        # x,y,tIntvl = x.to(self.device),y.to(self.device),tIntvl.to(self.device)
        # for idx,stim in enumerate(otherStim):
        #     if type(stim) is not dict:
        #         otherStim[idx] = stim.to(self.device)
        xDict,y,info = batch
        recordInfo = info['datasetName']
        x = xDict['vector'].to(self.device) if 'vector' in xDict else None
        if 'timeShiftCE' in xDict:
            if x is None:
                x = xDict['timeShiftCE'].to(self.device)
            else:
                xTemp = xDict['timeShiftCE'].to(self.device)
                x = torch.cat([x,xTemp],dim = 1)
        if 'tIntvl' in xDict:
            tIntvl = xDict['tIntvl']
            if isinstance(tIntvl, torch.Tensor):
                tIntvl = tIntvl.to(self.device)
            elif isinstance(tIntvl, list):
                for idx,item in enumerate(tIntvl):
                    tIntvl[idx] = item.to(self.device)
            else:
                raise NotImplementedError
        else:
            tIntvl = None
        if y is not None:
            y = y.to(self.device)
        otherStim = []
        for k,v in xDict.items():
            if k not in ['vector','tIntvl','timeShiftCE']:
                if isinstance(v, torch.Tensor):
                    otherStim.append(v.to(self.device))

        # #to prevent -100 time lag and latter result leak in here
        # lenResp = y.shape[-1]
        # if x is not None:
        #     boolTest = (tIntvl[0][0] > lenResp / self.oNonLinTRF.fs)
        #     if any(boolTest):
        #         cutIdx = torch.argmax(boolTest.short())
        #         x = x[:,:,:cutIdx]
        #         tIntvl = tIntvl[:,:,:cutIdx]
        
        # for idx in range(len(otherStim)):
        #     otherStim[idx] = otherStim[idx][:,:,:lenResp]
        
        return x,y,tIntvl,otherStim,recordInfo

    def forward(self,x,y,tIntvl,otherStim,info = None, ifSplitOtherPred = False):
        #otherStim is a list of tensors and others
        # print('model 155',score.shape)

        #get the scrch prediction
        predList = []
        if x is not None:
            if self.ifZscore:
                means = []
                stds = []
                for nBatch in range(tIntvl.shape[0]):
                    # print(tIntvl.shape)
                    nPaddedZero = torch.ceil(tIntvl[nBatch][1][-1] * self.oNonLinTRF.fs).long() - x[nBatch].shape[-1]
                    zeros = torch.zeros(x[nBatch].shape[0],nPaddedZero).to(self.device)
                    X_Zeros = torch.cat([x[nBatch],zeros],dim = -1)
                    mean = X_Zeros.mean(-1,keepdim = True)
                    std = X_Zeros.std(-1,keepdim = True)
                    # print(X_Zeros.shape,x[nBatch].mean(-1),mean,std)
                    means.append(mean)
                    stds.append(std)
                mean = torch.stack(means,dim = 0)
                std = torch.stack(stds,dim = 0)
                x = (x - mean) / std
                # print(x.shape,x.mean(-1),x.std(-1))
            nonLinPred = self.oNonLinTRF(x,tIntvl,info = info)
        else:
            nonLinPred = None
        
        if len(otherStim)>0:
            sharedLenOtherStim = min([i.shape[-1] for i in otherStim] + [y.shape[-1]])
            if nonLinPred is not None:
                sharedLen = min(nonLinPred.shape[-1],sharedLenOtherStim)
            else:
                sharedLen = sharedLenOtherStim

            otherStimToAdd = [stim[:,:,:sharedLen] for stim in otherStim]
            newOtherStim = torch.cat(otherStimToAdd,dim=1)
            if self.ifZscore:
                newOtherStim = (newOtherStim - newOtherStim.mean(-1,keepdim = True)) / newOtherStim.std(-1,keepdim = True)
                # print(newOtherStim.shape,newOtherStim.mean(-1),newOtherStim.std(-1))
            if info is None:
                linPred = self.oTRF(newOtherStim)
            else:
                assert len(set(info)) == 1
                linPred = self.oTRF[info[0]](newOtherStim)

            if nonLinPred is not None:
                predList.append(nonLinPred[:,:,:sharedLen])
            else:
                predList.append(0)
            predList.append(linPred)

        else:
            assert nonLinPred is not None
            sharedLen = min(nonLinPred.shape[-1],y.shape[-1])
            predList.append(nonLinPred[:,:,:sharedLen])
        cropedY = y[:,:,:sharedLen]

        assert len(predList) <= 2 and len(predList) > 0

        if len(predList) == 2:
            w1,w2 = 1,1
            if self.ifWeighted:
                w1, w2 = self.weights[0], self.weights[1]
                # self.debug = [predList[0].cpu().detach().numpy(),predList[1].cpu().detach().numpy(),self.weights[0].cpu().detach().numpy(),self.weights[1].cpu().detach().numpy()]
            else:
                # print(torch.max(predList[0]),torch.max(predList[1]))
                w1,w2 = 1,1 
            if ifSplitOtherPred:
                pred = [predList[0] * w1, predList[1] * w2]
            else:
                pred = predList[0] * w1 + predList[1] * w2 
        else:
            pred = predList[0]
        # stop
        return pred,cropedY