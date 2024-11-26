from StimRespFlow.DataProcessing.DeepLearning.Trainer import CTrainerFunc
import torch
import numpy as np
from nntrf.models import CNNTRF, ASTRF, FuncTRFsGen, TwoMixedTRF, LTITRFGen, msec2Idxs, Idxs2msec, TRFAligner
from nntrf import models as nntrf_models
from torch.nn.functional import pad
from matplotlib import pyplot as plt

class CTrainForwardFunc(CTrainerFunc):
    
    def func(self, engine, batch):
        self.trainer.optimizer.zero_grad()
        self.model.train()
        pred,y = self.model(*batch)
        loss= self.trainer.criterion(pred, y)
        # print('???',pred.shape,y.shape)
        loss.backward()
        self.trainer.optimizer.step()
        return None,y,pred,loss
        
class CEvalForwardFunc(CTrainerFunc):
    def func(self, engine, batch):
        self.model.eval()
        pred,y = self.model(*batch)
        return None,y,pred


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


def build_mixed_model(
    fs,
    tmin_ms,
    tmax_ms,
    linFeats,
    nonLinFeats,
    linInDim,
    nonlinInDim,
    outDim,
    transformer_name,
    limitOfShift_idx,
    nBasis,
    mode,
    auxInDim,
    nNonLinWin,
    device,
) -> TwoMixedTRF: 

    trf1 = CNNTRF(
        linInDim,
        outDim,
        tmin_ms,
        tmax_ms,
        fs
    )

    trf2 = ASTRF(
        nonlinInDim, 
        outDim, 
        tmin_ms, 
        tmax_ms, 
        fs, 
        trfsGen = torch.nn.Identity(),
        device = device
    )
    
    #print(trf2.bias)
    nTransParams = len(FuncTRFsGen.parse_trans_params(mode))
    #the module that will estimate the transformation parameters
    transformer = getattr(nntrf_models, transformer_name)(nonlinInDim + auxInDim, nTransParams, nNonLinWin)
    #module that estimates transformation parameter
    
    # # trf2.set_trfs_gen(trfsGen)
    trfsGen = FuncTRFsGen(
        nonlinInDim, 
        outDim, 
        tmin_ms, 
        tmax_ms, 
        fs, 
        basisTRFName='fourier', 
        limitOfShift_idx=limitOfShift_idx, 
        nBasis = nBasis,
        mode = mode,
        transformer = transformer,
        device = device
    ).to(trf2.device)
    trf2.trfsGen = trfsGen
    mixedRF = TwoMixedTRF(
        device,
        [trf1, trf2],
        [linFeats, nonLinFeats]
    )
    
    return mixedRF

def from_pretrainedMixedRF(configs,state_dict,cpu = False):
    oMixedRF = build_mixed_model(**configs)
    if isinstance(state_dict,str):
        if cpu:
            oMixedRF.load_state_dict(torch.load(state_dict,map_location=torch.device('cpu'))['state_dict'])
        else:
            oMixedRF.load_state_dict(torch.load(state_dict)['state_dict'])
    else:
        oMixedRF.load_state_dict(state_dict)
    return oMixedRF
