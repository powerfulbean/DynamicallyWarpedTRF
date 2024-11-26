import torch
from funcTRF.Parts import CTRFFintuner,CTRFAligner,CLinearTRF
from funcTRF import Parts
from funcTRF import Model
import numpy as np
import skfda
from scipy.stats import pearsonr
from nntrf.models import CNNTRF,msec2Idxs,Idxs2msec
from torch.nn.functional import pad
from torch.distributions.normal import Normal
from scipy.stats import zscore
from matplotlib import pyplot as plt
import math

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
):
    multiHeadKeys = ['nan']
    oTRFs = torch.nn.ModuleDict()
    for key in multiHeadKeys:
        oTRF = CNNTRF(
            linInDim,
            outDim,
            tmin_ms,
            tmax_ms,
            fs
        )
        oTRFs[key] = oTRF
    if transformer_name == 'CausalConv':
        transformer_name = 'CCNNSeqContexterNormalBiasPureConv'
    oNonLinTRF = Model.CFTRF(
        transformer_name,
        nonlinInDim,
        outDim,
        tmin_ms,
        tmax_ms,
        fs,
        device,
        ['nan'],
        limitOfShift_idx = limitOfShift_idx,
        nBasis = nBasis,
        mode = mode,
        auxInDim = auxInDim,
        nNonLinWin = nNonLinWin
    )
    oMixedRF = CMixedRF(device = device,
                         oTRF = oTRFs,oNonLinTRF = oNonLinTRF)
    # oMixedRF = oMixedRF.to(device)
    return oMixedRF

def from_pretrainedMixedRF(config,state_dict,cpu = False):
    oMixedRF = build_mixed_model(**config)
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
        self._model = Model.CMixedRF(
            device,
            oTRF,
            oNonLinTRF,
            ifWeighted = True,
            ifZscore = False
        )

    @property
    def trfs(self):
        return [self._model.oTRF['nan'], self._model.oNonLinTRF]

    def parseBatch(self,xDict,y):
        # x,y,tIntvl,otherStim,recordInfo = batch
        # x,y,tIntvl = x.to(self.device),y.to(self.device),tIntvl.to(self.device)
        # for idx,stim in enumerate(otherStim):
        #     if type(stim) is not dict:
        #         otherStim[idx] = stim.to(self.device)
        # print(xDict.keys())
        xDictNew = {}
        info = {'datasetName':['nan']}
        tIntvl = None
        if 'lex_sur' in xDict:
            tIntvl = xDict['lex_sur']['timeinfo']
        elif 'uni_pnt' in xDict:
            tIntvl = xDict['uni_pnt']['timeinfo']
        xDictNew['vector'] = xDict['lex_sur']['x']
        if 'uni_pnt' in xDict:
            xDictNew['timeShiftCE'] = xDict['uni_pnt']['x']
        xDictNew['tIntvl'] = tIntvl
        xDictNew['onset'] = xDict['onset']
        xDictNew['env'] = xDict['env']

        return self._model.parseBatch((xDictNew, y, info))

    def forward(self,xDict,y):
        #otherStim is a list of tensors and others
        # print('model 155',score.shape)

        x,y,tIntvl,otherStim,info = self.parseBatch(xDict,y)
        return self._model(x,y,tIntvl,otherStim,info)
    
    def fitFuncTRF(self, w):
        return self._model.oNonLinTRF.fitFTRFs(w,'nan')
    
    def vis(self):
        return self._model.oNonLinTRF.oFTRF['nan'].visResult()
    
    def set_linear_weights(self, w, b):
        return self._model.oNonLinTRF.loadFromMTRFpy(w, b, 'nan')


    @property
    def if_enable_trfsGen(self):
        return self._model.oNonLinTRF.ifEnableNonLin

    @if_enable_trfsGen.setter
    def if_enable_trfsGen(self,x):
        self._model.oNonLinTRF.ifEnableNonLin = x  

    def stop_update_linear(self):
        return self._model.oNonLinTRF.stopUpdateLinear()

    def get_params_for_train(self):
        return self._model.oNonLinTRF.getParamsForTrain()
