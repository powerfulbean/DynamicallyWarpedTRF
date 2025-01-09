from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
from StimRespFlow.Engines.ResearchManage import CStudy,CExpr
from StimRespFlow.DataProcessing.DeepLearning.Trainer import CTrainer,fPickPredTrueFromOutputT
from StimRespFlow.DataProcessing.DeepLearning.Metrics import CMPearsonr
from dynamically_warped_trf.utils import count_parameters
from dynamically_warped_trf.utils.io import pickle_save, CLog
from dynamically_warped_trf.mTRFpy.DataStruct import buildListFromSRFDataset
from dynamically_warped_trf.mTRFpy import Model as mtrfModel
from dynamically_warped_trf.core.model import (
    CTrainForwardFunc, 
    CEvalForwardFunc, 
    TwoMixedTRF, 
    ASTRF, 
    CNNTRF, 
    build_mixed_model, 
    from_pretrainedMixedRF,
    PlotInterm
)
# from dynamically_warped_trf.core.model2 import build_mixed_model, from_pretrainedMixedRF
from dynamically_warped_trf.core import torchdata

def iterFold(nFolds = 10):
    for i in range(nFolds-1,-1,-1):#9,-1,-1l
        yield i

def selectLambdaForMTRF(
    stimTrain,
    respTrain,
    stimDev,
    respDev,
    wds,
    fs, 
    tmin_ms, 
    tmax_ms,
    oLog
):
    finalR = []
    finalErr = []
    oTRFs = []
    
    for wd in wds:
        oLog(wd)
        oTRF = mtrfModel.CTRF()
        oTRF.train(stimTrain, respTrain, 1, fs, tmin_ms, tmax_ms, wd)
        _,r,err = oTRF.predict(stimDev,respDev)
        finalR.append(r)
        finalErr.append(err)
        oTRFs.append(oTRF)
    linR = [np.mean(i) for i in finalR]
    linErr = [np.mean(i) for i in finalErr]
    selLambdaIdx = np.argmax(linR)
    bestDevLinR = np.max(linR)
    bestDevLinErr = np.max(linErr)
    return selLambdaIdx,bestDevLinR,bestDevLinErr

def test_mtrf(
    studyName,
    datasets,
    fold_nFold,
    otherParam = {}
):
    studyName += '_puremtrf'
    tarDir = otherParam.get('tarDir','./')
    
    #config parameters for the model
    timeLags = otherParam.get('timeLags',(0,700))
    minLag = timeLags[0]
    maxLag = timeLags[1]
    fs = datasets['train'].srate
    
    tmin_ms = minLag
    tmax_ms = maxLag
    
    exprLogDict = {}
    exprLogDict.update(otherParam)
    print(exprLogDict)

    oStudy = CStudy(
        tarDir, studyName,
        [
            ['linStims','dataset'],
        ]
    )
    oExpr = CExpr(oStudy,exprLogDict)
    
    with oExpr.newRun(
        {
            'curFold':fold_nFold[0],
            'totalFold':fold_nFold[1]
        }
    ) as oRun:
        oLog = oRun.oLog
        oLog.ifPrint = True
        ''' start preparing Ridge Regression of TRF '''
        wds = 10**np.arange(-4,4).astype(float)
        dsTrainMTRF = datasets['train'].copy()
        dsDevMTRF = datasets['dev'].copy()

        stimTrain,respTrain,keys = buildListFromSRFDataset(
            dsTrainMTRF,
            zscore = False
        )
        stimDev,respDev,keys = buildListFromSRFDataset(
            dsDevMTRF,
            zscore = False
        )
        
        oLog('keys',keys)
        oLog('start train linear mtrf')

        selLambdaIdx,bestDevLinR,bestDevLinErr = selectLambdaForMTRF(
            stimTrain,
            respTrain,
            stimDev,
            respDev,
            wds,
            fs, 
            tmin_ms, 
            tmax_ms, 
            oLog
        )
        oLog(f'select {wds[selLambdaIdx]} as lambda, best dev r is {bestDevLinR}')
        
        modelMTRF = mtrfModel.CTRF()
        modelMTRF.train(
            stimTrain,
            respTrain,
            1,
            fs,
            tmin_ms,
            tmax_ms,
            wds[selLambdaIdx]
        )

        devResult_best = {}
        devResult_best['mTRF_r'] = bestDevLinR
        devResult_best['mTRF_loss'] = bestDevLinErr
        oRun['devResult'] = devResult_best
    
        testSet = datasets['test'].copy()
        testResults = []
        datasetNames = set([i.descInfo['datasetName'] for i in testSet.records])
        for c in datasetNames:
            subjIDNumList = set(
                [
                    i.descInfo['subj'] 
                    for i in testSet.selectByInfo({'datasetName':c}).records
                ]
            )
            for subjNum in subjIDNumList:
                # print(subjNum)
                result = {}
                result['subjNum'] = subjNum
                result['datasetName'] = c

                curDsTest = testSet.selectByInfo({'subj':subjNum,'datasetName':c})
                # print(len(curDsTest))
                if len(curDsTest) > 0:
                    dsTestMTRF = curDsTest.copy()
                    stimTest,respTest,keys = buildListFromSRFDataset(
                        dsTestMTRF,
                        zscore = False)
                    _,mTRF_r,mTRF_err = modelMTRF.predict(stimTest,respTest)
                    result['mTRF_r'] = np.mean(mTRF_r,axis = 0,keepdims=True)
                    result['mTRF_err'] = np.mean(mTRF_err, axis = 0,keepdims=True)
                    testResults.append(result)
                
        testMetricsReduce = {}
        testMetricsReduce['mTRF_r'] = np.mean([i['mTRF_r'] for i in testResults])
        testMetricsReduce['mTRF_loss'] = np.mean([i['mTRF_err'] for i in testResults])

        oRun['testResult'] = testMetricsReduce
        pickle_save(testResults,oRun.folder + '/testMetrics.dict')

    return devResult_best, testMetricsReduce,testResults, oExpr

def train_step(
    device,
    modelMTRF,
    linW_lrgrLag,
    model_config,
    stimTrain,
    dsTrainMTRF,
    datasets,
    epoch,
    minLr,
    maxLr,
    wd1,
    batchSize,
    logFileName,
    seed = 42, 
    optimStr = 'AdamW', 
    lrScheduler = 'cycle'
):

    #set torch random_seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # oLog = None
    oLog = CLog(logFileName[0],logFileName[1],'')
    oLog.ifPrint = True
    starttime = datetime.now()
    oLog("start time:",starttime)

    model_config['device'] = device
    trainerDir = logFileName[0]

    srate = datasets['train'].srate
    torchDatasets = dict.fromkeys(datasets.keys())

    for key in torchDatasets:
        torchDatasets[key] = torchdata.TorchDataset(datasets[key], device = device)#, zscore = zscore)
    # print(len(torchDatasets['train']))
    dataloaders = dict.fromkeys(torchDatasets.keys())
    # sample_batch = next(iter(
    #     torch.utils.data.DataLoader(torchdata.TorchDataset(datasets['train'], device = device))
    # ))
    # print(sample_batch[0]['lex_sur']['x'].shape)
    # stop
    stim_dict_tensor_old, resp = torchdata.TorchDataset(datasets['train'], device = device)[0]
    resp = resp.clone()[None,...]
    stim_dict_tensor = {}
    for k in stim_dict_tensor_old:
        feat = stim_dict_tensor_old[k]
        if isinstance(feat, dict):
            stim_dict_tensor[k] = {k2:feat[k2].clone()[None, ...] for k2 in feat}
        else:
            stim_dict_tensor[k] = feat.clone()[None, ...]

    
    sample_batch = (stim_dict_tensor, resp)
    assert batchSize == 1
    for key in dataloaders:
        dataloaders[key] = torch.utils.data.DataLoader(torchDatasets[key],batch_size = batchSize, shuffle = True) #need change back to shuffle = True
    
    linW = modelMTRF.w
    linB = modelMTRF.b
    
    oMixedRF:TwoMixedTRF = build_mixed_model(**model_config)
    oLog(oMixedRF)
    oLog('number of trainable parameters',count_parameters(oMixedRF))#,True, oLog))
    #additionaly we also fit a linear TRF with larger time lag for non-linear shifting of TRF
    
    cnntrf:CNNTRF = oMixedRF.trfs[0]
    astrf:ASTRF = oMixedRF.trfs[1]
    try:
        astrf.trfsGen.fitFuncTRF(linW_lrgrLag[2:])
    except:
        oMixedRF.fitFuncTRF(linW_lrgrLag[2:])

    try:
        fig = astrf.trfsGen.basisTRF.vis()
    except:
        fig = oMixedRF.vis()
    fig.savefig(f'{trainerDir}/visFTRF.png')
    plt.close(fig)
    

    cnntrf.loadFromMTRFpy(linW[0:2], linB/2,device)
    try:
        astrf.set_linear_weights(linW[2:], linB/2)
        astrf.if_enable_trfsGen = False
        astrf.stop_update_linear()
    except:
        oMixedRF.set_linear_weights(linW[2:], linB/2)
        oMixedRF.if_enable_trfsGen = False
        oMixedRF.stop_update_linear()
    # print(linW,linB)
    # oMixedRF.oNonLinTRF.ifEnableNonLin = False
    # oMixedRF.oNonLinTRF.stopUpdateLinear()
    # print(oModel.oNonLinTRF.LinearKernels.NS.bias.shape)

    def getLinModelWB(oModel):
        cachedW1 = oModel.trfs[0].oCNN.weight.detach().cpu().numpy()
        cachedB1 = oModel.trfs[0].oCNN.bias.detach().cpu().numpy()
        try:
            cachedW2 = oModel.trfs[1].ltiTRFsGen.weight.detach().cpu().numpy()
            cachedB2 = oModel.trfs[1].ltiTRFsGen.bias.detach().cpu().numpy()
        except:
            cachedW2 = oModel.trfs[1].LinearKernels['nan'].weight.detach().cpu().numpy()
            cachedB2 = oModel.trfs[1].LinearKernels['nan'].bias.detach().cpu().numpy()
        return cachedW1,cachedB1, cachedW2#, cachedB2

    cachedWB = getLinModelWB(oMixedRF)

    #validate the results are almost the same
    mTRFpyInput = stimTrain[0]
    dldr = torch.utils.data.DataLoader(torchdata.TorchDataset(dsTrainMTRF,device = device),batch_size = 1)
    nnTRFInput = next(iter(dldr))
    print(len(nnTRFInput), len(nnTRFInput[0]), len(nnTRFInput[1]))
    predTRFpy = modelMTRF.predict(mTRFpyInput)[0]
    # print(oMixedRF.parseBatch(nnTRFInput)[0].shape)
    try:
        real_feats_keys = oMixedRF.feats_keys
        oMixedRF.feats_keys = [['onset', 'env'], ['lex_sur']]
    except:
        pass
    predNNTRFOutput = oMixedRF(*nnTRFInput)
    predNNTRF = predNNTRFOutput[0].detach().cpu().numpy()[0].T
    # print(predTRFpy.shape, predNNTRF.shape)
    # print(mTRFpyInput, predTRFpy, predNNTRF)
    assert np.allclose(predNNTRF,predTRFpy,rtol=1e-04, atol=1e-07)
    #enable non-linear
    try:
        oMixedRF.feats_keys = real_feats_keys
    except:
        pass
    
    astrf.if_enable_trfsGen = True
    oMixedRF.if_enable_trfsGen = True
    
    criterion = torch.nn.MSELoss()

    if optimStr == 'AdamW':
        params_for_train = None
        try:
            params_for_train = astrf.get_params_for_train()
        except:
            params_for_train = oMixedRF.get_params_for_train()
        optimizer = torch.optim.AdamW(
                        params = params_for_train,
                        lr = minLr,
                        weight_decay = wd1)
    elif optimStr == 'AdamW-amsgrad':
        optimizer = torch.optim.AdamW(
                        params = oMixedRF.oNonLinTRF.getParamsForTrain(),
                        lr = minLr,
                        weight_decay = wd1, amsgrad = True)
    elif optimStr == 'Adam':
        optimizer = torch.optim.Adam(
                        params = oMixedRF.oNonLinTRF.getParamsForTrain(),
                        lr = minLr,
                        weight_decay = wd1, amsgrad = False)
    else:
        raise NotImplementedError()

    # oMixedRF.oNonLinTRF.stopUpdateLinear()
    try:
        astrf.stop_update_linear()
    except:
        oMixedRF.stop_update_linear()
    cycleIter = (len(datasets['train']) // batchSize) * 2
    if lrScheduler == 'cycle':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,minLr,maxLr,cycleIter,mode = 'triangular2',cycle_momentum=False)
    elif lrScheduler is None:
        lr_scheduler = None
    elif lrScheduler == 'reduce':
        assert minLr == maxLr
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 4)
    else:
        raise NotImplementedError()

    oTrainer = CTrainer(epoch, device, criterion, optimizer,lr_scheduler)
    oTrainer.setDir(oLog,trainerDir)
    oTrainer.setDataLoader(dataloaders['train'],dataloaders['dev'])
    fPlot = PlotInterm(srate,sample_batch)
    oTrainer.addPlotFunc(fPlot)
    
    metricPerson = CMPearsonr(output_transform=fPickPredTrueFromOutputT,avgOutput = False)
    oTrainer.addMetrics('corr', metricPerson)
    bestEpoch,bestDevMetrics= oTrainer.train(oMixedRF,'corr',
                                        trainingStep=CTrainForwardFunc,
                                        evaluationStep=CEvalForwardFunc,
                                        patience = 10)
    pickle_save(model_config,oTrainer.tarFolder + '/configs.bin')
    pickle_save(bestDevMetrics,oTrainer.tarFolder + '/devMetrics.bin')
    bestModel:TwoMixedTRF = from_pretrainedMixedRF(model_config, oTrainer.tarFolder + '/savedModel_feedForward_best.pt')        
    bestModel.trfs[1].if_enable_trfsGen = True
    bestModel.trfs[1].stop_update_linear()
    bestModel.eval()

    #assert the linear part is not changed
    newWB = getLinModelWB(bestModel)
    assert all([np.array_equal(cachedWB[i], newWB[i]) for i in range(len(cachedWB))])

    oLog.Mode = 'safe'
    starttime = datetime.now()
    oLog("end time:",starttime)

    oTrainer.trainer = None
    oTrainer.evaluator = None
    oTrainer.model = None
    oTrainer.optimizer = None
    oTrainer.lrScheduler = None
    oTrainer.oLog = None
    oTrainer.dtldTrain = None
    oTrainer.dtldDev = None
    oTrainer.fPlotsFunc = None
    # stop
    return None,model_config,oTrainer,bestEpoch,bestDevMetrics,trainerDir

def train(studyName,datasets,seed,fold_nFold,otherParam = {}, epoch = 100):
    #fold_nFold is a tuple, indicates [current fold, total fold]

    assert all([i in ['train','dev'] for i in datasets])
    assert not 'test' in datasets
    
    #config parameters for training framework
    tarDir = otherParam.get('tarDir','./')
    wd1 = otherParam.get('wd',10e-2)
    lr = otherParam.get('lr',[0.001,0.001])
    minLr = lr[0]
    maxLr = lr[1]
    batchSize = otherParam.get('batchSize',1)
    optimStr = otherParam.get('optimStr','AdamW')

    linStims = otherParam.get('linStims')
    nonLinStims = otherParam.get('nonLinStims')

    #config parameters for the model
    outDim = 128
    nNonLinWin = otherParam.get('nNonLinWin',-1)
    fTRFMode = otherParam.get('fTRFMode','a,b')
    timeLags = otherParam.get('timeLags',(0,700))
    lrScheduler = otherParam.get('lrScheduler',None)
    minLag = timeLags[0]
    maxLag = timeLags[1]
    nBasis = otherParam.get('nBasis',None)
    transformer_name = otherParam.get('ctxModel','CausalConv')
    
    fs = datasets['train'].srate
    extraTimeLag = 200
    limitOfShift_idx = int(np.ceil(fs * extraTimeLag/1000))

    #the last stim will be replaced by dynamic TRF
    linInDim = len(linStims) - 1
    stimFeats = nonLinStims
    print('stims being used: ',linStims, nonLinStims)
    nonlinInDim = 1 
    auxInDim = len(nonLinStims) - nonlinInDim

    tmin_ms = minLag
    tmax_ms = maxLag
    if torch.cuda.is_available():
        device = torch.device('cuda')
        try:
            torch.arange(0,1, device=device)
        except:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    model_config = {
        'tmin_ms':tmin_ms,
        'tmax_ms':tmax_ms,
        'fs':fs,
        'linInDim':linInDim,
        'nonlinInDim':nonlinInDim,
        'outDim':outDim,
        'transformer_name':transformer_name,
        'limitOfShift_idx':limitOfShift_idx,
        'nBasis':nBasis, 
        'mode':fTRFMode, 
        'auxInDim':auxInDim,
        'device':device,
        'nNonLinWin':nNonLinWin,
        'linFeats':linStims[:-1], 
        'nonLinFeats':nonLinStims
    }
    

    exprLogDict = {
        'limitOfShift_idx':limitOfShift_idx
    }
    exprLogDict.update(otherParam)
    print(exprLogDict)
    

    oStudy = CStudy(
        tarDir, studyName,
        [
            ['optimStr'],['lrScheduler','epoch'],
            ['wd','lr'],
            ['nonLinStims','ctxModel'],
            ['dataset','timeLags','nNonLinWin'],
            ['fTRFMode']
        ]
    )

    oExpr = CExpr(oStudy,exprLogDict)
    
    with oExpr.newRun({
        'curFold':fold_nFold[0],
        'totalFold':fold_nFold[1]
    }) as oRun:
        oLog = oRun.oLog
        oLog.ifPrint = True
        # stop
        oLog('linInDim', linInDim, 'nonlinInDim', nonlinInDim, 'auxInDim', auxInDim)
        ''' start preparing Ridge Regression of TRF '''
        wds = 10**np.arange(-4,4).astype(float) #[1] #
        dsTrainMTRF = datasets['train'].copy()
        dsDevMTRF = datasets['dev'].copy()

        dsTrainMTRF.stimFilterKeys = linStims
        dsDevMTRF.stimFilterKeys = linStims
        stimTrain,respTrain,keys = buildListFromSRFDataset(dsTrainMTRF,zscore = False)
        stimDev,respDev,keys = buildListFromSRFDataset(dsDevMTRF,zscore = False)
        
        oLog('linStims', linStims, 'nonLinStims', nonLinStims)
        oLog('mtrf keys',keys)
        oLog('start train linear mtrf')

        (
            selLambdaIdx,
            bestDevLinR,
            bestDevLinErr
        ) = selectLambdaForMTRF(
            stimTrain,
            respTrain,
            stimDev,
            respDev,
            wds,
            fs, 
            tmin_ms, 
            tmax_ms, 
            oLog
        )
        oLog(f'select {wds[selLambdaIdx]} as lambda, best dev r is {bestDevLinR}')
        
        modelMTRF = mtrfModel.CTRF()
        modelMTRF.train(stimTrain,respTrain,1,fs,tmin_ms,tmax_ms,wds[selLambdaIdx])
        # modelMTRF.save('./','testModel')
        oLog('end train linear mtrf')
        oLog.Save()

        modelMTRF_lrgrLag = mtrfModel.CTRF()
        modelMTRF_lrgrLag.train(
            stimTrain, respTrain, 
            1, 
            fs, 
            tmin_ms - extraTimeLag, 
            tmax_ms + extraTimeLag, 
            wds[selLambdaIdx]
        )
        linW_lrgrLag = modelMTRF_lrgrLag.w

        ''' end preparing Ridge Regression of TRF '''

        #make sure we didn't reuse the trained model
        assert 'oMixedRF' not in locals()
        assert 'oMixedRF' not in globals()

        #run the real train
        results = []
        logFileName = oRun.logFileName
        finalStims = list(set(linStims + nonLinStims))
        oLog('oMixed Stims', finalStims)
        for d_ in datasets:
            datasets[d_].stimFilterKeys = finalStims

        # torchds = torchdata.TorchDataset(datasets['train'])
        # print(torchds[0])

        (
            bestModel_best,
            configs_best,
            oTrainer_best,
            bestEpoch_best,
            bestDevMetrics_best,
            trainerDir_best
        ) = train_step(
            device,
            modelMTRF,
            linW_lrgrLag,
            model_config,
            stimTrain,
            dsTrainMTRF,
            datasets,
            epoch,
            minLr,
            maxLr,
            wd1,
            batchSize,
            logFileName,
            seed,
            optimStr = optimStr,
            lrScheduler = lrScheduler
        )
        
        #reload best model
        bestModel_best = from_pretrainedMixedRF(configs_best, trainerDir_best + '/savedModel_feedForward_best.pt')     
        # bestModel_best.oNonLinTRF.ifEnableNonLin = True
        # bestModel_best.oNonLinTRF.stopUpdateLinear()
        bestModel_best.trfs[1].if_enable_trfsGen = True
        bestModel_best.trfs[1].stop_update_linear()
        bestModel_best.eval()

        oRun['bestEpoch'] = bestEpoch_best
        devResult_best = {}
        devResult_best['r'] = np.mean(bestDevMetrics_best['corr'])
        devResult_best['loss'] = np.mean(bestDevMetrics_best['loss'])
        devResult_best['mTRF_r'] = bestDevLinR
        devResult_best['mTRF_loss'] = bestDevLinErr
        oRun['devResult'] = devResult_best

    return oTrainer_best,bestModel_best,modelMTRF,configs_best,devResult_best,oRun,oExpr

    
def test(oTrainer,model,modelMTRF,testSet,oRun = None, otherParam = None):
    #iterate over dataset and subj
    print([i.descInfo['subj'] for i in testSet.records])
    linStims = otherParam.get('linStims')
    nonLinStims = otherParam.get('nonLinStims')
    device = oTrainer.device
    testResults = []
    datasetNames = set([i.descInfo['datasetName'] for i in testSet.records])
    for c in datasetNames:
        subjIDNumList = set([i.descInfo['subj'] for i in testSet.selectByInfo({'datasetName':c}).records])
        for subjNum in subjIDNumList:
            # print(subjNum)
            result = {}
            result['subjNum'] = subjNum
            result['datasetName'] = c

            curDsTest = testSet.selectByInfo({'subj':subjNum,'datasetName':c})
            finalStims = []
            nonTIntvlLinStims = [s_ for s_ in linStims if s_ != 'tIntvl']
            finalStims = nonTIntvlLinStims[:-1] + nonLinStims
            print('oMixed Stims', finalStims)
            curDsTest.stimFilterKeys = finalStims
            # print(len(curDsTest))
            curDatasetTest = torchdata.TorchDataset(curDsTest, device = device)
            
            if len(curDatasetTest) > 0:
                dsTestMTRF = curDsTest.copy()
                dsTestMTRF.stimFilterKeys = linStims
                # if 'timeShiftCE' in dsTestMTRF.stimFilterKeys:
                    # dsTestMTRF.stimFilterKeys = dsTestMTRF.stimFilterKeys[:-1]
                stimTest,respTest,keys = buildListFromSRFDataset(dsTestMTRF,zscore = False)
                _,mTRF_r,mTRF_err = modelMTRF.predict(stimTest,respTest)
                curDataloaderTest = torch.utils.data.DataLoader(curDatasetTest,batch_size = 1, shuffle = True)
                testMetrics = oTrainer.test(model,curDataloaderTest,device = device,evaluationStep = CEvalForwardFunc)
                result['r'] = testMetrics['corr']
                result['loss'] = testMetrics['loss']
                result['mTRF_r'] = np.mean(mTRF_r,axis = 0,keepdims=True)
                result['mTRF_err'] = np.mean(mTRF_err, axis = 0,keepdims=True)
                testResults.append(result)
            
    testMetricsReduce = {}
    testMetricsReduce['r'] = np.mean([i['r'] for i in testResults])
    testMetricsReduce['loss'] = np.mean([i['loss'] for i in testResults])
    testMetricsReduce['mTRF_r'] = np.mean([i['mTRF_r'] for i in testResults])
    testMetricsReduce['mTRF_loss'] = np.mean([i['mTRF_err'] for i in testResults])

    if oRun is not None:
        oRun['testResult'] = testMetricsReduce
        pickle_save(testResults,oRun.folder + '/testMetrics.dict')
        oRun.update()

    return testMetricsReduce,testResults