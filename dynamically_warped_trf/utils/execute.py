import numpy as np
import torch
from StimRespFlow.Engines.ResearchManage import CStudy,CExpr
from dynamically_warped_trf.utils.io import pickle_save
from dynamically_warped_trf.mTRFpy.DataStruct import buildListFromSRFDataset
from dynamically_warped_trf.mTRFpy import Model as mtrfModel


def iterFold():
    for i in range(9,-1,-1):#9,-1,-1l
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
    ctxExtClassName = otherParam.get('ctxModel','CausalConv')
    
    fs = datasets['train'].srate
    extraTimeLag = 200
    limitOfShift_idx = int(np.ceil(fs * extraTimeLag/1000))

    inDimLinTRF = len(linStims) - 1
    if 'tIntvl' in linStims:
        inDimLinTRF -= 1
    stimFeats = nonLinStims
    print('stims being used: ', stimFeats)
    inDim = 1 
    auxInDim = len(stimFeats) - 1
    if 'tIntvl' in stimFeats:
        auxInDim -= 1


    tmin_ms = minLag
    tmax_ms = maxLag
    
    stateModuleName = ctxExtClassName
    device = torch.device('cpu')

    generalConfig = {
        'tmin_ms':tmin_ms,
        'tmax_ms':tmax_ms,
        'fs':fs,
    }
    
    
    linConfig = {
        'inDim':inDimLinTRF,
        'outDim':outDim,
    }
    
    
    nonLinConfig = {
        'TransformName':stateModuleName,
        'inDim':inDim,
        'outDim':outDim,
        'limitOfShift_idx':limitOfShift_idx,
        'nBasis':nBasis, 
        'mode':fTRFMode, 
        'auxInDim':auxInDim,
        'device':device,
        'nNonLinWin':nNonLinWin
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
        oLog('inDimLinTRF', inDimLinTRF, 'inDim', inDim, 'auxInDim', auxInDim)
        ''' start preparing Ridge Regression of TRF '''
        wds = 10**np.arange(-4,4).astype(float)
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
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        finalStims = []
        nonTIntvlLinStims = [s_ for s_ in linStims if s_ != 'tIntvl']
        finalStims = nonTIntvlLinStims[:-1] + nonLinStims
        oLog('oMixed Stims', finalStims)
        for d_ in datasets:
            datasets[d_].stimFilterKeys = finalStims

        stop

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
            generalConfig,
            linConfig,
            nonLinConfig,
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
        bestModel_best.oNonLinTRF.ifEnableNonLin = True
        bestModel_best.oNonLinTRF.stopUpdateLinear()
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
            curDatasetTest = CTorchDataset(curDsTest,T = False, device = device)
            
            if len(curDatasetTest) > 0:
                dsTestMTRF = curDsTest.copy()
                dsTestMTRF.stimFilterKeys = linStims
                # if 'timeShiftCE' in dsTestMTRF.stimFilterKeys:
                    # dsTestMTRF.stimFilterKeys = dsTestMTRF.stimFilterKeys[:-1]
                stimTest,respTest,keys = buildListFromSRFDataset(dsTestMTRF,zscore = False)
                _,mTRF_r,mTRF_err = modelMTRF.predict(stimTest,respTest)
                curDataloaderTest = torch.utils.data.DataLoader(curDatasetTest,batch_size = 1, shuffle = True,num_workers = 1)
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
        siIO.saveObject(testResults,oRun.folder + '/testMetrics.dict')
        oRun.update()

    return testMetricsReduce,testResults