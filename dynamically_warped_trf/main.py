import numpy as np
from dynamically_warped_trf.utils.io import load_dataset
from dynamically_warped_trf.utils.args import get_arg_parser
from dynamically_warped_trf.core import execute

if __name__ == '__main__':
    args = get_arg_parser()
    otherParam = {}
    for k in args.__dict__:
        otherParam[k] = args.__dict__[k]
    ds = load_dataset(args.dataset, r'/scratch/jdou3/Mapping/dataset')
    stimuliDict = ds.stimuliDict
    # for k in stimuliDict:
    #     print(stimuliDict[k].keys())
    stimFeats = args.linStims.copy()
    ds.stimFilterKeys = stimFeats

    foldList = args.foldList
    test_mtrf = args.test_mtrf
    studyName = args.studyName
    randomSeed = args.randomSeed

    testResults = []
    devResultsReduce = []
    testResultsReduce = []

    for i in execute.iterFold():
        datasets = ds.nestedKFold(i,10)
        if test_mtrf:
            (
                bestDevMetricsReduce, 
                testMetricsReduce,
                testMetrics, 
                oExpr
            ) = execute.test_mtrf(
                studyName, 
                datasets, 
                [i,10], 
                otherParam
            )
        else:
            (
                oTrainer,
                bestModel,
                modelMTRF,
                configs,
                bestDevMetricsReduce,
                oRun,
                oExpr
            ) = execute.train(
                studyName,
                {i:datasets[i] for i in ['train','dev']},
                randomSeed,[i,10],
                otherParam,
                args.epoch
            )
            (
                testMetricsReduce,
                testMetrics
            ) = execute.test(
                oTrainer,
                bestModel,
                modelMTRF,
                datasets['test'],
                oRun = oRun, 
                otherParam = otherParam
            )
        print(bestDevMetricsReduce)
        testResults.append(testMetrics)
        devResultsReduce.append(bestDevMetricsReduce)
        testResultsReduce.append(testMetricsReduce)