import numpy as np
import scipy
from StellarInfra import siIO, siDM, plt
from statsmodels.stats.multitest import fdrcorrection
from TRFOps.visualize import plotTopoplot

def testImprv(imprv, ths = 0.05):
    imprv_stat = []
    for i in range(imprv.shape[1]):
        stat,p = scipy.stats.wilcoxon(imprv[:,i],alternative='greater')
        imprv_stat.append([stat,p])
    imprv_stat = np.array(imprv_stat)
    rejected,pvalue_corrected = fdrcorrection(imprv_stat[:,1])
    chanIdx = np.where(pvalue_corrected <= ths)
    return imprv_stat,pvalue_corrected,imprv,chanIdx

def collectResultsAcrossFolds(root,tarFileName):
    results = []
    print(root, len(siDM.getSubFolderName(root)))
    assert (len(siDM.getSubFolderName(root)) in [4, 10, 15])
    for fd in siDM.getSubFolderName(root):
        fname = root / fd / tarFileName
        results.append(siIO.loadObject(fname))
    return results
    
def reduceR(results, ifReduce = True):
    collaR = {'r':[],'mTRF_r':[]}
    for r in results:
        collaR['r'].append([r_['r'] for r_ in sorted(r,key = lambda x:x['subjNum']) if 'r' in r_])
        collaR['mTRF_r'].append([r_['mTRF_r'][0] for r_ in sorted(r,key = lambda x:x['subjNum'])])

    if ifReduce:
        if len(collaR['r'] ) > 0:
            collaR['r'] = np.array(collaR['r']).mean(0)
        else:
            collaR['r'] = np.array(collaR['r'])
        collaR['mTRF_r'] = np.array(collaR['mTRF_r']).mean(0)
    else:
        collaR['r'] = np.array(collaR['r'])
        collaR['mTRF_r'] = np.array(collaR['mTRF_r'])
    return collaR

def collectR(path:str):
    results = collectResultsAcrossFolds(path,'testMetrics.dict')
    ### do the stat test
    collaR = reduceR(results)
    print(collaR['r'].shape,collaR['mTRF_r'].shape)
    return collaR

def testImprvOverLin(metrics,ths = 0.05):
    #metrics: {'r':(nSubj,nChan),'mTRF_r':(nSubj,nChan)}
    nSubj,nChan = metrics['r'].shape
    print(f'nSubj:{nSubj}, nChan:{nChan}')
    r = np.array(metrics['r'])
    mTRF_r = np.array(metrics['mTRF_r'])
    imprv = r - mTRF_r
    
    imprv_stat,pvalue_corrected,imprv,chanIdx = testImprv(imprv,ths)
    figs = []
    if imprv_stat.shape[1] > 1:
        f = plotTopoplot(pvalue_corrected, title = 'corrected p value', chanIdx = chanIdx, sensors = True, res = 1024, units = 'r')
        figs.append(f)
        f = plotTopoplot(imprv.mean(0),sensors =True, title = f'improve value (p<={ths})',chanIdx = chanIdx)
        figs.append(f)

    results = {'imprv_stat':imprv_stat[:,0],
                'imprv':imprv,
                'pvalue_corrected':pvalue_corrected,
                'chanIdx':chanIdx}

    def fSaveFig(root):
        for idx,fig in enumerate(figs):
            fig.savefig(root +  f'/imprv_stats_{idx}.png')
            plt.close(fig)

    return fSaveFig,results

def analyze_imprv(root):
    collaR = collectR(root)
    fSaveFig,results = testImprvOverLin(collaR)
    fSaveFig(root)
    siIO.saveObject(results,root + f'/imprv_stats.bin')
