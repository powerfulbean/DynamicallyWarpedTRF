import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--dataset', type = str, help = "list of dataset for combination")
    parser.add_argument('-c','--ctxModel', type = str, help = "context model",default = 'CLSTMSeqContexterBasic')
    parser.add_argument('--studyName',type = str, default = 'test')
    parser.add_argument('--test_mtrf',action='store_true')
    parser.add_argument('--timeLags', nargs='+', type=float)
    parser.add_argument('--linStims', nargs='+', default = []) #the last of linStims will be used for template
    parser.add_argument('--nonLinStims', nargs='+', default = [])
    parser.add_argument('--batchSize',default=1, type=int)
    parser.add_argument('--epoch',default=100, type=int)
    parser.add_argument('--randomSeed',default=42, type=int)
    parser.add_argument('--nBasis',default=21, type=int)
    parser.add_argument('--wd',default=0.01, type=float)
    parser.add_argument('--lr', nargs='+', type=float, default = [0.001,0.001])
    parser.add_argument('--tarDir', type=str, default = './')
    parser.add_argument('--fTRFMode',type=str, default = 'a,b')
    parser.add_argument('--foldList', nargs='+', type=int, default = [])
    parser.add_argument('--optimStr', type=str, default = 'AdamW')
    parser.add_argument('--lrScheduler', type = str, default = None)
    parser.add_argument('--nNonLinWin', type = int, default = -1)
    return parser.parse_args()