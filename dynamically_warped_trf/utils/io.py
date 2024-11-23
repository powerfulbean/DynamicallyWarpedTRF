import warnings
import os
from StimRespFlow.DataStruct.DataSet import CDataSet

def checkFolder(folderPath):
#    print(folderPath)
#    if not isinstance(folderPath,str):
#        return
    if not os.path.isdir(folderPath) and not os.path.isfile(folderPath):
        warnings.warn("path: " + folderPath + " doesn't exist, and it is created")
        os.makedirs(folderPath)

''' Python Object IO'''
def pickle_load(filePath):
    import pickle
    file = open(filePath, 'rb')
    temp = pickle.load(file)
    return temp

def pickle_save(Object,folderName,tag=None, ext = '.bin'):
    if tag is None:
        file = open(folderName, 'wb')
    else:
        checkFolder(folderName)
        file = open(folderName + '/' + str(tag) + ext, 'wb')
    import pickle
    pickle.dump(Object,file)
    file.close()

def load_dataset(datasetname, root):
    assert datasetname in ['ns', 'cpt', 'cpf']
    state_dict = pickle_load(f'{root}/{datasetname}.pkl')
    dataset = CDataSet.load(state_dict)
    return dataset

