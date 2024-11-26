import torch
from StimRespFlow.DataStruct.DataSet import CDataSet
class TorchDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset:CDataSet, device = 'cpu'):
        dataset.ifOldFetchMode = False
        self.dataset = dataset
        self.device = device

    def __getitem__(self, index):
        # print('torchdata', index)
        stim_dict, resp, _ = self.dataset[index]
        resp = torch.FloatTensor(resp).to(self.device)
        stim_dict_tensor = {}
        for k in stim_dict:
            stim = stim_dict[k]
            if isinstance(stim, dict):
                stim_dict_tensor[k] = {
                    'x':torch.FloatTensor(stim['x']).to(self.device),
                    'timeinfo':torch.FloatTensor(stim['timeinfo']).to(self.device)
                }
            else:
                stim_dict_tensor[k] = torch.FloatTensor(stim).to(self.device)
        return stim_dict_tensor, resp
    
    def __len__(self):
        return len(self.dataset)