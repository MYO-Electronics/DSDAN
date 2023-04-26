import scipy.io as sio
import numpy as np
from Config import *
from torch.utils.data import DataLoader, TensorDataset
from InterPolation import interpolation


def get_numpy(rootdir, sub, dataname='HD_sEMG'):
    if dataname == 'Low_density_sEMG':
        data_path = rootdir + str(sub) + '_sample.npy'
        label_path = rootdir + str(sub) + '_label.npy'
        source = np.load(data_path)
        label_train = np.load(label_path)
        source = np.stack((source,)*3, 1)
        label_train = label_train - 1
    elif dataname == 'HD_sEMG':
        path = rootdir + sub + '/data_test.mat'
        data = sio.loadmat(path)
        tem_source = data['data_test']
        source = tem_source
        source = np.transpose(source, axes=(0, 3, 1, 2))
        
        path = rootdir + sub + '/label_test.mat'
        data = sio.loadmat(path)
        label_train = data['label_test']
    else:
        source = None
        label_train = None
        print('dataset EORRO')
    source_label = np.transpose(label_train)
    source_label = source_label.reshape(-1)
    return source, source_label


def load_data(rootdir, subject_list, batch_size, kwargs, interpshape=10, dataname=dataset, shuffle=True, drop_last=True):
    sources = np.empty(shape=(0, 3, interpshape, interpshape))
    sources_label = []
    for i,sub in enumerate(subject_list):
        source, source_label = get_numpy(rootdir, sub, dataname)
        if interpshape != 10:
            source = interpolation(source, interpshape)
        sources = np.concatenate((sources, source), axis=0)
        sources_label = np.concatenate((sources_label, source_label), axis=0)
        
    sources_set = TensorDataset(torch.tensor(sources, dtype=torch.float),torch.tensor(sources_label, dtype=torch.float))
    source_loader = DataLoader(dataset=sources_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)

    return source_loader