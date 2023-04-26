import torch

# device
cuda_ids = '1'
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory':True} if torch.cuda.is_available() else {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset setting
dataset = 'HD_sEMG'
if dataset == 'Low_density_sEMG':
    class_num = 6
    domain_list = ['S1', 'S2', 'S3']
    root_path = './dataset/Low_density_sEMG/'
elif dataset == 'HD_sEMG':
    class_num = 6
    domain_list = ['S1', 'S2', 'S3']
    root_path = "./dataset/HD_sEMG/"
interpshape = 64


# model setting
seed = 2
pretrained = True

# training
batch_size = 32
lr = 0.01
lambda1 = 0.1
lambda2 = 1.0
epochs = 2

# optimizer
momentum = 0.9
l2_decay = 5e-4
