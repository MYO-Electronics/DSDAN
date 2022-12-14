from __future__ import print_function
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
import torch
import torch.nn.functional as F
import numpy as np
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import ResNet as models
from Config import *
from ranger21 import Ranger21
from lmmd_loss import lmmd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # cuda_ids
torch.autograd.set_detect_anomaly(True)


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, optimizer, source_loader, target_loader):
    model.train()
    iter_feature = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = max(len(source_loader), len(target_loader))
    loss_train1 = 0
    loss_train2 = 0

    for b in range(num_iter):
        if b % len(source_loader) == 0:
            iter_feature = iter(source_loader)
        if b % len(target_loader) == 0:
            iter_target = iter(target_loader)
        data_source, label_source = iter_feature.next()
        data_target, label_target = iter_target.next()

        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target, label_target = data_target.cuda(), label_target.cuda()

        # First Loop
        optimizer.zero_grad()
        label_source_pred1, s_features = model(data_source)
        loss_cls1_s = F.nll_loss(F.log_softmax(label_source_pred1, dim=1), label_source.long())
        label_target_pred1, t_features = model(data_target)
        loss_lmmd = lmmd(s_features, t_features, label_source.long(), F.softmax(label_target_pred1, dim=1))
        loss1 = loss_cls1_s + lambda1 * loss_lmmd
        loss1.backward()
        optimizer.step()

        # generate the pseudo_label
        label_target_pseudo, _ = model(data_target)
        target_pseudo = label_target_pseudo.data.max(1)[1]

        # Second Loop
        optimizer.zero_grad()
        label_target_pred2, _ = model(data_target, student=True)
        loss_cls2_t = F.nll_loss(F.log_softmax(label_target_pred2, dim=1), target_pseudo.long())
        label_source_pred2, _ = model(data_source, student=True)
        loss_cls2_s = F.nll_loss(F.log_softmax(label_source_pred2, dim=1), label_source.long())
        loss2 = loss_cls2_s + lambda2 * loss_cls2_t
        loss2.backward()
        optimizer.step()

        loss_train1 += loss1.detach().item()
        loss_train2 += loss2.detach().item()

    return loss_train1, loss_train2


def test(model, target_loader):
    model.eval()
    test_loss_true = 0
    correct_tea = 0
    correct_online = []
    iter_target = iter(target_loader)
    num_iter = len(target_loader)
    with torch.no_grad():
        for i in range(num_iter):
            verify_data, verify_label = iter_target.next()
            if cuda:
                verify_data, verify_label = verify_data.cuda(), verify_label.cuda().long()

            x_pred, _ = model(verify_data)
            loss = F.nll_loss(F.log_softmax(x_pred, dim=1), verify_label.long())
            test_loss_true += loss.item()
            pred_labels = x_pred.data.max(1)[1]
            correct_online.append(pred_labels.eq(verify_label.data.view_as(pred_labels)).cpu().sum().item())
            correct_tea += correct_online[-1]

    return correct_tea, test_loss_true


if __name__ == '__main__':
    print("torch.cuda.is_available() ???", torch.cuda.is_available())
    print(domain_list)
    set_random_seed(seed)

    for i in range(len(domain_list)):
        torch.cuda.empty_cache()
        target_list = [domain_list[i]]
        source_list = [x for x in domain_list if x not in target_list]
        print('source: {}, target:{}'.format(source_list, target_list))
        target_train_loader = data_loader.load_data(root_path, target_list, batch_size, kwargs, interpshape,
                                                    dataname=dataset, shuffle=True, drop_last=False)
        source_loader = data_loader.load_data(root_path, source_list, batch_size, kwargs, interpshape,
                                                    dataname=dataset, shuffle=True, drop_last=True)

        len_target_dataset = len(target_train_loader.dataset)
        len_source_dataset = len(source_loader.dataset)
        print('Data loading....')
        print('Loaded len_target_dataset : ', len_target_dataset)
        print('Loaded len_source_dataset : ', len_source_dataset)

        model = models.DSDAN(class_num, True)

        if cuda:
            model.cuda()
        #     if torch.cuda.device_count()>1:
        #        model = torch.nn.DataParallel(model,device_ids=[0,1])

        accuracy_tea = []
        correct_best = 0
        correct_last = 0
        loss_min = np.inf
        time_start = time.time()

        optimizer = Ranger21(model.parameters(), lr=lr,
                             num_batches_per_epoch=len(source_loader) * 2,
                             num_epochs=epochs,
                             use_warmup=True,
                             warmdown_active=False,
                             )

        target_datas, target_labels = data_loader.load_data_numpy(root_path, target_list, interpshape, dataset)

        blocks = 30 # len(target_train_loader)-1
        for block in tqdm(range(blocks)):
            # get the acquired target data from target domain
            target_set = TensorDataset(
                torch.tensor(target_datas[0:(batch_size * block + batch_size)], dtype=torch.float),
                torch.tensor(target_labels[0:(batch_size * block + batch_size)], dtype=torch.float))
            target_train_loader1 = DataLoader(dataset=target_set, batch_size=batch_size, shuffle=True, drop_last=False,
                                              **kwargs)
            # get the lastest block of target data
            # lastest_block_data = torch.tensor(target_datas[(batch_size * block):(batch_size * block + batch_size)],
            #                                   dtype=torch.float)
            # lastest_block_label = torch.tensor(target_labels[(batch_size * block):(batch_size * block + batch_size)],
            #                                    dtype=torch.float)

            # adapt the model with acquired target data and source data
            for epoch in range(1):
                loss_train1, loss_train2 = train(model, optimizer, source_loader, target_train_loader1)

            # test the whole target domain after adapting blocks of target datas
            test_correct, test_loss = test(model, target_train_loader)
            accuracy_tea.append(test_correct / len_target_dataset)

            print('\nloss true: ', test_loss / len_target_dataset)
            print('Acc for all, {} correct : {}/{}  ave_correct_rate : {}'.format(target_list, test_correct,
                                                                                  len_target_dataset, accuracy_tea[-1]))

        print('cost time:', time.time() - time_start)
        np.save('./online_test_{}.npy'.format(target_list[0]), np.array(accuracy_tea))
        plt.plot(range(blocks), [100*x for x in accuracy_tea])
        plt.xlabel('Blocks')
        plt.ylabel('accuracy (%)')
        plt.title('online test for {}'.format(target_list[0]))
        plt.savefig('./online_test_for_{}.png'.format(target_list[0]))
        print('process for {} finished!'.format(target_list))
