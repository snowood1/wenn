import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

from torch import autograd, optim
from models import *
from helper import *

import torchvision.utils as vutils
import pickle
import datetime
import time
import pandas as pd
pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if round(x, 0) == x else '{:.3f}'.format(x)
pd.options.display.max_columns = 20
pd.options.display.width = 100
import torch.nn as nn


'''
test_in: test in-distribution data
'''
def test_in(model, test_loader, num_classes, df_test, df_test_avg, epoch):
    with torch.no_grad():
        df_tmp = pd.DataFrame(
            columns=['idxs_mask', 'in_ent', 'in_vac', 'in_dis', 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis'])

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # onehot = torch.eye(num_classes, device=device)[target]
            alpha = model(data)  # TODO
            p = alpha / alpha.sum(1, keepdim=True)

            pred = p.argmax(dim=1, keepdim=True)
            idxs_mask = pred.eq(target.view_as(pred)).view(-1)
            ent = cal_entropy(p)
            disn = getDisn(alpha)
            vac_in = (num_classes / torch.sum(alpha, dim=1))
            succ_ent = ent[idxs_mask]
            fail_ent = ent[~idxs_mask]
            succ_dis = disn[idxs_mask]
            fail_dis = disn[~idxs_mask]

            df_tmp.loc[len(df_tmp)] = [i.tolist() for i in
                                       [idxs_mask, ent, vac_in, disn, succ_ent, fail_ent, succ_dis, fail_dis]]

        # print(df_test_batch.keys())
        in_score = df_tmp.sum()
                                    #26032
        fpr, tpr, roc_auc = get_pr_roc(in_score['succ_ent'], in_score['fail_ent'])
        bnd_dect_ent = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

        fpr, tpr, roc_auc = get_pr_roc(in_score['succ_dis'], in_score['fail_dis'])
        bnd_dect_dis = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

        df_test.loc[len(df_test)] = [epoch, *in_score, bnd_dect_ent, bnd_dect_dis]
        df_test_avg.loc[len(df_test_avg)] = [epoch, *in_score.apply(np.average), bnd_dect_ent['auroc'],
                                             bnd_dect_dis['auroc']]

        # print('Test:\t', df_test_avg.tail(1).to_string().replace('\n', '\n\t\t'))
        # print('Test:', df_test_avg.tail(1).to_string(index=False).replace('\n', '\n\t'))
        return df_test, df_test_avg, in_score


'''
test_out: test out-of-distribution data
'''

def test_out(model, ood_loader, num_classes, in_score, df_ood, df_ood_avg, epoch):
    with torch.no_grad():

        df_tmp = pd.DataFrame(
            columns=['ood_ent', 'ood_vac', 'ood_dis'])

        for batch_idx, ood_data in enumerate(ood_loader):
            if type(ood_data) == list:
                ood_data = ood_data[0]
            ood_data = ood_data.to(device)
            alpha_bar = model(ood_data)
            p_bar = alpha_bar / alpha_bar.sum(1, keepdim=True)

            ent_bar = cal_entropy(p_bar)
            disn_bar = getDisn(alpha_bar)
            vac_bar = num_classes / torch.sum(alpha_bar, dim=1)

            df_tmp.loc[len(df_tmp)] = [i.tolist() for i in [ent_bar, vac_bar, disn_bar]]

        out_score = df_tmp.sum()

        fpr, tpr, roc_auc = get_pr_roc(in_score['in_ent'], out_score['ood_ent'])
        ood_dect_ent = {'auroc': round(roc_auc, 3), 'fpr': fpr, 'tpr': tpr}

        fpr, tpr, roc_auc = get_pr_roc(in_score['in_vac'], out_score['ood_vac'])
        ood_dect_vac = {'auroc': round(roc_auc, 3), 'fpr': fpr, 'tpr': tpr}

        df_ood.loc[len(df_ood)] = [epoch, *out_score, ood_dect_ent, ood_dect_vac]
        df_ood_avg.loc[len(df_ood_avg)] = [epoch, *out_score.apply(np.average), ood_dect_ent['auroc'],
                                           ood_dect_vac['auroc']]
        # print('%s'%ood_name, df_ood_avg.tail(1).to_string(index=False).replace('\n','\n\t'))

        return df_ood, df_ood_avg, out_score

'''
First pretrain a classifier to reach a good accuracy. 
Then feed the pre-trained classifier into the wgan framework to calibrate its uncertainty. 
We already have pre-trained weights in the folder 'pretrain'
'''

def pretrain(model, optimizer, train_loader, test_loader, num_classes, epochs=300):
    # for epoch in range(epochs):
    #     if epoch + 1 in [50, 75, 90]:
    #         for group in optimizer.param_groups:
    #             group['lr'] *= .1
    for i in range(epochs):
        model.train()
        model.apply(update_bn_stats)
        accuracy = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            onehot = torch.eye(num_classes, device=device)[target]

            optimizer.zero_grad()

            ##########################
            # (1) Loss from P real
            ###########################

            alpha = model(data)
            s = alpha.sum(1, keepdim=True)
            # s = torch.sum(alpha, dim=1, keepdim=True)
            p = alpha / s
            # vac_in = num_classes / s
            acc_loss = torch.sum((onehot - p) ** 2, dim=1).mean() + \
                       torch.sum(p * (1 - p) / (s + 1), axis=1).mean()

            pred = p.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item() / data.size(0)

            loss1 = acc_loss
            loss1.backward()
            optimizer.step()
        print('epoch, %d\ttrain %.4f ' % (i, accuracy / len(train_loader)), end='\t')

        model.eval()
        model.apply(freeze_bn_stats)

        with torch.no_grad():
            accuracy = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                # onehot = torch.eye(num_classes, device=device)[target]
                alpha = model(data)  # TODO
                p = alpha / alpha.sum(1, keepdim=True)

                pred = p.argmax(dim=1, keepdim=True)
                # idxs_mask = pred.eq(target.view_as(pred)).view(-1)
                accuracy += pred.eq(target.view_as(pred)).sum().item() / data.size(0)
            print('test %.4f' % (accuracy / len(test_loader)))
    return model


def main(
    gamma = 0.01,
    network = 'resnet20',
    dataname = 'CIFAR10',
    ood_dataname= None,  # This parameter is deprecated. We've defined each dataset's ood datasets below.
    grad_clip = 1,
    d_iters = 2,
    e_iters = 1,
    n_epochs = 60,
    batch_size = 256,
    lambda_term=10,
    weight_decay=0.0001,
    nz=128,
    log_interval=5,
    use_augment=True,
    use_validation = False,
    root = 'results',
    use_pretrain = True,
    use_G_grad = False
):

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    foldname = '{}_beta_{}_time_{}'.format(network, gamma, current_time)

    path = '{}/{}/{}'.format(root, dataname, foldname)
    # print('\nPATH: ', path)
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

    if dataname == 'CIFAR5':
        num_classes, split = (5, 5000)
    elif dataname == 'CIFAR100':
        num_classes, split = (100, 10000)
    else:
        num_classes, split = (10, 10000)

    in_channel = 1 if dataname in ['MNIST', 'FashionMNIST'] else 3

    args =['path','dataname', 'ood_dataname', 'network', 'gamma', 'batch_size', 'n_epochs', 'grad_clip','nz',
           'd_iters','e_iters', 'use_augment', 'lambda_term', 'weight_decay', 'log_interval', 'use_validation', 'num_classes', 'split', 'use_pretrain']

    record ={}
    print('---------%s-----------\n'%current_time)
    for arg in args:
        print('{}:\t{}'.format(arg, eval(arg)), end='\n')
        record[arg] = eval(arg)
    print('\n')
    with open('{}/log.txt'.format(path), "a") as file:
        print('---------%s-----------\n' % current_time, file=file)
        for arg in args:
            print('{}:\t{}'.format(arg, eval(arg)), end='\n', file=file)
        print('\n', file=file)


    # #################  show results ####################################
    
    if type(ood_dataname) == str:
        ood_dataname = [ood_dataname]

    ood_loader_list={}

    if dataname =='CIFAR5':

        model = ENN(resnet20(num_classes=num_classes)).to(device)
        model.classifier.load_state_dict(torch.load('pretrain/resnet20_cifar5_eval.pt'))

        train_loader, valid_loader, test_loader, ood_loader = get_cifar5(batch_size, split, use_augment, False)
        ood_loader_list['CIFAR5'] = ood_loader

    if dataname == 'CIFAR10':
        model = ENN(resnet20(10)).to(device)
        model.classifier.load_state_dict(torch.load('pretrain/resnet20_cifar10_eval.pt'))

        train_loader, valid_loader, test_loader = get_data(dataname, batch_size, use_augment=use_augment, use_split= False)
        _, _, ood_loader_list['CIFAR100'] = get_data(dataname='CIFAR100', batch_size=batch_size, split=10000, use_augment=False)
        ood_loader_list['LSUN'] = get_lsun(batch_size, split=10000)
        _, ood_loader_list['SVHN'], _ = get_data('SVHN', batch_size=batch_size, split=10000, use_augment=False)


    if dataname == 'SVHN':
        model = ENN(resnet20(10)).to(device)
        model.classifier.load_state_dict(torch.load('pretrain/resnet20_svhn_eval.pt'))

        train_loader, valid_loader, test_loader = get_data(dataname, batch_size, use_augment=use_augment, use_split= False)
        _, _, ood_loader_list['CIFAR10'] = get_data(dataname='CIFAR10', batch_size=batch_size, split=10000, use_augment=False)
        _, _, ood_loader_list['CIFAR100'] = get_data(dataname='CIFAR100', batch_size=batch_size, split=10000, use_augment=False)
        ood_loader_list['LSUN'] = get_lsun(batch_size, split=10000)

    if dataname == 'MNIST':

        model = ENN(lenet5(1)).to(device)
        model.classifier.load_state_dict(torch.load('pretrain/lenet_mnist_eval.pt'))

        train_loader, valid_loader, test_loader = get_data(dataname, batch_size, use_augment=use_augment, use_split= False)
        notmnist_loader = get_notmnist(batch_size, split)
        _, fmnist_loader, _ = get_data('FashionMNIST', batch_size, use_augment=False, use_split=True, split=split)
        ood_loader_list['notMNIST'] = notmnist_loader
        ood_loader_list['FashionMNIST'] = fmnist_loader

    if dataname == 'FashionMNIST':
        model = ENN(resnet20(10, 1)).to(device)
        model.classifier.load_state_dict(torch.load('pretrain/resnet20_fmnist_eval.pt'))

        train_loader, valid_loader, test_loader = get_data(dataname, batch_size, use_augment=use_augment, use_split= False)
        notmnist_loader = get_notmnist(batch_size, split)
        _, mnist_loader, _ = get_data('MNIST', batch_size, use_augment=False, use_split=True, split=split)
        ood_loader_list['notMNIST'] = notmnist_loader
        ood_loader_list['MNIST'] = mnist_loader


    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if not use_pretrain:
        model = pretrain(model, optimizer, train_loader, test_loader, num_classes, epochs=30)
        torch.save(model.classifier.state_dict(), 'pretrain/lenetbn_mnist_eval.pt')

    if dataname in ['MNIST', 'FashionMNIST']:
        netG = Generator_bw().to(device)
        netD = Discriminator_bw().to(device)
    else:
        netG = Generator_color().to(device)
        netD = Discriminator_color().to(device)

    netG.apply(weights_init)
    optimizerG = optim.Adam(netG.parameters(), weight_decay=weight_decay)

    netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=weight_decay)
    fixed_noise = torch.randn(100, nz)

    df_train = pd.DataFrame(
        columns=['epoch', 'batch', 'train_acc', 'train_in_vac', 'g_vac', 'dist', 'loss'])
    df_train_avg = pd.DataFrame(
        columns=['epoch', 'train_acc', 'train_in_vac', 'g_vac', 'dist', 'loss'])

    df_test = pd.DataFrame(
        columns=['epoch', 'idxs_mask', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'bnd_ent_roc', 'bnd_dis_roc'])
    df_test_avg = pd.DataFrame(
        columns=['epoch', 'test_acc', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'bnd_ent_auroc', 'bnd_dis_auroc'])

    df_ood = pd.DataFrame(
        columns=['epoch', 'ood_ent', 'ood_vac', 'ood_dis', 'ood_ent_roc', 'ood_vac_roc'])

    df_ood_avg = pd.DataFrame(
        columns=['epoch', 'ood_ent', 'ood_vac', 'ood_dis', 'ood_ent_auroc', 'ood_vac_auroc'])

    df_ood_list, df_ood_avg_list = {}, {}
    for ood_name in ood_loader_list.keys():
        df_ood_list[ood_name] = df_ood.copy()
        df_ood_avg_list[ood_name] = df_ood_avg.copy()

    total_start = time.time()
    for epoch in range(1, n_epochs + 1):
        print('Epoch:%d' % epoch)

        start = time.time()

        model.train()
        train_loader.dataset.offset = np.random.randint(len(train_loader.dataset))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            onehot = torch.eye(num_classes, device=device)[target]

            for d_time in range(d_iters):
                ###########################
                # (1) Update D network
                ###########################
                optimizerD.zero_grad()

                D_x = netD(data)
                errD_real = D_x.mean()
                (-errD_real).backward()

                noise = torch.randn(data.size(0), nz).to(device)

                fake = netG(noise).detach()
                D_g = netD(fake)

                errD_fake = D_g.mean()
                errD_fake.backward()

                # gradient penalty of wgan-gp

                epsilon = torch.rand(data.size(0), 1, 1, 1).to(device)
                epsilon = epsilon.expand_as(data)

                interpolation = (epsilon * data.data + (1 - epsilon) * fake.data).requires_grad_(True)
                interpolation_logits = netD(interpolation)
                grad_outputs = torch.ones(interpolation_logits.size()).to(device)
                gradients = autograd.grad(outputs=interpolation_logits,
                                          inputs=interpolation,
                                          grad_outputs=grad_outputs,
                                          create_graph=True,
                                          retain_graph=True)[0]

                grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
                grad_penalty.backward()

                optimizerD.step()

            # Get Wasserstein distance
            with torch.no_grad():
                D_x = netD(data)
                errD_real = D_x.mean().item()
                D_g = netD(fake)
                errD_fake = D_g.mean().item()
                dist = errD_real - errD_fake

            ###########################
            # (2) Update G network
            ###########################
            noise = torch.randn(data.size(0), nz).to(device)

            optimizerG.zero_grad()

            fake = netG(noise)
            D_g = netD(fake)
            errG = - D_g.mean()

            if use_G_grad:
                alpha_fake = model(fake)
                s = alpha_fake.sum(1, keepdim=True)
                vac_ood = num_classes / s
                errG = errG + gamma * vac_ood.mean()

            errG.backward()
            optimizerG.step()

            ###########################
            # (3) Update ENN
            ###########################

            for i in range(e_iters):
                model.train()
                model.apply(update_bn_stats)  ## Freeze the BN so that it won't update the statistics.
                # To avoid the early low-quality generated samples destroy the batch-norm statistics
                optimizer.zero_grad()

                ##########################
                # (1) Loss from P real
                ###########################

                alpha = model(data)
                s = alpha.sum(1, keepdim=True)
                # s = torch.sum(alpha, dim=1, keepdim=True)
                p = alpha / s
                vac_in = num_classes / s
                acc_loss = torch.sum((onehot - p) ** 2, dim=1).mean() + \
                           torch.sum(p * (1 - p) / (s + 1), axis=1).mean()

                pred = p.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(target.view_as(pred)).sum().item() / data.size(0)

                loss1 = acc_loss
                loss1.backward()
                optimizer.step()

                ###########################
                # (2) Loss from P fake
                ###########################
                model.apply(freeze_bn_stats)

                optimizer.zero_grad()
                noise = torch.randn(data.size(0), nz).to(device)
                fake = netG(noise).detach()

                alpha_fake = model(fake)
                s = alpha_fake.sum(1, keepdim=True)
                vac_ood = num_classes / s

                loss2 = gamma * vac_ood.mean()
                (-loss2).backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                loss = loss1 - loss2

            df_train.loc[len(df_train)] = [epoch, batch_idx,  accuracy, vac_in.mean().item(), vac_ood.mean().item(), dist, loss.item()]

        df_train_avg.loc[len(df_train_avg)] = df_train[df_train.epoch == epoch].mean()
        df = df_train_avg.tail(1)
        train_log = 'Train:\t\tacc: {:.3f},\t'\
              'loss: {:.4f}\t\t\t'\
              'vac: {:.3f},\t'\
              'g_vac: {:.3f},\t\t\t'\
              'dist: {:.2f},\t'.format(df['train_acc'].iloc[0],
                                        df['loss'].iloc[0],
                                        df['train_in_vac'].iloc[0],
                                        df['g_vac'].iloc[0],
                                        df['dist'].iloc[0]
                                       )

        print(train_log)

        # #######################
        # # 2.  Test in-distribution #
        # #######################

        model.eval()
        model.apply(freeze_bn_stats)

        df_test, df_test_avg, in_score = test_in(model, test_loader, num_classes, df_test, df_test_avg, epoch)
        df = df_test_avg.tail(1)
        test_log ='Test in:\tacc: {:.3f},\t'\
                  'ent: {:.3f}({:.3f}/{:.3f}),\t'\
                  'vac: {:.3f},\t'\
                  'disn: {:.3f}({:.3f}/{:.3f}),\t'\
                  'bnd_auroc: [ent {:.3f}, disn {:.3f}]'.format(df['test_acc'].iloc[0],
                                                                df['in_ent'].iloc[0],
                                                                df['succ_ent'].iloc[0],
                                                                df['fail_ent'].iloc[0],
                                                                df['in_vac'].iloc[0],
                                                                df['in_dis'].iloc[0],
                                                                df['succ_dis'].iloc[0],
                                                                df['fail_dis'].iloc[0],
                                                                df['succ_ent'].iloc[0],
                                                                df['bnd_ent_auroc'].iloc[0],
                                                                df['bnd_dis_auroc'].iloc[0])
        print(test_log)

        # #######################
        # # 3.  Test OOD #
        # #######################
        ood_log_list=[]

        for ood_name in ood_loader_list:
            ood_loader = ood_loader_list[ood_name]
            df_ood = df_ood_list[ood_name]
            df_ood_avg = df_ood_avg_list[ood_name]

            df_ood, df_ood_avg, out_score = test_out(model, ood_loader, num_classes, in_score, df_ood, df_ood_avg, epoch)
            df = df_ood_avg.tail(1)
            ood_log ='Test out:\t{:10s}\tent: {:.3f},\t\t\t' \
                       'vac: {:.3f},\tdisn: {:.3f}\t\t\t' \
                       'ood_auroc: [ent {:.3f},  vac {:.3f}]'.format(ood_name,
                                                                    df['ood_ent'].iloc[0],
                                                                    df['ood_vac'].iloc[0],
                                                                    df['ood_dis'].iloc[0],
                                                                    df['ood_ent_auroc'].iloc[0],
                                                                    df['ood_vac_auroc'].iloc[0])
            ood_log_list.append(ood_log)
            print(ood_log)

            df_ood_list[ood_name] = df_ood
            df_ood_avg_list[ood_name] = df_ood_avg

        done = time.time()

        print('\t\t---- time:{:.3f} per epoch ----\n'.format(done - start))

        if epoch % log_interval == 0:

            torch.save(model.state_dict(), '{}/model_{}.pt'.format(path, epoch))
            # torch.save(netD.state_dict(), '{}/netD_{}.pt'.format(path, epoch))
            # torch.save(netG.state_dict(), '{}/netG_{}.pt'.format(path, epoch))

            with open('{}/log.txt'.format(path), "a") as file:
                print('Epoch:%d' % epoch, file=file)
                print(train_log, file=file)
                print(test_log, file=file)
                for ood_log in ood_log_list:
                    print(ood_log, file=file)

            for df_record in ['df_train', 'df_train_avg', 'df_test','df_test_avg','df_ood_list','df_ood_avg_list']:
                record[df_record] = eval(df_record)
            record['epoch'] = epoch
            with open('{}/record.pt'.format(path), "wb") as file:
                pickle.dump(record, file)

            fake_samples = netG(fixed_noise.to(device))
            vutils.save_image(fake_samples, '{}/gan_samples_{}.png'.format(path, epoch), nrow=10,
                              normalize=True)

    total_time = str(datetime.timedelta(seconds=time.time() - total_start))
    print('----- finished -------- elapsed: %s'%total_time)
    for arg in args:
        print('{}:\t{}'.format(arg, eval(arg)), end=', ')
        record[arg] = eval(arg)
    print('\n')

    return record, model
