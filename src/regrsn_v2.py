#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

'''
Regression version 2. Built in PyTorch.
'''

import os
# import sys
import pickle
import argparse
# import h5py
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, MultiTaskLasso
# import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data
from lorelei_utils import get_ivectors_from_h5


class ParallelDataset(data.Dataset):
    """ Dataset for training transformations from source to target space """

    def __init__(self, il_feats, eng_feats):
        """ Initialization """

        self.il_feats = il_feats
        self.eng_feats = eng_feats

    def __len__(self):
        """ Total number of samples """
        return self.il_feats.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data from each set """
        return (torch.from_numpy(self.il_feats[index, :]).squeeze(),
                torch.from_numpy(self.eng_feats[index, :]).squeeze())


def train_torch_model(eng_train, eng_test, tar_train, tar_test, alphas, cuda, args):
    """ Train torch based regression model """

    train_dset = ParallelDataset(tar_train, eng_train)
    train_loader = data.DataLoader(train_dset, 128, shuffle=True)

    valid_dset = ParallelDataset(tar_test, eng_test)
    valid_loader = data.DataLoader(valid_dset, 128)

    models = []

    errors = []

    dims = [tar_train.shape[1], eng_train.shape[1]]

    for alpha in alphas:

        model = nn.Sequential(nn.Linear(dims[0], dims[1]))

        objective = torch.nn.MSELoss()
        # valid_obj = torch.nn.MSELoss()

        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        device = torch.device("cuda" if cuda else "cpu")
        model.to(device=device)

        lam = nn.Parameter(torch.Tensor([alpha]).to(dtype=torch.float,
                                                    device=device),
                           requires_grad=False)

        for i in range(1, 101):
            bno = 0
            for il_feats, eng_feats in train_loader:
                if cuda:
                    il_feats = il_feats.to(device=device)
                    eng_feats = eng_feats.to(device=device)
                eng_pred = model(il_feats)
                loss = objective(eng_feats, eng_pred)
                for params in model.parameters():
                    loss += (lam * (params ** 2).sum()).item()

                print("\re: {:3d} b: {:4d} L: {:2f}".format(i, bno, loss.item()), end=" ")

                optim.zero_grad()
                loss.backward()
                optim.step()
                bno += 1

            # get err on valid dataset
            valid_err = torch.Tensor([0.]).to(dtype=torch.float, device=device)
            for il_feats, eng_feats in valid_loader:
                if cuda:
                    il_feats = il_feats.to(device=device)
                    eng_feats = eng_feats.to(device=device)
                eng_pred = model(il_feats)
                loss = objective(eng_feats, eng_pred)
                valid_err += loss.detach().item()

            errors.append(valid_err.cpu().item())
            print('valid err: {:.4f}'.format(errors[-1]), end=" ")
            optim.zero_grad()


        if cuda:
            model.cuda()

        models.append(model.cpu())
        sfx = "torch_{:.0e}_xtr_{:d}".format(alpha, i)
        out_file = args.out_dir + sfx + ".pt"
        torch.save(model.state_dict(), out_file)
        print("Model saved", out_file)

    print()

    return errors, models


def prediction(model_selection, e_train, e_test, tar_train, tar_test, alphas, cuda, args):

    print(model_selection)

    if model_selection == 'ridge':
        errors, models = ridgeCV(e_train, e_test, tar_train, tar_test, alphas)
    elif model_selection == 'lasso':
        errors, models = lassoReg(e_train, e_test, tar_train, tar_test, alphas)
    elif model_selection == 'mtlasso':
        errors, models = mtLasso(e_train, e_test, tar_train, tar_test, alphas)
    else:
        errors, models = train_torch_model(e_train, e_test, tar_train, tar_test,
                                           alphas, cuda, args)
    return errors, models


def mean_squared_error(training_pred, testing_pred, e_train, e_test):
    train_error = np.mean((training_pred - e_train) * (training_pred - e_train))
    test_error = np.mean((testing_pred - e_test) * (testing_pred - e_test))
    print('Train Error: {:.4f}'.format(train_error), end=" ")
    print('Test Error: {:.4f}'.format(test_error))
    # print('*************')
    return train_error, test_error


def ridgeCV(e_train, e_test, tar_train, tar_test, alphas):

    print('Ridge Regression')
    errors = []
    models = []
    for alpha in alphas:
        print('alpha = {:.0e}'.format(alpha), end=" ")
        ridge = Ridge(alpha)
        ridge.fit(tar_train, e_train)
        train_pred = ridge.predict(tar_train)
        test_pred = ridge.predict(tar_test)
        trn_err, test_err = mean_squared_error(train_pred, test_pred,
                                               e_train, e_test)
        errors.append([trn_err, test_err])
        models.append(ridge)

    return np.asarray(errors), models


def lassoReg(e_train, e_test, tar_train, tar_test, alphas):

    print('Lasso Regression')
    errors = []
    models = []
    for alpha in alphas:
        lassoReg = Lasso(alpha)
        lassoReg.fit(tar_train, e_train)
        train_pred = lassoReg.predict(tar_train)
        test_pred = lassoReg.predict(tar_test)
        trn_err, test_err = mean_squared_error(train_pred, test_pred,
                                               e_train, e_test)
        errors.append([trn_err, test_err])
        models.append(lassoReg)

    return np.asarray(errors), models


def mtLasso(e_train, e_test, tar_train, tar_test, alphas):

    print('MultiTaskLasso Regression')
    errors = []
    models = []
    for alpha in alphas:
        mtlassoReg = MultiTaskLasso(alpha, max_iter=500)
        mtlassoReg.fit(tar_train, e_train)
        train_pred = mtlassoReg.predict(tar_train)
        test_pred = mtlassoReg.predict(tar_test)
        trn_err, test_err = mean_squared_error(train_pred, test_pred,
                                               e_train, e_test)
        errors.append([trn_err, test_err])
        models.append(mtlassoReg)

    return np.asarray(errors), models


def main():
    """ main """

    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    model_selection = args.model_type

    ext = os.path.basename(args.eng_feats).split(".")[-1]
    if ext == 'h5':

        iter_num = args.iter_num
        if args.iter_num == -1:
            iter_num = int(os.path.splitext(os.path.basename(
                args.eng_feats))[0].split("_")[-1][1:])

        X_eng_total = get_ivectors_from_h5(args.eng_feats, iter_num=iter_num)
        print('Size X_eng_total:', X_eng_total.shape)
        topic_start = np.loadtxt(args.eng_info_file, dtype=int)

        X_tar_para = get_ivectors_from_h5(args.il_feats, iter_num=iter_num)
        X_eng_para = X_eng_total[0 : X_eng_total.shape[0] - topic_start, :]

    else:
        X_eng_total = sio.mmread(args.eng_feats).tocsc()
        print('Size X_eng_total:', X_eng_total.shape)
        topic_start = np.loadtxt(args.eng_info_file, dtype='int')
        X_eng_para = X_eng_total[0 : X_eng_total.shape[0] - topic_start, :]
        X_tar_para = sio.mmread(args.il_feats).tocsc()

    print('Size X_eng_para :', X_eng_para.shape)
    print('Size X_il_para  :', X_tar_para.shape)
    e_train, e_test, tar_train, tar_test = train_test_split(X_eng_para,
                                                            X_tar_para,
                                                            test_size=0.2,
                                                            random_state=42)
    print('e_train: ', e_train.shape, type(e_train),
          'tar_train:', tar_train.shape, type(tar_train))
    print('e_test: ', e_test.shape, 'tar_test:', tar_test.shape)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    errors, models = prediction(model_selection, e_train, e_test, tar_train,
                                tar_test, alphas, args.cuda, args)

    print('errors:', errors.shape)
    if model_selection == 'torch':
        best_ix = np.argmin(errors)
        sfx = model_selection + "_{:.0e}_xtr_{:d}".format(alphas[best_ix],
                                                          iter_num)
        out_file = args.out_dir + sfx + ".pt"
        torch.save(models[best_ix].state_dict(), out_file)

    else:
        best_ix = np.argmin(errors[:, 1])
        sfx = model_selection + "_{:.0e}_xtr_{:d}".format(alphas[best_ix],
                                                          iter_num)
        out_file = args.out_dir + sfx + ".pkl"
        pickle.dump(models[best_ix], open(out_file, 'wb'))

    np.savetxt(args.out_dir + 'errors_' + sfx + '.txt', errors)

    print("Model saved", out_file)

    # plt.figure(1)
    # plt.plot(alphas, errors[:, 1], 'C0.-', label='Test Error')
    # plt.plot(alphas, errors[:, 0], 'C1.-', label='Train Error')
    # plt.suptitle(model_selection.title() + ' Regression', fontsize=16)
    # plt.xlabel('Regularization coefficient alpha')
    # plt.ylabel('MSE')
    # plt.grid()
    # plt.legend()
    # plt.show()

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_type", choices=['ridge', 'lasso', 'mtlasso', 'torch'])
    parser.add_argument("il_feats", help="path to il feats file")
    parser.add_argument("eng_feats", help="path to eng feats file")
    parser.add_argument("eng_info_file", help="path to info reg. labelled data")
    parser.add_argument("out_dir", help="output dir")
    parser.add_argument("-iter_num", default=-1, type=int, help='xtr iter num')
    parser.add_argument("--nocuda", action='store_true',
                        help='Do not use GPU (default: %(default)s)')

    args = parser.parse_args()

    args.cuda = not args.nocuda and torch.cuda.is_available()

    return args

if __name__ == "__main__":

    main()
