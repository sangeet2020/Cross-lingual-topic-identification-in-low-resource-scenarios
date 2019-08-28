#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

'''
Perform regression (Ridge, Lasso, MultiTaskLasso) and save the best model. 
'''

import os
import sys
import argparse
import pickle
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import LassoCV, Lasso, MultiTaskLasso
import matplotlib.pyplot as plt
from lorelei_utils import get_ivectors_from_h5


def prediction(model_selection, e_train, e_test, tar_train, tar_test, alphas, args):

    print(model_selection)

    if model_selection == 'ridge':
        errors = ridgeCV(e_train, e_test, tar_train, tar_test, alphas, args)
    elif model_selection == 'lasso':
        errors = lassoReg(e_train, e_test, tar_train, tar_test, alphas)
    elif model_selection == 'mtlasso':
        errors = mtLasso(e_train, e_test, tar_train, tar_test, alphas)
    return errors


def mean_squared_error(training_pred, testing_pred, e_train, e_test):
    train_error = np.mean((training_pred - e_train) * (training_pred - e_train))
    test_error = np.mean((testing_pred - e_test) * (testing_pred - e_test))
    print('Train Error:', train_error, 'Test Error:', test_error)
    print('*************')
    return train_error, test_error


def ridgeCV(e_train, e_test, tar_train, tar_test, alphas, args):

    errors = []
    t_err = []
    i = 0
    for alpha in alphas:

        print('alpha = ', alpha)
        ridge = Ridge(alpha)
        ridge.fit(tar_train, e_train)
        train_pred = ridge.predict(tar_train)
        test_pred = ridge.predict(tar_test)
        print('Ridge Regression')
        trn_err, test_err = mean_squared_error(train_pred, test_pred,
                                               e_train, e_test)
        errors.append([trn_err, test_err])
        t_err.append(errors[i][1])
        #print('Currently the least test error is: ', min(t_err) )
        if (i > 0):
            if(smallest_err == min(t_err)):
                print('Error did not change')
            else:
                print('Least error changed to: ',   min(t_err))
                least_error_model = args.out_dir + 'least_test_error_model' +  '.pkl'
                pickle.dump(ridge, open(least_error_model, 'wb'))
        else:
                print('Currently the least test error is: ', min(t_err))
                least_error_model = args.out_dir + 'least_test_error_model' +  '.pkl'
                pickle.dump(ridge, open(least_error_model, 'wb'))
        i += 1
        smallest_err = min(t_err)

    return np.asarray(errors)



def lassoReg(e_train, e_test, tar_train, tar_test, alphas):

    errors = []
    for alpha in alphas:
        lassoReg = Lasso(alpha)
        lassoReg.fit(tar_train, e_train)
        train_pred = lassoReg.predict(tar_train)
        test_pred = lassoReg.predict(tar_test)
        print('Lasso Regression')
        trn_err, test_err = mean_squared_error(train_pred, test_pred,
                                               e_train, e_test)
        errors.append([trn_err, test_err])

        # Save the only model which has the least error (do this for all regressions using pkl.dump)
    return np.asarray(errors)


def mtLasso(e_train, e_test, tar_train, tar_test, alphas):

    errors = []
    for alpha in alphas:
        mtlassoReg = MultiTaskLasso(alpha, max_iter=100)
        mtlassoReg.fit(tar_train,e_train)
        train_pred = mtlassoReg.predict(tar_train)
        test_pred = mtlassoReg.predict(tar_test)
        print('MultiTaskLasso Regression')
        trn_err, test_err = mean_squared_error(train_pred, test_pred,
                                               e_train, e_test)
        errors.append([trn_err, test_err])

    return np.asarray(errors)


def main():
    """ main """

    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    model_selection = args.model_type

    # check file extension, if its h5, use get_ivectors_from_h5 from lorelei_utils.py
    if os.path.basename(args.il_feats).split(".")[1] == 'h5':
        print('iVectors feats file loaded from '+ args.il_feats)
        X_tar_para = get_ivectors_from_h5(args.il_feats) # will return numpy array
        X_eng_total = get_ivectors_from_h5(args.eng_feats)
        print('Size X_tar_para', X_tar_para.shape, 'Size X_eng_total', X_eng_total.shape)
        # X_tar_para = np.load(X_tar_para)
        # X_eng_total = np.load(X_eng_para)
        topic_start = np.loadtxt(args.eng_info_file, dtype='int')
        X_eng_para = X_eng_total[0 : X_eng_total.shape[0] - topic_start , :]
        print('done')
        # exit()
    else:
        X_eng_total = sio.mmread(args.eng_feats).tocsc()
        print('Size X_eng_total: ', X_eng_total.shape)
        topic_start = np.loadtxt(args.eng_info_file, dtype='int')
        X_eng_para = X_eng_total[0 : X_eng_total.shape[0] - topic_start, :]
        X_tar_para = sio.mmread(args.il_feats).tocsc()


    print('Size X_eng_para: ', X_eng_para.shape, type(X_eng_para))
    print('Size X_il_para:', X_tar_para.shape, type(X_tar_para))
    e_train, e_test, tar_train, tar_test = train_test_split(X_eng_para,
                                                            X_tar_para,
                                                            test_size=0.2,
                                                            random_state=42)
    print('e_train: ', e_train.shape, type(e_train),
          'tar_train:', tar_train.shape, type(tar_train))
    print('e_test: ', e_test.shape, 'tar_test:', tar_test.shape)

    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
    model_train_error = []
    model_test_error = []

    errors = prediction(model_selection, e_train, e_test, tar_train,
                        tar_test, alphas, args)

    np.savetxt(args.out_dir + 'errors_' + model_selection, errors)
    np.savetxt(args.out_dir + 'alphas_' + model_selection, alphas)

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
    parser.add_argument("model_type", choices=['ridge', 'lasso', 'mtlasso'])
    parser.add_argument("il_feats", help="path to il feats file")
    parser.add_argument("eng_feats", help="path to eng feats file")
    parser.add_argument("eng_info_file", help="path to info reg. labelled data")
    parser.add_argument("out_dir", help="output dir")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    main()
