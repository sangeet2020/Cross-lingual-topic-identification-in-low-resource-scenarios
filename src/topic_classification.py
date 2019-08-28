#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#for each label we have the best alpha which is to be read out form the 
#best_params.txt. For each label take the alpha and train the classifier. 
#Once training is done for that label predict the class for each document present 
# in the eng.tx
#Output = [prob_not_in_class, prob_class] - for each doc


"""
Topic classification for unlabelled text data using 
best alpha from baseline_ENG_SF.py
"""


import os
import sys
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, balanced_accuracy_score, make_scorer,
                             classification_report)
from sklearn.metrics import average_precision_score as aps
from scipy.io import mminfo,mmread
from pprint import pprint
from misc.io import read_simple_flist

np.set_printoptions(precision=2)


def compute_f1(test_pred, test_labels):
    """ Compute F1-score for label 1 """

    lab_ixs = np.where(test_labels == 1)[0]
    recl = np.mean(test_pred[lab_ixs])
    prec = (np.where(test_pred[lab_ixs]==1)[0].shape[0] /
            np.where(test_pred == 1)[0].shape[0])

    return 2. / ((1./prec) + (1./recl))


def conv_to_mlab(label_str, max_labels):
    """ Convert o multi label format """

    lab_arr = np.zeros(shape=(max_labels,), dtype=int)
    labs = label_str.split(",")
    for lab in labs:
        lab_arr[int(lab)] = 1
    return lab_arr

def grid_cv(data, multi_labs, best_alphas, labeled_doc_start, args):
    """ Run grid search over hyper params and do cv """

    best_scores = {'lr':{}, 'nb':{}}
    
    cvect = CountVectorizer(analyzer=args.ana, ngram_range=(args.ng, args.ng),
                            strip_accents='unicode', min_df=args.mdf)
    mnb = MultinomialNB()

    tfidf = TfidfVectorizer(analyzer=args.ana, ngram_range=(args.ng, args.ng),
                            strip_accents='unicode',
                            min_df=args.mdf, norm='l2', use_idf=True,
                            smooth_idf=True)



    p = labeled_doc_start
    r = data.shape[0]
    test_data = data[0: r-p, :]
    train_data = data[r-p : r, :]
    print('Train data shape:', train_data.shape)    
    final_prob = np.zeros(shape=(test_data.shape[0],len(best_alphas)),
                          dtype=np.float32)
    #print(final_prob.shape)
    for i in range(len(best_alphas)):

        lab_ixs = np.where(multi_labs[:, i] == 1)[0]
        ovr_labs = np.zeros(shape=(multi_labs.shape[0],), dtype=int)
        ovr_labs[lab_ixs] = 1
        #print(ovr_labs.shape)
        #exit()
        clf = LogisticRegression(solver='liblinear', max_iter=3000, n_jobs=1,
                                 class_weight='balanced', C=best_alphas[i])

        #print('Labels shape:', ovr_labs.shape)
        #print(test_data.shape[0])
        clf.fit(train_data, ovr_labs)
        final_prob[:, i] = clf.predict_proba(test_data)[:,1]
    np.save(args.out_dir + 'final_test_prob', final_prob)


def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    #labels = read_simple_flist(args.params_f)  # read all labels into list
    labels = [] 
    best_alphas = {}
    with open(args.params_f, 'r') as fpr:

        for line in fpr:
            if line.startswith("#"):
                continue
    
            best_alphas[int(line.split(",")[0])] = float(line.split(",")[-2])

    #pprint(best_alphas)
    #print('Len of best_alphas:',len(best_alphas))
    # data = mmread(args.mtx_f)
    # data = data.tocsr()
    data = mmread(args.mtx_f)
    data = data.tocsr()
    #Reading labels 
    labels = read_simple_flist(args.label_f)
    multi_labs = np.zeros(shape=(len(labels), args.max_labels), dtype=int)
    lno = 0
    # with open(args.text_f, 'r') as fpr:
    for line in labels:
            # data.append(line.strip())
            # convert the labels in corresponding line to multi_lab format
            multi_labs[lno] = conv_to_mlab(labels[lno], args.max_labels)
            lno += 1
    #print(multi_labs)
    print('Docs:', data.shape[0], 'Labels:', multi_labs.shape)
    labled_doc_start = np.loadtxt(args.labeled_data_f, dtype = 'int')    
    grid_cv(data, multi_labs, best_alphas, labled_doc_start, args)


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mtx_f", help="path to english_tfidf mtx file")
    parser.add_argument("label_f", help="path for the label or theme file")
    parser.add_argument("params_f", help="path to best_params file")
    parser.add_argument("out_dir", help="path to output dir")
    parser.add_argument("labeled_data_f", help="path to labelled data info file")
    parser.add_argument("-max_labels", default=11, type=int,
                        help='max number of unique labels')
    parser.add_argument("-ana", default='char_wb', choices=['word', 'char_wb'],
                        help='analyzer for count vect')
    parser.add_argument("-ng", type=int, default=3, choices=[1, 2, 3],
                        help='ngram tokens')
    parser.add_argument('-mdf', type=int, default=2, help='min doc freq.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
