#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in


"""
Topic classification for unlabelled text data using best alpha from baseline_ENG_SF.py
"""


# import os
# import sys
import copy
import pickle
import argparse
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import (f1_score, balanced_accuracy_score, make_scorer,
#                             classification_report)
# from sklearn.metrics import average_precision_score as aps
from scipy.io import mmread
from baseline_ENG_SF import conv_to_mlab
from misc.io import read_simple_flist

np.set_printoptions(precision=2)


def train_and_classify(train_data, multi_labs, test_data, best_alphas, args):
    """ Train 11 one-vs-rest classifiers using best hyper-params on training data
    and predict label probabilities for test data """


    final_prob = np.zeros(shape=(test_data.shape[0], len(best_alphas)),
                          dtype=np.float32)

    final_clfs = {}

    # key, value pair, where key is label and value is alpha or C
    for i, alpha in best_alphas.items():

        binary_labels = multi_labs[:, i]
        clf = LogisticRegression(solver='liblinear', max_iter=3000,
                                 n_jobs=args.nj, class_weight='balanced',
                                 C=alpha)

        clf.fit(train_data, binary_labels)
        final_prob[:, i] = clf.predict_proba(test_data)[:, 1]

        final_clfs[i] = copy.deepcopy(clf)

    # saving scores / probs files
    np.save(args.out_scores_file, final_prob)
    print("Predictions saved to:", args.out_scores_file)

    # saving classifiers
    pickle.dump(final_clfs, open(args.out_clf_file, 'wb'))
    print("Classifiers saved to:", args.out_clf_file)


def main():
    """ main method """
    args = parse_arguments()

    # Load best alphas per label
    labels = []
    best_alphas = {}
    with open(args.params_f, 'r') as fpr:
        for line in fpr:
            if line.startswith("#"):
                continue
            best_alphas[int(line.split(",")[0])] = float(line.split(",")[-2])

    # Load ENG features
    print("Loading:", args.eng_mtx_f)
    eng_feats = mmread(args.eng_mtx_f).tocsr()

    # Reading labels
    labels = read_simple_flist(args.label_f)
    # Convert them to multi_labels format
    multi_labs = np.zeros(shape=(len(labels), args.max_labels), dtype=int)
    lno = 0
    for line in labels:
        multi_labs[lno] = conv_to_mlab(labels[lno], args.max_labels)
        lno += 1
    print('ENG Docs:', eng_feats.shape[0], 'Labels:', multi_labs.shape)

    p = np.loadtxt(args.labeled_data_f, dtype='int')
    r = eng_feats.shape[0]

    test_data = eng_feats[0: r-p, :]
    train_data = eng_feats[r-p : r, :]
    print('Train data shape:', train_data.shape)
    print('Test data shape:', test_data.shape)

    train_and_classify(train_data, multi_labs, test_data, best_alphas, args)


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eng_mtx_f", help="path to english_tfidf mtx / npy file")
    parser.add_argument("label_f", help="path for the label or theme file")
    parser.add_argument("params_f", help="path to best_params file")
    parser.add_argument("labeled_data_f", help="path to labelled data info file")
    parser.add_argument("out_scores_file", help="path to output file")
    parser.add_argument("out_clf_file", help="path to save all the classifiers")
    parser.add_argument("-max_labels", default=11, type=int,
                        help="maximum number of labels (themes, situation frames)")
    parser.add_argument("-nj", default=1, type=int, help='number of jobs')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
