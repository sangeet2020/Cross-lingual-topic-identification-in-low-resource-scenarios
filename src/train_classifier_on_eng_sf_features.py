#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in


#for each label we have the best alpha which is to be read out form the
#best_params.txt. For each label take the alpha and train the classifier.
#Once training is done for that label predict the class for each document present
# in the eng.txt.
#Output = [prob_not_in_class, prob_class] - for each doc


"""
Topic classification for unlabelled text data using
best alpha from baseline_ENG_SF.py
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

from misc.io import read_simple_flist
from baseline_ENG_SF import conv_to_mlab
np.set_printoptions(precision=2)


def train_classifiers(train_data, multi_labs, best_alphas, args):
    """ Train 11 one-vs-rest classifiers using best hyper-params on training data """

    final_clfs = {}

    # key, value pair, where key is label and value is alpha or C
    for i, alpha in best_alphas.items():

        binary_labels = multi_labs[:, i]
        clf = LogisticRegression(solver='liblinear', max_iter=3000,
                                 n_jobs=args.nj, class_weight='balanced',
                                 C=alpha)

        clf.fit(train_data, binary_labels)
        final_clfs[i] = copy.deepcopy(clf)

    # saving classifiers
    pickle.dump(final_clfs, open(args.out_clf_file, 'wb'))
    print("Classifiers saved to:", args.out_clf_file)


def main():
    """ main method """
    args = parse_arguments()

    # Load best alphas per label
    labels = []
    best_alphas = {}
    with open(args.params_file, 'r') as fpr:
        for line in fpr:
            if line.startswith("#"):
                continue
            best_alphas[int(line.split(",")[0])] = float(line.split(",")[-2])

    # Load ENG features
    print("Loading:", args.eng_feats_file)
    eng_feats = mmread(args.eng_feats_file).tocsr()

    # Reading labels
    labels = read_simple_flist(args.label_file)
    # Convert them to multi_labels format
    multi_labs = np.zeros(shape=(len(labels), args.max_labels), dtype=int)
    lno = 0
    for line in labels:
        multi_labs[lno] = conv_to_mlab(labels[lno], args.max_labels)
        lno += 1
    print('ENG Docs:', eng_feats.shape[0], 'Labels:', multi_labs.shape)

    p = np.loadtxt(args.labeled_data_info_file, dtype='int')
    r = eng_feats.shape[0]
    train_data = eng_feats[r-p : r, :]
    print('Train data shape:', train_data.shape)

    train_classifiers(train_data, multi_labs, best_alphas, args)


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eng_feats_file", help="path to english_tfidf mtx / npy / h5 file")
    parser.add_argument("label_file", help="path for the sf.themes file")
    parser.add_argument("params_file", help="path to best_params file")
    parser.add_argument("labeled_data_info_file", help="path to labelled data info file")
    parser.add_argument("out_clf_file", help="path to save all the classifiers")
    parser.add_argument("-max_labels", default=11, type=int,
                        help="maximum number of labels (themes, situation frames)")
    parser.add_argument("-nj", default=1, type=int, help='number of jobs')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
