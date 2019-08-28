#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

'''
For LEIDOS data, where number of documents are in million, train classifier batchwise
'''

import os
import copy
import pickle
import math
import logging
import argparse
import numpy as np
from time import time
import scipy.io as sio
from scipy.io import mmread
import matplotlib.pyplot as plt
from misc.io import read_simple_flist
from baseline_ENG_SF import conv_to_mlab
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from operator import itemgetter
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, f1_score)

def load_text_batchwise(complete_text_set, chunk_size):

    for i in range(0, len(complete_text_set), chunk_size):
        chunked_data = yield complete_text_set[i:i + chunk_size]


def re_train_classifiers(train_data, binary_labels, alphas, clf, args):
    """ Train 11 one-vs-rest classifiers using best hyper-params on training data """

    # key, value pair, where key is label and value is alpha or C
    for i, alpha in enumerate(alphas):
        curr_clf = clf[i]
        curr_clf.fit(train_data, binary_labels)
        # clf[i].fit(train_data, binary_labels)
        clf[i] = copy.deepcopy(curr_clf)
    return clf


def train_classifiers(train_data, binary_labels, alphas, args):
    """ Train 11 one-vs-rest classifiers using best hyper-params on training data """

    # key, value pair, where key is label and value is alpha or C
    clf = {}
    for i, alpha in enumerate(alphas):
        uni_clf = SGDClassifier(loss='log', max_iter=2, tol=1e-3,
                            n_jobs=args.nj, class_weight='balanced',
                            alpha=alpha)

        uni_clf.fit(train_data, binary_labels)
        clf[i] = copy.deepcopy(uni_clf)
    return clf

def compute_avgAP_scores(clf, test_data_tfidf, labs_test, args):
    """Computer average precision scores using the 20% test data """

    # test_probs = np.zeros(shape=(test_data_tfidf.shape[0], 1), dtype=np.float32)
    avg_pre = {}
    for i in range(len(clf)):
        test_probs = clf[i].predict_proba(test_data_tfidf)[:,1]
        score = average_precision_score(labs_test, test_probs)
        avg_pre[i] = score
    return avg_pre


def get_doc_and_labs(chunk, labs_train, data_train):

    chunked_data = list(itemgetter(*chunk)(data_train))
    # chunked_labs = list(itemgetter(*chunk)(labs_train))
    chunked_labs = labs_train[np.asarray(chunk, dtype=int)]
    return chunked_data, chunked_labs

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def my_smart_chunker(labs_train, chunk_size):

    zeros_list = []
    ones_list = []
    for pos, i in enumerate(labs_train):
        if i == 0:
            zeros_list.append(pos)
            np.append
        else:
            ones_list.append(pos)
    zeros = list(split(zeros_list, chunk_size))
    ones = list(split(ones_list, chunk_size))
    chunks = []
    for chn in range(len(ones)):
        chunks.append(ones[chn] + zeros[chn])
    return chunks



# @profile
def main():
    '''main function starts here'''
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    args = parse_arguments()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    logging.basicConfig(format='%(asctime)s %(message)s',
                        filename=args.out_dir + '/' + 'training_clf_' + str(args.clf_int) + '.log', filemode='a',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    complete_text_labs = read_simple_flist(args.label_file)
    multi_labs = np.zeros(shape=(len(complete_text_labs), args.max_labels), dtype=int)
    lno = 0
    for line in complete_text_labs:
        multi_labs[lno] = conv_to_mlab(complete_text_labs[lno], args.max_labels)
        lno += 1

    # print('number of docs per label:', np.sum(multi_labs, axis=0))

    uni_class_labs = multi_labs[ : , int(args.clf_int)-1]

    alphas = [0.01, 0.1, 0.5, 1.0, 10.0]
    chunk_size = math.floor(len(complete_text_labs)/int(args.batch_size))
    # print('number of chunks:', chunk_size   )
    cvect = pickle.load(open(args.cvect_pkl_file, 'rb'))
    tfidf = pickle.load(open(args.tfidf_pkl_file, 'rb'))

    if os.path.basename(args.labeled_text_path).split(".")[1] == 'pkl':
        # Load complete datast once and keep loading a chunk of it. Do feats extr -> train_clf -> repeat
        complete_text_set = pickle.load(open(args.labeled_text_path, 'rb'))
        doc_len = len(complete_text_set)
    else:
        complete_text_set = read_simple_flist(args.labeled_text_path)
        doc_len = len(complete_text_set)
    # import pdb; pdb.set_trace()
    data_train, data_test, labs_train, labs_test = train_test_split(complete_text_set,
                                                                    uni_class_labs,
                                                                    test_size=0.2,
                                                                    random_state=None,
                                                                    stratify=uni_class_labs)

    # import pdb
    # pdb.set_trace()
    test_data_counts = cvect.transform(data_test)
    test_data_tfidf = tfidf.transform(test_data_counts)
    chunks = my_smart_chunker(labs_train, chunk_size)

    # import pdb
    # pdb.set_trace()

    avg_iter = {}
    for k in range(100):
        batch_avg_pre = {}
        for i in range(len(chunks)):
            stime = time()
            file_tfidf = args.out_dir + "/" + "leidos_" + str(i) + '_' + str(args.batch_size) + "_tfidf.mtx"
            file_labs = args.out_dir + "/" + "leidos_" + str(i) + '_' + str(args.batch_size) + "_labs.pkl"
            if k == 0:
                if os.path.isfile(file_labs):
                    pass
                else:
                    chunked_data, chunked_labs = get_doc_and_labs(chunks[i], labs_train, data_train)
                    # print('get doc and labels:', time() - stime)
                    item_cd = chunked_data
                    item_cd_counts = cvect.transform(item_cd)
                    item_cd_tfidf = tfidf.transform(item_cd_counts)
                    sio.mmwrite(file_tfidf, item_cd_tfidf)
                    pickle.dump(chunked_labs, open(file_labs, 'wb'))

            labels = pickle.load(open(file_labs, 'rb'))
            labels = np.asarray(labels, dtype=np.int32)
            train_data = mmread(file_tfidf).tocsr()
            # print('train_data', train_data.shape, 'labels', labels.sum(), labels.shape)
            if k + i == 0:
                # train classifier from first chunk of data
                clf = train_classifiers(train_data, labels, alphas, args)
            else:
                # now retrain the classifier using new data in every iteration
                clf = re_train_classifiers(train_data, labels, alphas, clf, args)

            avg_pre = compute_avgAP_scores(clf, test_data_tfidf, labs_test, args)
            # print(avg_pre)
            for p in range(len(alphas)):
                batch_avg_pre.setdefault(p, []).append(avg_pre[p])
            # my_acc.append(test_acc)
            # p.append(z)
            # plt.plot(p, my_acc, color='g')
            # z+=1
            # plt.pause(0.05)
            logging.info('%s: %d, %s: %d %s: %d %s: %d %s: %.2f %s %s | %.4f | %.4f | %.4f | %.4f | %.4f |',
                        'Iter no:', k,
                        'Batch',i,
                        'ENG Docs',train_data.shape[0],
                        'Labels',len(labels),
                        'Time', (time() - stime),'sec',
                        'AP Score', avg_pre[0], avg_pre[1], avg_pre[2], avg_pre[3], avg_pre[4])

        for p in range(len(alphas)):
            avg_iter.setdefault(p, []).append(np.mean(batch_avg_pre[p]))

    k = list(range(1, k+2))
    plt.xlabel("iteration no.")
    plt.ylabel("AP score per batch")
    plt.title("training classifier")
    for p, alpha in enumerate(alphas):
        plt.plot(k, avg_iter[p], label="alpha: %s"%alpha)

    plt.legend()
    plt.savefig(args.out_dir+'/'+'avg_AP_score_' + str(args.clf_int) + ' .pdf'.format(2))
    #Finally, save the classifier
    file_clf = args.out_dir + '/' + 'clf_leidos_label_' + str(args.clf_int) + '.pkl'
    pickle.dump(clf, open(file_clf, 'wb'))
    logging.info('%s: %s','Classifier saved to:', file_clf)
    logging.info('%s', '-- Done --')



def parse_arguments():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cvect_pkl_file", help="path to count vectorizer pkl")
    parser.add_argument("tfidf_pkl_file", help="path to tfidf vectorizer pkl")
    parser.add_argument("labeled_text_path", help="path to labeled text data or pkl file")
    parser.add_argument("batch_size", help="no.of docs in one batch")
    parser.add_argument("clf_int", help="You are training 11 clfs. Enter clf no. between 1 to 11")
    parser.add_argument("label_file", help="path for the labels of the labeled text data")
    parser.add_argument("params_file", help="path to best_params file")
    parser.add_argument("out_dir", help="path to save all the classifiers")
    parser.add_argument("-max_labels", default=11, type=int,
                        help="maximum number of labels (themes, situation frames)")
    parser.add_argument("-nj", default=1, type=int, help='number of jobs')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    main()
