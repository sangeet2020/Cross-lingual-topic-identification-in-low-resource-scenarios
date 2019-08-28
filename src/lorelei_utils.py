#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 14 Nov 2016
# Last modified : 14 Nov 2016

"""
Basic utils for LORELEI evaluations
"""

import os
import sys
import codecs
import pickle
import json
import h5py
from collections import OrderedDict
from copy import deepcopy
import scipy.sparse
import numpy as np
# from misc.io import read_simple_flist


JSON_D = OrderedDict({"DocumentID": '', "PlaceMention": '', "Type": '',
                      "TypeConfidence": ''})
N_STYPES = 11
OOD_LABEL = N_STYPES


MAPS = {"med" : "Medical Assistance",
        "utils": "Utilities, Energy, or Sanitation",
        "water": "Water Supply",
        "shelter": "Shelter",
        "search": "Urgent Rescue",
        "infra" : "Infrastructure",
        "evac": "Evacuation",
        "food": "Food Supply",
        "terrorism": "Terrorism or other Extreme Violence",
        "regimechange": "Elections and Politics",
        "crimeviolence" : "Civil Unrest or Wide-spread Crime",
        "out-of-domain": "out-of-domain"}


def read_simple_flist(fname):
    """ rsf """

    lst = []
    with codecs.open(fname, 'r', 'utf-8') as fpr:
        lst = [line.strip() for line in fpr if line.strip()]
    return lst


def get_training_and_test_data(data, all_fids, train_fids, test_fids):
    """ Get the training and test matrix using the list of fids """

    if data.shape[1] == len(all_fids):
        data = data.T

    train_data = []
    test_data = []
    new_train_fids = []
    new_test_fids = []
    for i, fid in enumerate(all_fids):
        if fid in train_fids:
            train_data.append(data[i, :])
            new_train_fids.append(fid)
        elif fid in test_fids:
            test_data.append(data[i, :])
            new_test_fids.append(fid)
        else:
            # ignore the fid
            pass
    return (np.asarray(train_data), new_train_fids,
            np.asarray(test_data), new_test_fids)


def get_ground_truth_labels(fpaths_flist):
    """ Get multi-label ground truth given a file
    with a list of file paths """

    fid_label_st = pickle.load(open(ETC_D + "fID_label_ST.pkl", "rb"))
    fid_label_io = pickle.load(open(ETC_D + "fID_label_IO.pkl", "rb"))

    fids = []
    mlabels = []

    if isinstance(fpaths_flist, str):
        with open(fpaths_flist, 'r') as fpr:
            for line in fpr:
                fid = os.path.splitext(os.path.basename(line.strip()))[0]
                fids.append(fid)
                try:
                    # out of domain, map it to last label
                    if fid_label_io[fid] == 0:
                        mlabels.append([OOD_LABEL])
                    else:
                        mlabels.append(fid_label_st[fid])
                except KeyError:
                    pass

    else:
        for fid in fpaths_flist:
            fids.append(fid)
            try:
                # out of domain, map it to last label
                if fid_label_io[fid] == 0:
                    mlabels.append([OOD_LABEL])
                else:
                    mlabels.append(fid_label_st[fid])
            except KeyError:
                pass

    return mlabels, fids


def repeat_feats_and_sync_labels(data, mlabels):
    """ Repeat the features corresponding to multiple labels,
    and return the reapeated data, and the labels """

    if len(mlabels) != data.shape[0]:
        data = data.T

    # import pdb
    # pdb.set_trace()
    labels = []
    new_data = []

    for i, mlab in enumerate(mlabels):

        lab_ixs = np.where(mlab == 1)[0]
        if len(lab_ixs) == 1:
            new_data.append(data[i, :])
            labels.append(lab_ixs[0])
        else:
            for lab in lab_ixs:
                labels.append(lab)
                new_data.append(data[i, :])

    return scipy.sparse.csc_matrix(np.asarray(new_data)), np.asarray(labels)


def repeat_data_and_sync_labels(data, fids, mlabels):
    """ Repeat the data (list) corresponding to multiple labels,
    and return the reapeated data, and the labels """

    if len(mlabels) != len(data):
        print("No. of rows in data and labels do not match.")
        sys.exit()

    labels = []
    new_data = []
    new_fids = []

    for i, mlab in enumerate(mlabels):

        lab_ixs = np.where(mlab == 1)[0]
        if len(lab_ixs) == 1:
            new_data.append(data[i])
            new_fids.append(fids[i])
            labels.append(lab_ixs[0])
        else:
            for lab in lab_ixs:
                labels.append(lab)
                new_data.append(data[i])
                new_fids.append(fids[i])

    return np.asarray(new_data), np.asarray(new_fids), np.asarray(labels)


def write_json_file(file_ids, scores, json_file):
    """ Write scores in output JSON file format.

    Parameters:
    -----------
    file_ids (list): list of file IDs \n
    scores (numpy ndarray): no. of fIDs x Labels confidence scores \n
    json_file (str): full path to output json file \n

    """

    sf_labels = []
    with open(ETC_D + "SF_labels.txt", "r") as fpr:
        fpr.readline()
        for line in fpr:
            line = line.strip()
            vals = line.split()
            sf_labels.append(" ".join(vals[:-1]))

    json_list = []
    for i, fid in enumerate(file_ids):

        for j, stype in enumerate(sf_labels):

            # skip out-of-domian
            if j == 0:
                continue

            tmp_d = deepcopy(JSON_D)
            tmp_d["DocumentID"] = fid
            tmp_d["PlaceMention"] = ''
            tmp_d["Type"] = stype
            tmp_d["TypeConfidence"] = scores[i, j]

            json_list.append(tmp_d)

    with codecs.open(json_file, 'w') as fpw:
        json.dump(json_list, fpw, indent=2)

    print(json_file, 'saved.')


def get_ivectors_from_h5(ivecs_h5_file, iter_num=-1, dim='half'):
    """ Load ivectors from h5 file and return them in numpy array
        for a given iter num. If its -1, then the final iteration
        i-vectors are returned. """

    max_iters = int(os.path.splitext(os.path.basename(
        ivecs_h5_file))[0].split("_")[-1][1:])
    print('get_ivectors_from_h5: max_iters:', max_iters)
    ivecs_h5f = h5py.File(ivecs_h5_file, 'r')
    ivecs_h5 = ivecs_h5f.get('ivecs')

    if iter_num == -1:
        ivecs = ivecs_h5.get(str(max_iters))[()]
    else:
        ivecs = ivecs_h5.get(str(iter_num))[()]

    # if ivecs.shape[1] > ivecs.shape[0]:
    ivecs = ivecs.T

    if dim == 'half':
        d = ivecs.shape[1]
        ivecs = ivecs[:, :d//2]

    return ivecs


def load_multi_labels(label_file):
    """ Load labels from file and convert them into multi-label format """

    max_label = 0
    labels = []
    with open(label_file, 'r') as fpr:
        for line in fpr:
            labs = [int(i) for i in line.strip().split(",") if i]
            if np.max(labs) > max_label:
                max_label = np.max(labs)
            labels.append(labs)

    multi_labels = np.zeros(shape=(len(labels), max_label+1), dtype=int)
    for i, labs in enumerate(labels):
        multi_labels[i, np.asarray(labs, dtype=int)] = 1

    return multi_labels


def compute_f1(true_labels, pred_labels, return_pr=False):
    """ Compute F1-score for binary labels where target label is 1

    Args:
    -----
        true_labels (numpy.ndarray): True binary labels
        test_pred (numpy.ndarray): Predicted binary labels
        return_pr (bool): Return precision, and recall along with f1

    Returns:
    --------
        float: F1-score

    """

    prec = 0.
    recl = 0.
    lab_ixs = np.where(true_labels == 1)[0]
    if lab_ixs.any():
        recl = np.mean(pred_labels[lab_ixs])
        prec = (np.where(pred_labels[lab_ixs]==1)[0].shape[0] /
                np.where(pred_labels == 1)[0].shape[0])
        if prec * recl == 0:
            f1_scr = 0.
        else:
            f1_scr = (2. * prec * recl) / (prec + recl)
    else:
        f1_scr = 0.

    if return_pr:
        ret_vals = (prec, recl, f1_scr)
    else:
        ret_vals = f1_scr

    return ret_vals
