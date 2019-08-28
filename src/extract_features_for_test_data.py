#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

"""
Extract features (counts, tfidf, log_probs) for test data, given the cvect, tfidf pkl file
"""

import os
import pickle
import argparse
import scipy.io as sio
from misc.io import read_simple_flist


def main():
    """ main method """

    args = parse_arguments()

    cvect = pickle.load(open(args.cvect_pkl_file, 'rb'))
    tfidf = pickle.load(open(args.tfidf_pkl_file, 'rb'))

    if os.path.basename(args.in_test_file).split(".")[1] == 'pkl':
        test_docs = pickle.load(open(args.in_test_file, 'rb'))
        # test_docs = test_docs[:20000]
    else:
        test_docs = read_simple_flist(args.in_test_file)

    print('number of test docs:', len(test_docs))

    test_counts = cvect.transform(test_docs)
    test_tfidf = tfidf.transform(test_counts)

    print('test counts:', test_counts.shape)
    print('test tfidf:', test_tfidf.shape)

    sio.mmwrite(args.out_file_basename + "_counts.mtx", test_counts)
    sio.mmwrite(args.out_file_basename + "_tfidf.mtx", test_tfidf)

    print('-- Done --')

def parse_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cvect_pkl_file", help="path to count vectorizer pkl")
    parser.add_argument("tfidf_pkl_file", help="path to tfidf vectorizer pkl")
    parser.add_argument("in_test_file", help="IL test documents (unseq)")
    parser.add_argument("out_file_basename",
                        help="path to out file base name (eg: IL9_test")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    main()
