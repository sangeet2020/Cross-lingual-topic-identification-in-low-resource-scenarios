#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

"""
From the computed probablities for each of the english
documents (using topic_classification.py), find
the documents which are theme related.
set prob_thresdhold = 0.7

input:
>>  Read final_test_prob.npy
    Set threshold = 0.7
>>  Identify indices of document whose
    prob of belonging to that class is
    equal to set thrshold (0.7) or greater.

Output:
>>  Indices of docs with prob equal or greater
    set the threshold value.
"""

import os
import sys
import argparse
import numpy as np

def main():
    args = parse_arguments()
    probabs = np.load(args.test_prop_f)
    tr_probabs = np.unique(np.where(probabs > args.threshold)[0])
    np.savetxt(args.out_dir + 'tr_indices.txt', tr_probabs,fmt='%d')



def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_prop_f", help="path to final test prop numpy file")
    parser.add_argument("out_dir", help="output dir")
    parser.add_argument('-threshold', type=float, default=0.7, help='threshold prob of a doc belonging to that class ')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    main()
