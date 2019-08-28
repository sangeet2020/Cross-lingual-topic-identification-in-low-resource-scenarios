#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

'''
Given English text, remove ilegal characters like characters from other languages
'''

import os
import sys
import codecs
import pickle
import argparse
import numpy
import numpy as np
import scipy.io as sio
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from  langdetect import detect
from sklearn.feature_extraction.text import TfidfTransformer

def remove_illegal_words(line, i):

    clean_line = []
    line = line.split()
    # line = [item for item in line if not item.isdigit()]
    # print("Cleaning text docs")
    print('Doc number=', i)
    for word in line:
        # print(word)
        code = ord(word[0])
        maxi = int('97F', 16)
        mini = int('900', 16)

        try:
            if(mini <= code <= maxi):
                clean_line.append(word)
            # word_lang = detect(str(word))
            # if word_lang != 'hi':
            else:
                pass
        except:
            pass
    clean_line = " ".join(clean_line)
    # print(clean_line)

    return clean_line


def main():

    args = parse_arguments()

    unlabelled_text_path = args.unlabelled_text_path
    out_dir = args.out_dir
    i = 1
    with codecs.open(unlabelled_text_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            clean_text = remove_illegal_words(line,i)
            i=i+1
            with codecs.open(out_dir + 'clean_text.txt', 'a', 'utf-8') as f:
                f.write(clean_text+'\n')

    print('DONE')
    f.close()


def parse_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unlabelled_text_path",
                        help="path to parallel hindi text file")
    parser.add_argument("out_dir", help="path to save clean hindi text")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    main()
