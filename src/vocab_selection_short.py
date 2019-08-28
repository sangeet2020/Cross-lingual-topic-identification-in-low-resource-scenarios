#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

'''
Select top vocabs/tokens from each label
'''

import re
import codecs
import numpy as np
import collections
from time import time
import scipy.io as sio
from collections import Counter
import argparse
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


def conv_to_mlab(label_str, max_labels):
    """ Convert to multi label format """

    lab_arr = np.zeros(shape=(max_labels,), dtype=int)
    labs = label_str.split(",")
    for lab in labs:
        lab_arr[int(lab)] = 1
    return lab_arr

def fetch_vocabs(score_dic_complete_vocab, ngram_counts, N, themes, ngram_dic):
    num_labs = score_dic_complete_vocab.shape[0]
    print(num_labs)
    top_vocab_list = {}
    for i in range(num_labs):

        score_ith_row = score_dic_complete_vocab[i].tolist()

        score_ith_row_dic = dict(enumerate(score_ith_row, 0))
#         print('ith row dict:', score_ith_row_dic)

        N_top_scores = Nmaxelements(score_ith_row, N)
        N_top_scores = np.unique(np.array(N_top_scores)).tolist()
#         print('Top 2 scores:', N_top_scores)
        vocab_index_list = []

        for score in N_top_scores:

            for index, vocab_score in score_ith_row_dic.items():

                if score == vocab_score:
                    vocab_index_list.append(index)
                    top_vocab_list.setdefault(
                        themes[i], []).append(ngram_dic.get(index))
#                     top_vocab_list.append(ngram_dic.get(index))


    return top_vocab_list


def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0.

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]

        list1.remove(max1)
        final_list.append(max1)
    return final_list


def count_vocabs(ngrams, vocab):
    count_values = ngrams.toarray().sum(axis=0)
    # print(count_values)
    ngram_counts = {}
    for ng_count, ng_text in sorted([(count_values[i], k) for k, i in vocab.items()], reverse=False):
        #     print(ng_count, ng_text)
        ngram_counts[ng_text] = ng_count
#         score_dic_complete_vocab[ng_text] = []
    return ngram_counts


def main():
    """ main function starts here"""

    args = parse_arguments()
    labeled_text_f = args.labeled_text_f
    labs_f = args.labs_f
    themes_f = args.themes_f

    text = []

    with codecs.open(labeled_text_f, 'r', 'utf-8') as f:
    #     texts = [line.rstrip('\n') for line in f]
        for line in f:
            line = line.rstrip('\n')
            # line = re.sub(r'\d+', '', line) # remove numbers
            line = re.sub(r'[^\w\s]','',line)
            line = line.strip()
            line = re.sub("([^\x00-\x7F])+"," ",line) #removes chinese characters
            # line = ' '.join(word for word in line.split(' ')
            #          if not word.startswith('http'))
            text.append(line)

    with codecs.open(labs_f, 'r', 'utf-8') as f:
        labs = [line.rstrip('\n') for line in f]

    with codecs.open(themes_f, 'r', 'utf-8') as f:
        themes = [line.rstrip('\n') for line in f]

    print('No of docs:', len(text), ', No of labs:', len(labs))
    ng = args.ng
    ana = args.ana
    mdf = args.mdf
    c_vec = CountVectorizer(ngram_range=(ng, ng),
                            strip_accents='unicode',
                            min_df=mdf, analyzer=ana,
                            max_features=None)

    ngrams = c_vec.fit_transform(text)
    vocab = c_vec.vocabulary_
    print('Vocab size:', len(vocab))
    # vocab = c_vec.get_feature_names()


    theme_int = []
    for i, thm in enumerate(themes):
        theme_int.append(i)

    print(themes)
    score_dic_complete_vocab = {}


    multi_labs = np.zeros(shape=(len(labs), len(themes)), dtype=int)
    lno = 0
    for line in labs:
        multi_labs[lno] = conv_to_mlab(labs[lno], len(themes))
        lno += 1

    multi_labs = multi_labs.T

    ngram_counts = count_vocabs(ngrams, vocab)
    # print(ngram_counts)
    ngram_keys_list = list(ngram_counts.keys())
    # print(ngram_keys_list)
    ngram_dic = {}
    for i, key in enumerate(ngram_keys_list):
    #     print(key)
        ngram_dic[i] = key
    # print(ngram_dic)

    print('Count Vectorizer dense matrix shape', ngrams.toarray().shape)
    # print('Count Vectorizer dense matrix',ngrams.toarray())

    lab_by_vocab = np.matmul(multi_labs, ngrams.toarray())
    print('Labels by vocabulary matrix',lab_by_vocab)
    score_lab_by_vocab = np.divide(lab_by_vocab,lab_by_vocab.sum(axis=0), dtype=np.float32)
    print('Final score matrix',score_lab_by_vocab)

    N=int(args.N)
    vocab_list = fetch_vocabs(score_lab_by_vocab, vocab, N, themes, ngram_dic)
    # print('top theme specific words',vocab_list)
    for k,v in vocab_list.items():
        print(k,v)

    for i in range(score_lab_by_vocab.shape[0]):
        k = score_lab_by_vocab[i]
        print(Counter(k).get(1.0))

    # make list of top vocabs.
    top_list = []
    for thm, lst in vocab_list.items():
        for v in lst:
            top_list.append(v)

    top_list = np.unique(top_list).tolist()
    out_dir = args.out_dir
    with codecs.open(out_dir + '/top_vocab_list_' + ana +  '_mdf_'+str(mdf)+'_ng_'+str(ng)+ '_N_'+ str(N) +'.txt','w','utf-8') as f:
        f.write('\n'.join(top_list))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labeled_text_f', help='path to labeled text file')
    parser.add_argument('labs_f', help='path to labeled text label file')
    parser.add_argument('themes_f', help='path to thmemes txt file')
    parser.add_argument('N', help='number of vocabs to be generated from each label')
    parser.add_argument('out_dir', help='out directory to save generated vocabs')
    parser.add_argument('-mdf', default=3, type=int, help='min doc frquency')
    parser.add_argument('-ng', default=3, type=int, choices=[1,2,3], help='ngram tokens')
    parser.add_argument('-ana', default='char_wb', choices=['word', 'char_wb'], help='analyzer word or char_wb')

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    main()
