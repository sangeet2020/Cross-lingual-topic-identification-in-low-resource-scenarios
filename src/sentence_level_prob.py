#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

"""
Allocate documnet level score to all the sentences in that document.
"""


import codecs
import argparse
import json
import numpy as np
from collections import OrderedDict



def get_line_number(phrase, file_n):
    occ_list = []
    with open(file_n) as myFile:
        for num, line in enumerate(myFile, 1):
            if phrase in line:
                occ_list.append(num)
    return occ_list

def convert_all_sentences_to_one_doc(idx, doc_scores, num_lines, file_n, k, count, args):

    doc_ids = []
    sentn_level_scores = [None] * (len(idx))
    for i in range(num_lines):
        prsnt_idx = idx[i].rsplit("_", 1)[0]
        doc_ids.append(prsnt_idx)
        unique_doc_ids = list(OrderedDict.fromkeys(doc_ids))
    # print("unique doc ids: ", unique_doc_ids)
    print("Total docs: ", len(unique_doc_ids))

    for i in range(len(unique_doc_ids)):
        pointer = unique_doc_ids[i]
        occ_list = get_line_number(pointer, file_n)
        # print(occ_list)
        for idx in occ_list:
            doc_level_current_sentn_score = fetch_current_doc_score(doc_scores, i).strip('\n')
            sentn_level_scores[idx-1] = doc_level_current_sentn_score
            # print(doc_level_current_sentn_score)

    for one_score in sentn_level_scores:
        # print(one_score)
        with codecs.open(args.out_dir + 'sentn_level_out_scores.txt', 'a+', 'utf-8') as f:
            f.write(one_score + '\n')
    f.close()

    return


def calc_avg_score(doc_level_current_sentn_score, sentn_level_current_sentn_score):

    doc_score = doc_level_current_sentn_score.split()
    doc_score = list(map(float, doc_score))
    sentn_score = sentn_level_current_sentn_score.split()
    sentn_score = list(map(float, sentn_score))
    ip = [doc_score , sentn_score]

    # print(doc_score)
    # print(sentn_score)
    avg_score = np.mean(ip, axis=0)
    avg_score = np.array_str(avg_score).lstrip('[').rstrip(']').replace('\n', ' ')
    # print(avg_score.replace('\n', ''))
    return avg_score



def fetch_current_doc_score(doc_scores, i):
    doc_level_current_sentn_score = doc_scores[i]
    return doc_level_current_sentn_score


def pick_best_between_sentnLevel_docLevel_scores(doc_level_current_sentn_score, sentn_level_current_sentn_score, curr_idx, gt_dict, count):
    # Perform check if sentence level score is better or doc level score is better
    doc_score = doc_level_current_sentn_score.split()
    sentn_score = sentn_level_current_sentn_score.split()
    current_idx = curr_idx.strip('\n')
    mlab_ixs = np.asarray(gt_dict.get(current_idx)).astype(int)
    idx_len = len(mlab_ixs)
    # print(mlab_ixs[0])
    # print(doc_score[mlab_ixs[0]-1])
    # print(sentn_score[mlab_ixs[0]-1])
    # exit()
    # print(idx_len)
    for i in range(idx_len):
        if (doc_score[mlab_ixs[i]-1] < sentn_score[mlab_ixs[i]-1]):
            print(curr_idx.strip('\n'),'Doc_Score ', 'Sentn_Score', doc_score[mlab_ixs[0]-1], sentn_score[mlab_ixs[0]-1])
            # print("Doc_id: ", curr_idx.strip('\n'))
            count += 1
            best_score = sentn_level_current_sentn_score
            break
        else:
            best_score = doc_level_current_sentn_score
            pass

    return best_score,count

def main():
    '''main method '''
    k = 0
    args = parse_arguments()
    num_lines = sum(1 for line in open(args.asr_out_idx))
    with open(args.asr_out_idx) as f:
        idx = f.readlines()
    with open(args.doc_level_scores) as f:
        doc_scores = f.readlines()
    file_n = args.asr_out_idx
    count = 0
    convert_all_sentences_to_one_doc(idx, doc_scores, num_lines, file_n, k, count, args)



def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asr_out_idx", help="path to sentence level asr output idx")
    parser.add_argument("doc_level_scores", help="path to doc level out scores")
    # parser.add_argument("sentn_level_scores", help="path to sentence level out scores")
    # parser.add_argument("ground_truth_file", help="path to ground truth.json")
    parser.add_argument("out_dir", help="path to fid label map")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
