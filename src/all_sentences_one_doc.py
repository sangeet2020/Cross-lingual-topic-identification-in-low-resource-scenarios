#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

"""
Combine all sentences of one documnent into separate sentences.
"""

import codecs
import argparse
from collections import OrderedDict
import re

def get_line_number(phrase, file_n):

    occ_list = []
    with open(file_n) as myFile:
        for num, line in enumerate(myFile, 1):
            if phrase in line:
                occ_list.append(num)
    return occ_list

def convert_all_sentences_to_one_doc(idx, docx, file_n, num_lines, args):

    doc_ids = []
    for i in range(num_lines):
        prsnt_idx = idx[i].rsplit("_", 1)[0]
        doc_ids.append(prsnt_idx)
        unique_doc_ids = list(OrderedDict.fromkeys(doc_ids))
    # print("unique doc ids: ", unique_doc_ids)
    # print("Total docs: ", len(unique_doc_ids))

    for i in range(len(unique_doc_ids)):
        pointer = unique_doc_ids[i]
        occ_list = get_line_number(pointer, file_n)
        # print(occ_list)
        for sent in occ_list:
            # print(sent)
            current_sent = docx[sent-1]
            with codecs.open(args.out_dir + '/asr_all_sentn_one_doc.txt', 'a+', 'utf-8') as f:
                f.write(str(current_sent).rstrip('\n'))

        with codecs.open(args.out_dir + '/asr_all_sentn_one_doc.txt', 'a+', 'utf-8') as f:
            f.write('\n')
        f.close()

def main():
    '''main method '''
    args = parse_arguments()
    num_lines = sum(1 for line in open(args.asr_out_idx))
    with open(args.asr_out_idx) as f:
        idx = f.readlines()
    with open(args.asr_out_txt) as f:
        docx = f.readlines()
    file_n = args.asr_out_idx
    convert_all_sentences_to_one_doc(idx, docx, file_n, num_lines, args)


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asr_out_idx", help="path to asr output idx")
    parser.add_argument("asr_out_txt", help="path to asr outpur text")
    parser.add_argument("out_dir", help="path to fid label map")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
