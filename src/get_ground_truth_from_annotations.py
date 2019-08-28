#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : xsagar00
# e-mail : 15uec053[at]lnmiit[dot]ac[dot]in

"""
Prepare ground truth
1. in-domain vs out-of-domain
2. multi-label annotation
"""

import os
import json
import argparse
import numpy as np
from misc.io import read_simple_flist
from lorelei_utils import MAPS


def inverse_map():
    """ Return inverse mapping of key value one-2-one mapping """
    inv_map = {}
    for k, v in MAPS.items():
        inv_map[v] = k

    return inv_map


def outlier_detected(lab, theme_list, ground_truth_prep, basename):

    inv_map_theme = 'utils'
    lab.append(theme_list.get(inv_map_theme))
    ground_truth_prep[basename] = lab


def process_annot_file(annot_files, theme_list):
    answer = {}
    ground_truth_prep = {}
    p=0
    for i in range(len(annot_files)):
        annot = annot_files[i]
        basename = os.path.basename(annot)
        basename = basename.split('.')[0]

        with open(annot, 'r') as document:
            for line in document:
                line = line.strip()
                if not line:  # empty line?
                    continue
                line = line.replace("Utilities, Energy, or Sanitation", "UES")
                line = line.split(":")

                if line[0] in answer:
                    answer[line[0]] += "," + line[1].strip()
                else:
                    answer[line[0]] = line[1].strip()

        # import pdb
        # pdb.set_trace()
        themes = answer.get('TYPE')  #' '.join(answer.get("TYPE"))
        themes = themes.split(",") # Split string on the basis of comma
        themes = [x.strip() for x in themes] # remove white spaces


        inv_map = inverse_map()
        inv_map["UES"] = "utils"
        lab = []
        for theme in themes:

            if inv_map.get(theme) == None:

                for word in theme.split():
                    word = word.replace(',', '')
                    inv_map_theme = inv_map.get(word)
                    lab.append(theme_list.get(inv_map_theme))
                ground_truth_prep[basename] = lab

            else:
                inv_map_theme = inv_map.get(theme)
                lab.append(theme_list.get(inv_map_theme))
                ground_truth_prep[basename] = lab

        answer = {} # Empty the dictionary

    print("======== Processing Annotations Completed ========")
    return ground_truth_prep


def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    in_annot_list = args.in_annot_list
    annot_files = read_simple_flist(in_annot_list)
    print('Input annotation file list loaded')
    print("No. of annot files:", len(annot_files))
    themes2int_json = args.themes2int_json
    # Reading themes2int Jason file and storing it as dictionary
    with open(themes2int_json, encoding='utf-8') as F:
        theme_list = json.loads(F.read())

    ground_truth_prep = process_annot_file(annot_files, theme_list)
    json.dump(ground_truth_prep, open(args.out_dir + 'ground_truth_prep.json', 'w'), indent=2)
    print("Json file saved in "+ args.out_dir + 'ground_truth_prep.json')
    # for k,v in ground_truth_prep.items():
        # print(*v, sep = ",")


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_annot_list", help="input annotation list")
    parser.add_argument("themes2int_json", help="path to themes2int json file")
    parser.add_argument("out_dir", help="path to output dir")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
