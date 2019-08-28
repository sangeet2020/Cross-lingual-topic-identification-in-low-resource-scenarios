
import os
import glob
import re
import argparse
import matplotlib.pyplot as plt
from misc.io import read_simple_flist
import numpy as np


def main():
    """ main """
    args = parse_arguments()

    basepath = args.base_dir
    scores_files = glob.glob(basepath+'/models/'+ '/*.txt')
    out_scores_dir = glob.glob(basepath + '/out_scores*')
    print(out_scores_dir)

    #this for loop creates a dict, storing alpha and for each alpha
    #it's stores the corresponding weighted avg AP score.
    # Here it's a little mess since we are reading alpha from the dir name.
    wgt_avg_dict = {}
    for dir in out_scores_dir:
        alpha = dir.split("_")[5]
        out_score_file = dir + '/scores.txt'
        in_ap_score = read_simple_flist(out_score_file)
        for line in in_ap_score:
            if re.match("wgt-average-AP-scores", line):
                score = float(line.split(": ")[1])
                wgt_avg_dict[alpha] = score


    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    i = 0
    out_scores_dict = {}
    alphas = []

    # Plot for no. of iter vs validation loss
    for score_f in scores_files:
        alpha = os.path.splitext(os.path.basename(score_f).split("_")[2])[0]
        alphas.append(alpha)
        out_scores = np.loadtxt(score_f)
        out_scores = np.array(out_scores)
        t_trng_iter = len(out_scores)
        trng_iters = np.arange(1,t_trng_iter+1,1)

        ax1.plot(trng_iters, out_scores, color='C'+str(i), label = 'alpha='+ alpha, linewidth=1)
        ax1.legend()

        out_scores_dict[alpha] =  p = min(out_scores)
        i = i + 1


    ax1.set_xlabel('Training iterations')
    ax1.set_ylabel('Valid loss (MSE)')
    fig1.savefig(args.base_dir+'/'+'fig1.pdf'.format(1))


    # multi-bar plot for validation loss and least validation loss vs alpha
    validation_loss = []
    weighted_avg_AP = []

    for alpha in alphas:
        validation_loss.append(out_scores_dict.get(alpha))
        weighted_avg_AP.append(wgt_avg_dict.get(alpha))

    bar_width = 0.35
    pos = np.arange(len(alphas))
    ax2.bar(pos, validation_loss, bar_width, label='Validation loss')
    ax2.bar(pos+bar_width, weighted_avg_AP, bar_width, label='Weighted avg AP scores')
    plt.xticks(pos + bar_width/2 , alphas)
    ax2.legend()
    fig2.savefig(args.base_dir+'/'+'fig2.pdf'.format(2))

    # print(alphas)
    # print(validation_loss)
    # print(weighted_avg_AP)



def parse_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", help="path to dir full_set or thm_reltd")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    main()
