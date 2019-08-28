#!/bin/bash

# Generates bash files to learn transformation and submit jobs as well to GPUs

if [ $# -ne 6 ] ; then
    echo "usage: $0 <il> <n> <ana> <mdf> <ng> <mv> <analyzer> <spec>"
    echo "  eg : $0 il9 200 char_wb 3 3 20000 char full_set"
    exit
fi

root=/my/root/directory/cross_ling_topic_id/          # Edit this line with path of your root directory
script=/my/root/directory/cross_ling_topic_id/src     # Edit this line with path of source code directory

il=${1}         # choice: il9, il10, hin, zul
n=${2}          # Number of vocabs from each topic
ana=${3}        # analyzer: char_wb or word
mdf=${4}        # min document frequency
ng=${5}         # ngram
mv=${6}         # maximum vocabulary size for IL
analyzer=${7}   # Choice: char , word
spec=${8}       # Choice full_set, thm_reltd

eng_mdf=1
eng_mv=None

rm -r ${root}/expt_tfidf/${il}/${analyzer}/gpu_learn_trans_task

feats_f=${root}/expt_tfidf/${il}/${analyzer}/${spec}_doc_level_best_vocabs_N_${n}_new
echo "export \`gpu\`" > ${feats_f}/learn_trans_N_${n}.tsk
echo "python ${script}/learn_transformation.py" \
"${feats_f}/feats/${il}_tfidf_${ana}_n${ng}_mdf_${mdf}_mv_${mv}.mtx" \
"${feats_f}/feats/eng_tfidf_${ana}_n${ng}_mdf_${eng_mdf}_mv_${eng_mv}.mtx" \
"${feats_f}/feats/labelled_data_info.txt" \
"${feats_f}/models/" >> ${feats_f}/learn_trans_N_${n}.tsk

realpath ${feats_f}/learn_trans_N_${n}.tsk > ${root}/expt_tfidf/${il}/${analyzer}/gpu_${n}
chmod +x ${root}/expt_tfidf/${il}/${analyzer}/gpu_${n}
chmod +x ${feats_f}/learn_trans_N_${n}.tsk

manage_task.sh -q long.q@@gpu -l gpu=1,gpu_ram=8G,matylda3=1 ${root}/expt_tfidf/${il}/${analyzer}/gpu_${n}

# AVG AP score
# paste -d" " ${feats_f}/avg_AP_scores_alpha_1e-04.txt ${feats_f}/avg_AP_scores_alpha_1e-02.txt ${feats_f}/avg_AP_scores_alpha_1e+00.txt ${feats_f}/avg_AP_scores_alpha_1e+01.txt

# Vocab count
# wc -l ${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/top_vocab_list_${analyzer}_wb_mdf_1_ng_3_N_${n}.txt
# echo "Done"
