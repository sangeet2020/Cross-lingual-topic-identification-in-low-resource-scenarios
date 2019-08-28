#!/bin/bash

# This wrapper file performs 5-fold cross-validation on the labeled English text data and saves the best alpha (regularization constant)
# which with the highest average precision score.

if [ $# -ne 3 ] ; then
    echo "usage: $0 <il> <analyzer> <ngram tokens>"
    echo "  eg : $0 il9 char_wb 3"
    exit
fi

root=/mnt/matylda3/xsagar00/LORELI_LDC
il=${1}         # incident language
ana=${2}        # analyzer. Choice- char_wb or word
ng=${3}         # ngrams
md=("1" "2" "3") # min document frequency

mkdir -p ${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/5_fold_cv_all_vocab
out_dir=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/5_fold_cv_all_vocab

for mdf in "${md[@]}"; do
    echo "python ${root}/lorelei/src/baseline_ENG_SF.py" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.txt.pruned" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.themes.pruned" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/themes.txt" \
        "${out_dir}/${ana}_ng_${ng}_mdf_${mdf}_" \
        "-ana ${ana} -ng ${ng} -mdf ${mdf}" >> ${out_dir}/cv_blade
done
chmod +x ${out_dir}/cv_blade