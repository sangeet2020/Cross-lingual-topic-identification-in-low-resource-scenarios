#!/bin/bash

# Performs feature extraction on parallel text data and labeled English text data
#         root: /my/root/directory/cross_ling_topic_id/
#         script: /my/root/directory/cross_ling_topic_id/src

if [ $# -ne 7 ] ; then
    echo "usage: $0 <il> <N> <ana> <mdf> <ng> <mv> <analyzer>"
    echo "  eg : $0 il9 200 char_wb 3 3 20000 char"
    exit
fi

il=${1}         # choice: il9, il10, hin, zul
N=${2}          # Number of vocabs from each topic
ana=${3}        # analyzer: char_wb or word
mdf=${4}        # min document frequency
ng=${5}         # ngram
mv=${6}         # maximum vocabulary size for IL
analyzer=${7}   # Choice: char , word

eng_mdf=1
eng_mv=None
root=/my/root/directory/cross_ling_topic_id/          # Edit this line with path of your root directory
script=/my/root/directory/cross_ling_topic_id/src     # Edit this line with path of source code directory

specs=("full_set" "thm_reltd")

for spec in "${specs[@]}"; do

    current_dir=${root}/expt_tfidf/${il}/${analyzer}/${spec}_doc_level_best_vocabs_N_${N}
    mkdir -p ${current_dir}
    eng_mtx=eng_tfidf_${ana}_n${ng}_mdf_${eng_mdf}_mv_${eng_mv}.mtx
    il_mtx=${il}_tfidf_${ana}_n${ng}_mdf_${mdf}_mv_${mv}.mtx


    eng_feats=${current_dir}/feats/${eng_mtx}
    il_feats=${current_dir}/feats/${il_mtx}
    sf_themes=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.themes.pruned
    sf_txt=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.txt.pruned
    best_params=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/best_params.txt
    labeled_data=${current_dir}/feats/labelled_data_info.txt

    out_dir=${current_dir}

    echo "--------------------------------"
    echo "Parallel text feature extraction"
    echo "--------------------------------"


    eng_txt=${root}/lorelei/${il^^}/parallel_text/eng.txt
    il_txt=${root}/lorelei/${il^^}/parallel_text/${il}.txt
    mkdir -p ${root}/expt_tfidf/${il}
    mkdir -p ${root}/expt_tfidf/${il}/${analyzer}
    mkdir -p ${root}/expt_tfidf/${il}/${analyzer}/full_set
    mkdir -p ${root}/expt_tfidf/${il}/${analyzer}/thm_reltd
    mkdir -p ${out_dir}/feats
    best_vocabs_p=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/top_vocab_list_${ana}_mdf_1_ng_3_N_${N}.txt
    #Feat extr for eng text
    python ${root}/${script_p}/feature_xtr_from_text.py ${eng_txt} eng ${out_dir}/feats/ -labeled_text_f ${sf_txt} -ana ${ana} -ng ${ng} -best_vocabs_p ${best_vocabs_p}

    #Feat extr for il text
    python ${root}/${script_p}/feature_xtr_from_text.py ${il_txt} ${il} ${out_dir}/feats/ -ana ${ana} -ng ${ng} -mdf ${mdf} -mv ${mv}

    echo "--------------------------------"
    echo "Sorting theme related docs"
    echo "--------------------------------"
    out_dir_thm=${root}/expt_tfidf/${il}/${analyzer}/thm_reltd_doc_level_best_vocabs_N_${N}

    python ${root}/${script_p}/topic_classification.py ${eng_feats} ${sf_themes} ${best_params} ${out_dir_thm} ${labeled_data} -ana ${ana} -ng ${ng} -mdf ${mdf}

    test_prop_f=${out_dir_thm}/final_test_prob.npy
    python ${root}/${script_p}/thm_reltd_docs_selectn.py ${test_prop_f} ${out_dir_thm}

    #Extracting feats for thm_related docs

    echo "----------------------------------------"
    echo "Feature extraction of theme related docs"
    echo "----------------------------------------"

    thm_doc_list=${out_dir_thm}/tr_indices.txt
    #Feat extr for eng text
    python ${root}/${script_p}/feature_xtr_from_text.py ${eng_txt} eng ${out_dir_thm}/feats/ -thm_doc_list ${thm_doc_list} -labeled_text_f ${sf_txt} -ana ${ana} -ng ${ng} -best_vocabs_p ${best_vocabs_p}
    #Feat extr for il text
    python ${root}/${script_p}/feature_xtr_from_text.py ${il_txt} ${il} ${out_dir_thm}/feats/ -thm_doc_list ${thm_doc_list} -ana ${ana} -ng ${ng} -mdf ${mdf} -mv ${mv}

done
done
