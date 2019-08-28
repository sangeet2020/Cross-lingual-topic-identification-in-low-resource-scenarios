#!/bin/bash

# Does following task:
#         (i)     Generate best vocabulary from each class of labeled English text data
#         (ii)    Finds the best alpha for each topic/label
#         (iii)   Generates bash file to generate features for parallel text
#         (iv)    Generates bash file to compute final average precision scores for the given test data.
#         (v)     Display the score for all N (vocab size for each label) at once.

if [ $# -ne 5 ] ; then
    echo "usage: $0 <il> <analyzer> <ngram tokens> <N> <root>"
    echo "  eg : $0 il9 char_wb 3 200 /my/root/directory/cross_ling_topic_id/"
    exit
fi


il=${1}     # il: il9, il10, zul, hin
ana=${2}    # analyzer: char_wb or word
ng=${3}     # ngram
N=${4}      # vocab size for each class eg. 100, 200 etc
root=${5}   # your root directory: /my/root/directory/cross_ling_topic_id/

rm -rf ${root}/expt_tfidf/${il}/char/all_Score_one_file.txt
rm -rf ${root}/expt_tfidf/${il}/generate_vocabs_blade
rm -rf ${root}/expt_tfidf/${il}/baseline_best_params_generator_blade.bsn
rm -rf ${out_dir}/feats_extraction_blade

echo -n "paste -d\" \" " > ${root}/expt_tfidf/${il}/char/all_Score_one_file.txt
chmod +x ${root}/expt_tfidf/${il}/char/all_Score_one_file.txt

for n in "${N[@]}"; do
        echo "Generate best vocabs"
        echo "python ${root}/lorelei/src/vocab_selection_short.py" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.txt.pruned" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.themes.pruned" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/themes.txt" \
        "${n}" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/" \
        "-ana ${ana} -ng ${ng} -mdf 1" >> ${root}/expt_tfidf/${il}/generate_vocabs_blade
        chmod +x ${root}/expt_tfidf/${il}/generate_vocabs_blade

        # Find best parameters
        echo "python ${root}/lorelei/src/baseline_ENG_SF_best_vocabs.py" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.txt.pruned" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.themes.pruned" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/themes.txt" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/top_vocab_list_${ana}_mdf_1_ng_${ng}_N_${n}.txt" \
        "${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/${ana}_mdf_1_ng_${ng}_N_${n}_" \
        "-ana ${ana} -ng ${ng} -mdf 1" >> ${root}/expt_tfidf/${il}/baseline_best_params_generator_blade.bsn
        chmod +x ${root}/expt_tfidf/${il}/baseline_best_params_generator_blade.bsn

        # Feature extraction
        out_dir=${root}/expt_tfidf/${il}/char
        echo ${root}/lorelei/wrappers/wrapper_feats_gen.sh ${il} ${n} >> ${out_dir}/feats_extraction_blade
        chmod +x ${out_dir}/feats_extraction_blade

        # Score Generator
        echo ${root}/lorelei/wrappers/wrapper_final_scores_generator_new_doc_level.sh ${n} >> ${root}/expt_tfidf/${il}/char/final_score_gen.tsk

        # Final score for all N in one line
        my_dir=${root}/expt_tfidf/${il}/char/full_set_doc_level_best_vocabs_N_${n}_new
        echo -n "${my_dir}/avg_AP_scores_alpha_1e-04.txt" \
        "${my_dir}/avg_AP_scores_alpha_1e-02.txt" \
        "${my_dir}/avg_AP_scores_alpha_1e+00.txt" \
        "${my_dir}/avg_AP_scores_alpha_1e+01.txt " >> ${root}/expt_tfidf/${il}/char/all_Score_one_file.txt

        echo "done"

done
