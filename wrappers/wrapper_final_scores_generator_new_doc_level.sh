#!/bin/bash

# Task performed:
#         (i)     Training English Classifier
#         (ii)    Combine all sentences of one documnent into separate sentences.
#         (iii)   Feature exxtraction of text data
#         (iv)    Apply transoformation from IL space to English space
#         (v)     Predict topic
#         (vi)    Compute averge precision scores

root=/my/root/directory/cross_ling_topic_id/          # Edit this line with path of your root directory
script=/my/root/directory/cross_ling_topic_id/src     # Edit this line with path of source code directory

if [ $# -ne 8 ] ; then
    echo "usage: $0 <il> <n> <ana> <mdf> <ng> <mv> <analyzer> <spec>"
    echo "  eg : $0 il9 200 char_wb 3 3 20000 char full_set"
    exit
fi

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
asr_out=asr_1_out       # folder name for ASR outputs


current_dir=${root}/expt_tfidf/${il}/${analyzer}/${spec}_doc_level_best_vocabs_N_${n}
eng_mtx=eng_tfidf_${ana}_n${ng}_mdf_${eng_mdf}_mv_${eng_mv}.mtx
il_mtx=${il}_tfidf_${ana}_n${ng}_mdf_${mdf}_mv_${mv}.mtx

eng_feats=${current_dir}/feats/${eng_mtx}
il_feats=${current_dir}/feats/${il_mtx}
sf_themes=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.themes.pruned
sf_txt=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/all_text.txt.pruned
best_params=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/${ana}_mdf_${eng_mdf}_ng_${ng}_N_${n}_best_params.txt
labeled_data=${current_dir}/feats/labelled_data_info.txt


out_dir=${current_dir}
mkdir -p ${out_dir}/clfs
echo "--------------------"
echo "Training Classifiers"
echo "--------------------"
python ${root}/${script_p}/train_classifier_on_eng_sf_features.py ${eng_feats} ${sf_themes} ${best_params} ${labeled_data} ${out_dir}/clfs/clfs

asr_out_subset_docs_ids=${root}/lorelei/${il^^}/${asr_out}/asr_subset_docs.ids
themes2int_file=/mnt/matylda3/xsagar00/LORELI_LDC/lorelei/IL9/etc/themes2int.json
ground_truth_file=${root}/lorelei/${il^^}/etc/ground_truth_prep.json
asr_out_subset_docs_txt=${root}/lorelei/${il^^}/${asr_out}/asr_subset_docs.txt

echo "------------------------------"
echo "Sent per line ->  Doc per line"
echo "------------------------------"
rm -rf ${out_dir}/asr_all_sentn_one_doc.txt
python ${root}/${script_p}/all_sentences_one_doc.py ${asr_out_subset_docs_ids} ${asr_out_subset_docs_txt} ${out_dir}

echo "-----------------------------"
echo ${il^^} "Test Feature Representation"
echo "-----------------------------"

il_cvect_pkl=${il}_count_vect_${ana}_n${ng}_mdf_${mdf}_mv_${mv}.pkl
il_tfidf_pkl=${il}_tfidf_${ana}_n${ng}_mdf_${mdf}_mv_${mv}.pkl

cvect_path=${current_dir}/feats/${il_cvect_pkl}
tfidf_path=${current_dir}/feats/${il_tfidf_pkl}
il_test_doc=${out_dir}/asr_all_sentn_one_doc.txt

mkdir -p ${out_dir}/test_feats
test_feat_dir=${out_dir}/test_feats
# This converts non-latin script into latin script
#perl /mnt/matylda6/baskar/uroman/bin/uroman.pl < ${il_test_doc} > ${out_dir}/asr_all_sentn_one_doc_prons.txt
python ${root}/${script_p}/extract_features_for_test_data.py ${cvect_path} ${tfidf_path} ${out_dir}/asr_all_sentn_one_doc.txt ${test_feat_dir}/test_feats


echo "------------------------"
echo "Applying Transformations"
echo "------------------------"
mkdir -p ${out_dir}/il2eng

test_feats_path=${test_feat_dir}/test_feats_tfidf.mtx
model_path=${current_dir}/models
for file in ${model_path}/*.pt; do
    model=${file}
    #--do something--
    echo "$(basename "$file")"
    base="$(basename "$file")"
    name=${base%_iter*}
    name=${name#*_}

    python ${root}/${script_p}/apply_transformation.py ${test_feats_path} ${model} ${eng_feats} ${out_dir}/il2eng/trans_il2eng_${name}

done

# Remove previous score if any
rm -rf ${out_dir}/pred_sf_labels_sentn_level/
rm -rf  ${out_dir}/pred_sf_labels_sentn_level/

echo "--------------------"
echo "Predicting SF labels"
echo "--------------------"

mkdir -p ${out_dir}/pred_sf_labels_doc_level
mkdir -p ${out_dir}/pred_sf_labels_sentn_level
clf_pkl=${current_dir}/clfs/clfs

for file in ${out_dir}/il2eng/*; do
    trans_feats=${file}
    echo "$(basename "$file")"
    base="$(basename "$file")"
    name=${base%.*}
    name=${name#*eng_}
    python ${root}/${script_p}/predict_sf_labels_for_transformed_il_feats.py ${trans_feats} ${clf_pkl} ${out_dir}/pred_sf_labels_doc_level/out_scores_${name}

    python ${root}/${script_p}/sentence_level_prob.py ${asr_out_subset_docs_ids} ${out_dir}/pred_sf_labels_doc_level/out_scores_${name} ${out_dir}/pred_sf_labels_sentn_level/out_scores_${name}_

done


echo "---------------------------"
echo "Final Scores with PR Curves"
echo "---------------------------"
for file in ${out_dir}/pred_sf_labels_sentn_level/*; do

    python ${root}/${script_p}/evaluate_sf_scores.py ${file} ${asr_out_subset_docs_ids} ${themes2int_file} ${ground_truth_file} ${out_dir}/

done

for file in ${out_dir}/il2eng/*; do
    trans_feats=${file}
    echo "$(basename "$file")"
    base="$(basename "$file")"
    name=${base%.*}
    name=${name#*eng_}

    scores=${out_dir}/out_scores_${name}_sentn_level_out_scores/scores.txt
    echo $scores
    cat ${scores} | awk -F" " '{print $(NF-2)}' | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | head -13 > ${out_dir}/avg_AP_scores_${name}.txt

done
