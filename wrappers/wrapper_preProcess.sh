#!/bin/bash

# Task performed:
#         (i)     Generate ground truth annotations
#         (ii)    Preparing ground truth of the annotations
#         (iii)   Convert ASR outputs to text file and document id file
#         (iv)    Analyze English labeled text data
#         (v)     Eliminate out-of-domain docs


root = /my/root/directory/cross_ling_topic_id/          # Edit this line with path of your root directory
script = /my/root/directory/cross_ling_topic_id/src     # Edit this line with path of source code directory

if [ $# -ne 7 ] ; then
    echo "usage: $0 <il> <ana> <mdf> <ng> <dataset_path> <ldc_dir>"
    echo "  eg : $0 il9 char_wb 3 3 path/to/annotations path/to/labeled_text_data/"
    exit
fi

il=${1}         # choice: il9, il10, hin, zul
ana=${2}        # analyzer: char_wb or word
mdf=${3}        # min document frequency
ng=${4}         # ngram
########################## Configure these according to the IL ##########################
dataset_path=${5}      # path to annotations for given language pack
ldc_dir=${6}           # path to labeled English text jason file
########################################################################################


asr_out=("asr_1_out" "asr_2_out")
analyzers=("char")


for il in "${ils[@]}"; do

    mkdir -p ${root}/expt_tfidf/${il}/labeled_text_no_${il^^}
    out_dir=${root}/lorelei/${il^^}
    mkdir -p ${out_dir}/etc
    echo "---------------------------"
    echo "Generating annotations list"
    echo "---------------------------"

    find ${dataset_path} -type f -name "*.txt" > ${out_dir}/etc/annot_${il^^}.flist
    echo "DONE"


    echo "-----------------------------------------"
    echo "Preparing ground truth of the annotations"
    echo "-----------------------------------------"

    in_annot_list=${out_dir}/etc/annot_${il^^}.flist
    themes2int_json=${root}/lorelei/IL9/etc/themes2int.json
    python ${root}/${script_p}/get_ground_truth_from_annotations.py ${in_annot_list} ${themes2int_json} ${out_dir}/etc/

    echo "----------------------------------"
    echo "ASR --> doc per line & id per line"
    echo "----------------------------------"


    for asr in "${asr_out[@]}"; do
        in_asr_file=${out_dir}/${asr}/${asr}.txt
        python ${root}/${script_p}/asr_out_to_docs.py ${in_asr_file} ${out_dir}/${asr}/asr_out_all_docx.txt ${out_dir}/${asr}/asr_out_all_docx.ids
    done


    echo "---------------------"
    echo "Analyzing ENG SF data"
    echo "---------------------"
    python ${root}/${script_p}/process_LDC_EN_SF_data.py ${ldc_dir} ${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/

    echo "----------------------------"
    echo "Eliminating out-of-domain docs"
    echo "----------------------------"

    labeled_out_dir=${root}/expt_tfidf/${il}/labeled_text_no_${il^^}
    txt=${labeled_out_dir}/all_text.txt
    themes=${labeled_out_dir}/all_text.themes


    pr -Jmt -S" % " ${themes} ${txt} > ${labeled_out_dir}/all_text.txt.themes
    grep -v ^-1 ${labeled_out_dir}/all_text.txt.themes  > ${labeled_out_dir}/all_text.txt.themes.pruned
    cat ${labeled_out_dir}/all_text.txt.themes.pruned | cut -d% -f1 > ${labeled_out_dir}/all_text.themes.pruned
    cat ${labeled_out_dir}/all_text.txt.themes.pruned | cut -d% -f2- > ${labeled_out_dir}/all_text.txt.pruned

done
