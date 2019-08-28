
########################## Configure these according to the IL ##########################
#enter path containing the annotations

#HINDI dataset
# dataset_path=/mnt/matylda2/data/LORELEI/LDC2017E88_LORELEI_Hindi_Speech_Database/HIN_EVAL_20170601/HIN_EVAL_20170601/
# ldc_dir=/mnt/matylda4/kesiraju/code/lorelei/LDC_processed/subset_no_HIN/
# ils=("hin")

#ZULU dataset
# dataset_path=/mnt/matylda2/data/LORELEI/LDC2017E93_LORELEI_Zulu_Speech_Database/ZUL_EVAL_20170920/ZUL_EVAL_20170920/
# ldc_dir=/mnt/matylda3/xsagar00/LORELI_LDC/lorelei/ZUL/subset_no_ZUL/
# ils=("zul")

# IL10 dataset
# dataset_path=/mnt/matylda3/xsagar00/LORELI_LDC/lorelei/IL10/Unsequestered_IL10_SetE_SpeechAnnotation_20180914/
# ldc_dir=/mnt/matylda3/xsagar00/LORELI_LDC/lorelei/IL10/subset_no_IL10/
# ils=("il10")


asr_out=("asr_1_out" "asr_2_out")
mdf=3
analyzers=("char")
ana=char_wb
ng=3
########################################################################################

root=/mnt/matylda3/xsagar00/LORELI_LDC
script_p=lorelei/src

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


    echo "----------------------------------------------"
    echo "Baseline classification systems on ENG SF data"
    echo "----------------------------------------------"


    text_f=${labeled_out_dir}/all_text.txt.pruned
    label_f=${labeled_out_dir}/all_text.themes.pruned
    themes_txt=${labeled_out_dir}/themes.txt
    python ${root}/${script_p}/baseline_ENG_SF.py ${text_f} ${label_f} ${themes_txt} ${labeled_out_dir}/ -ana ${ana} -ng ${ng} -mdf ${mdf}

done
