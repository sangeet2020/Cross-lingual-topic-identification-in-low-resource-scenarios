# Generates bash files to learn transformation and submit jobs as well  to GPUs

N=('600' '800' '1000' '1200' '1400' '1600' '1800' '2000' '2200' '2400')
il=zul
ana=char_wb
ng=3
mdf=3
eng_mdf=1
mv=20000
eng_mv=None
root=/mnt/matylda3/xsagar00/LORELI_LDC

rm -r ${root}/expt_tfidf/${il}/char/gpu_learn_trans_task
for n in "${N[@]}"; do

     feats_f=${root}/expt_tfidf/${il}/char/full_set_doc_level_best_vocabs_N_${n}_new
     echo "export \`gpu\`" > ${feats_f}/learn_trans_N_${n}.tsk
     echo "python ${root}/lorelei/src/learn_transformation.py" \
     "${feats_f}/feats/${il}_tfidf_${ana}_n${ng}_mdf_${mdf}_mv_${mv}.mtx" \
     "${feats_f}/feats/eng_tfidf_${ana}_n${ng}_mdf_${eng_mdf}_mv_${eng_mv}.mtx" \
     "${feats_f}/feats/labelled_data_info.txt" \
     "${feats_f}/models/" >> ${feats_f}/learn_trans_N_${n}.tsk

     realpath ${feats_f}/learn_trans_N_${n}.tsk > ${root}/expt_tfidf/${il}/char/gpu_${n}
     chmod +x ${root}/expt_tfidf/${il}/char/gpu_${n}
     chmod +x ${feats_f}/learn_trans_N_${n}.tsk

     manage_task.sh -q long.q@@gpu -l gpu=1,gpu_ram=8G,matylda3=1 ${root}/expt_tfidf/${il}/char/gpu_${n}

     # AVG AP score
     # paste -d" " ${feats_f}/avg_AP_scores_alpha_1e-04.txt ${feats_f}/avg_AP_scores_alpha_1e-02.txt ${feats_f}/avg_AP_scores_alpha_1e+00.txt ${feats_f}/avg_AP_scores_alpha_1e+01.txt

     # Vocab count
     # wc -l ${root}/expt_tfidf/${il}/labeled_text_no_${il^^}/top_vocab_list_char_wb_mdf_1_ng_3_N_${n}.txt
     # echo "Done"

done
