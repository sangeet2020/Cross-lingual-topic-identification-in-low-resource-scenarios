#!/bin/bash

# This wrapper file removes 'out-of-domain' from the labeled English text data
# Language: Hindi (hin), Zulu (zul)

root=/mnt/matylda3/xsagar00/LORELI_LDC
ils=("hin" "zul")
for il in "${ils[@]}"; do

    txt=${root}/labeled_text_no_${il^^}/all_text.txt
    themes=${root}/labeled_text_no_${il^^}/all_text.themes
    out_dir=${root}/labeled_text_no_${il^^}

    pr -Jmt -S" % " ${themes} ${txt} > ${out_dir}/all_text.txt.themes
    grep -v ^-1 ${out_dir}/all_text.txt.themes  > ${out_dir}/all_text.txt.themes.pruned
    cat ${out_dir}/all_text.txt.themes.pruned | cut -d% -f1 > ${out_dir}/all_text.themes.pruned
    cat ${out_dir}/all_text.txt.themes.pruned | cut -d% -f2- > ${out_dir}/all_text.txt.pruned

done