#!/usr/bin/bash

# for seed in 0 1 2 3 4
for seed in  0 1 2 3 4

do
    for multimodal_method in 'smc'
    do
        for method in 'smc'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in  'MIntRec' # 'MIntRec' # 'MELD-DA'#  'IEMOCAP-DA'
                do
                    python run.py \
                    --dataset $dataset \
                    --data_path '/root/autodl-tmp/home/Share/Dataset/LZH' \
                    --logger_name $method \
                    --setting 'semi_supervised'\
                    --merge_dev True \
                    --multimodal_method $multimodal_method \
                    --method $method\
                    --train \
                    --tune \
                    --save_results \
                    --seed $seed \
                    --gpu_id '1' \
                    --save_model \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_${dataset} \
                    --results_file_name "results_smc.csv" \
                    --output_path "/root/autodl-tmp/home/lizhuohang/reaserch/EMNLP/SMC/outputs_test/${dataset}"
                done
            done
        done
    done
done
