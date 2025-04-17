#!/usr/bin/bash
# --save_model \
for seed in 0 # 0 1 2 3 4
do
    for multimodal_method in 'umc'
    do
        for method in 'testmethod'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in 'MIntRec' # 'MELD-DA' 'IEMOCAP-DA'
                do
                    python run.py \
                    --dataset $dataset \
                    --data_path '/root/autodl-tmp/home/Share/Dataset/LZH' \
                    --logger_name $method \
                    --multimodal_method $multimodal_method \
                    --method $method\
                    --setting 'semi_supervised'\
                    --train \
                    --tune \
                    --save_results \
                    --save_model \
                    --seed $seed \
                    --gpu_id '0' \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_${dataset} \
                    --results_file_name "results_smc_test.csv" \
                    --output_path "/root/autodl-tmp/home/lizhuohang/reaserch/EMNLP/SMC/outputs_test"
                done
            done
        done
    done
done
