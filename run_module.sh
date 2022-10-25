export BASE_PATH='/work/snaseri_umass_edu/'

cd ./src
python3 rerank.py \
        --mode "AUTO" \
        --seed 42 \
        --model_name xlm-roberta-base \
        --output_dir $BASE_PATH/better_P3/experiments/BetterRerankerS1/runs \
        --checkpoint $BASE_PATH/multi_lang_dpr/checkpoints/msmarco/dpr_baseline_msmaroco_xlmr_multi_lang/ar_fa_ru/checkpoint_at_0.pth \
        --collection_file $BASE_PATH/better_P3/data/BETTER_PHASE3_COMBO_ARABIC_FARSI_RUSSIAN/combined-corpus.jl \
        --task_file $BASE_PATH/better_P3/data/BETTER_PHASE3_COMBO_ARABIC_FARSI_RUSSIAN/62e171cedebb2f0d2619f2f6.analytic_tasks.json \
        --test_qrels $BASE_PATH/better_P3/data/BETTER_PHASE3_COMBO_ARABIC_FARSI_RUSSIAN/IR-relevance-assessments.qrels.GALAGO \
        --test_run $BASE_PATH/better_P3/experiments/BetterRerankerS1/runs/62e171cedebb2f0d2619f2f6.FINAL.out \
        --rerank_topK 1000