# BetterRerankerS1

#### DPR-based Supervised reranker trained on mMarco and then test on BETTER

### Reranking

```
cd ./src
python3 rerank.py \
        --mode [BETTER mode "AUTO" or "AUTO-HITL"] \
        --seed [initial seed value] \
        --model_name [pretrained encoder] \
        --output_dir [output directory] \
        --checkpoint [fine-tuned model] \
        --collection_file [collection file to access the document text .jl] \
        --task_file [BETTER task (i.e. query) json] \
        --test_run [baseline run to re-rank] \
        --rerank_topK [rerank top K documents]
        --test_qrels [optional, relevant judgment] \
```
