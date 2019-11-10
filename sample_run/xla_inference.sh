nohup python rerank.py \
  --model vanilla_bert \
  --datafiles sample_data/queries.tsv sample_data/documents.tsv \
  --run sample_data/test.run \
  --model_weights /data/cedr_crys/model/f1.vbert/weights.p \
  --out_path sample_results/test.run &