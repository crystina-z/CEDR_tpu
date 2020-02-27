nohup python rerank.py \
  --model vanilla_bert \
  --datafiles data/robust/queries.tsv /data/cedr/data/robust_FIX/documents.tsv \
  --run data/robust/f1.test.run \
  --model_weights /data/cedr_crys/model/f1.vbert/weights.p \
  --out_path /data/cedr_crys/model/f1.vbert/test.run &