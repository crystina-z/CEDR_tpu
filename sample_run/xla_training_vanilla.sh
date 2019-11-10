nohup python train.py \
  --model vanilla_bert \
  --datafiles sample_data/queries.tsv sample_data/documents.tsv \
  --qrels sample_data/qrels \
  --train_pairs sample_data/train.pairs \
  --valid_run sample_data/valid.run \
  --model_out_dir sample_results/ &