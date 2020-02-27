nohup python train.py \
  --model vanilla_bert \
  --datafiles datafile/robust/queries.tsv /data/cedr/data/robust_FIX/documents.tsv \
  --qrels datafile/robust/qrels \
  --train_pairs /data/cedr/data/robust/f1.train.pairs \
  --valid_run datafile/robust/f1.valid.run \
  --model_out_dir /data/cedr_crys/model/f1.vbert &