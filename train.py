import os
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import data

import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

import torch_xla.core.xla_model as xm
device = xm.xla_device()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('device in training.py:', device)  # xla:1

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}


def main(model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir):
    model.to(device)

    LR = 0.001
    BERT_LR = 2e-5
    # MAX_EPOCH = 100
    MAX_EPOCH = 10

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    epoch = 0
    top_valid_score = None
    valid_scores, losses = [], []
    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels)
        losses.append(loss)
        print(f'train epoch={epoch} loss={loss}')

        valid_score = validate(model, dataset, valid_run, qrelf, epoch, model_out_dir)
        valid_scores.append(valid_score)
        print(f'validation epoch={epoch} score={valid_score}')
        
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            model.save(os.path.join(model_out_dir, 'weights.p'))

    plot(losses, name='training loss')
    plot(valid_scores, name='validation score (p20)')

    
def plot(values, name):
    plt.figure()
    plt.plot(values)
    plt.title(name)
    plt.savefig(f'{name}.png')
    print(f'saved {name}.png')

def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 32
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])

            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()

            total_loss += loss
            total += count

            if total % BATCH_SIZE == 0:
                # optimizer.step()
                xm.optimizer_step(optimizer, barrier=True)
                optimizer.zero_grad()
            pbar.update(count)

            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                # return total_loss.item()
                return total_loss.item()


def validate(model, dataset, run, qrelf, epoch, model_out_dir):
    VALIDATION_METRIC = 'P.20'
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    run_model(model, dataset, run, runf)
    return trec_eval(qrelf, runf, VALIDATION_METRIC)


def run_model(model, dataset, run, runf, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores.detach().cpu().numpy()):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')


def trec_eval(qrelf, runf, metric):
    trec_eval_f = 'bin/trec_eval'
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])


def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', nargs='+')
    # parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()
    
    print('datafiles: ', args.datafiles)
    datafiles = [open(f, 'r', encoding='utf-8') for f in args.datafiles]
    model = MODEL_MAP[args.model]()
    
    # dataset = data.read_datafiles(args.datafiles)
    dataset = data.read_datafiles(datafiles)

    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    main(model, dataset, train_pairs, qrels, valid_run, args.qrels.name, args.model_out_dir)


if __name__ == '__main__':
    main_cli()
