import os
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import data

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
device = xm.xla_device()
devices = xm.get_xla_supported_devices(max_devices=3)

LR = 0.001
BERT_LR = 2e-5
MAX_EPOCH = 100
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 32
GRAD_ACC_SIZE = 8

print('device in training.py:', device)

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, it, length):
        self.it = it
        self.length = length
        self.keys = ['query_tok', 'query_mask', 'doc_tok', 'doc_mask']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = next(self.it)
        return {
            'query_tok': x['query_tok'],
            'query_mask': x['query_mask'],
            'doc_tok': x['doc_tok'],
            'doc_mask': x['doc_mask'],
        }

def main(model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir):
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)
    # optimizer = torch.optim.SGD([non_bert_params, bert_params], lr=LR, momentum=0.9)

    model.to(device)
    # model_parallel = dp.DataParallel(model, device_ids=devices)

    epoch = 0
    top_valid_score = None
    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss}')

        # train_set = TrainDataset(
        #     it=data.iter_train_pairs(model, dataset, train_pairs, qrels, 1),
        #     length=BATCH_SIZE * BATCHES_PER_EPOCH
        # )
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=GRAD_ACC_SIZE,)
        # # for i, tr in enumerate(train_loader):
        # #     for tt in tr:
        # #         print(tt, tr[tt].size())
        # #     break
        # # print('finished')
        # return
        # model_parallel(train_iteration_multi, train_loader)

        valid_score = validate(model, dataset, valid_run, qrelf, epoch, model_out_dir)
        print(f'validation epoch={epoch} score={valid_score}')
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            model.save(os.path.join(model_out_dir, 'weights.p'))

def train_iteration_multi(model, loader, device, context):
    total = 0
    model.train()
    total_loss = 0.

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.SGD([non_bert_params, bert_params], lr=LR, momentum=0.9)

    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train') as pbar:
        for record in loader:

            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()
            total_loss += loss.item()
            total += count

            if n_iter > 0:
                import torch_xla.debug.metrics as met
                print(n_iter, len(record['query_tok']))

            if total % BATCH_SIZE == 0:
                print('*'*5, n_iter, len(record['query_tok']))
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()
            pbar.update(count)
            # if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
    return total_loss

def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    model.train()
    total = 0
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train') as pbar:
        for n_iter, record in enumerate(data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE)):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()
            # total_loss += loss.item()
            total_loss += loss
            # total_loss += 0.4
            total += count

            # if n_iter > 0:
            #     print(n_iter, [len(record[x]) for x in record])
            #     print(n_iter, [(record[x].size(), record[x].device) for x in ['query_tok', 'query_mask', 'doc_tok', 'doc_mask']])
            #     # import torch_xla.debug.metrics as met
            #     # print(met.metrics_report())

            if total % BATCH_SIZE == 0:
                # print('*'*5, n_iter, len(record['query_tok']))
                xm.optimizer_step(optimizer, barrier=True)
                optimizer.zero_grad()

            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                # return total_loss
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
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
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
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()

    model = MODEL_MAP[args.model]()

    check_model_size(model);

    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    main(model, dataset, train_pairs, qrels, valid_run, args.qrels.name, args.model_out_dir)

def check_model_size(model, input_size=(16,1,256,256)):
    print('checking model size')
    from pytorch_modelsize import SizeEstimator

    # se = SizeEstimator(model, input_size=(16,1,256,256))
    se = SizeEstimator(model, input_size=input_size)

    se.get_parameter_sizes()
    se.calc_param_bits()

    print('param_bits: ', se.param_bits)

if __name__ == '__main__':
    main_cli()
