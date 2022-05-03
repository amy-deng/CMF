def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random
import argparse
import numpy as np
import time
import os
from sklearn.utils import shuffle
import pickle
import csv

parser = argparse.ArgumentParser(description='')
parser.add_argument("--path", type=str, default="../data/", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("-hd","--n_hidden", type=int, default=64, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=-1, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='gdelt_AS_both', help="dataset to use")
parser.add_argument("--grad_norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--epochs", type=int, default=25, help="maximum epochs")
parser.add_argument("-sl","--seq_len", type=int, default=5)
parser.add_argument("-ho","--horizon", type=int, default=1)
parser.add_argument("-bs","--batch_size", type=int, default=32)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("-pa","--patience", type=int, default=5)
parser.add_argument("--use_gru", type=int, default=1, help='1 use gru 0 rnn')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--runs", type=int, default=5, help='number of runs')
parser.add_argument("--train", type=float, default=0.6, help='')
parser.add_argument("--val", type=float, default=0.2, help='')
parser.add_argument("-s","--shuffle", action="store_true", help='')
parser.add_argument("-l","--loop", type=int, default=1, help='')
parser.add_argument("--metric", type=str, default='macro', help='macro,micro,weighted')
parser.add_argument("-w","--weight_loss", action="store_true", help='')
parser.add_argument("--n_evt", type=int, default=20, help='')
parser.add_argument("-m","--model", type=str, default='cmf', help='')
parser.add_argument("--spl", action="store_true", help='down sampling popular labels')
parser.add_argument("--eid", type=int, default=-1, help='one type of event')
parser.add_argument("-nl","--node_layer", type=int, default=1, help='')
parser.add_argument("-td","--textdim", type=int, default=64, help='')
parser.add_argument("-cr","--cri", type=str, default='', help='loss')


args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
from torch.utils.data import DataLoader
from modules import *
from models import *
from data import *
import utils

use_cuda = args.gpu >= 0 and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed) 
device = torch.device('cuda' if use_cuda else 'cpu')
print("cuda:",use_cuda,'device:',device)


# load data
data_dict = load_loc_data(args)

if 'gdelt_US' in args.dataset and data_dict['num_class'] > 1:
    args.multiclass = True
else:
    args.multiclass = False
early_stop_crietira = 'f1' if data_dict['num_class'] == 1 else 'w-f1'
print(data_dict['num_class'],"num_class")
train_data = LocEventData(data_dict['train_time'],data_dict['train_loc'],data_dict['train_y'],device) 
val_data = LocEventData(data_dict['val_time'],data_dict['val_loc'],data_dict['val_y'],device)
test_data = LocEventData(data_dict['test_time'],data_dict['test_loc'],data_dict['test_y'],device)

class_weight = data_dict['class_weight'].to(device)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
 
with open('../data/{}/loc_text_emb.pkl'.format(args.dataset), 'rb') as f:
    doc_embeds = pickle.load(f)
doc_embeds = torch.FloatTensor(doc_embeds)
if use_cuda:
    doc_embeds = doc_embeds.cuda()

 
os.makedirs('models', exist_ok=True)
os.makedirs('models/' + args.dataset, exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/' + args.dataset, exist_ok=True)
os.makedirs('explain', exist_ok=True)
os.makedirs('explain/' + args.dataset, exist_ok=True)
def prepare(args):
    if args.model == 'cmf':
        model = CMF(args.n_hidden, data_dict['num_ents'], data_dict['num_rels'], num_class=data_dict['num_class'], seq_len=args.seq_len, 
                        use_gru=args.use_gru,maxpool=args.maxpool, class_weight=class_weight,weight_loss=args.weight_loss, 
                        n_label=args.n_evt, emb_dim=300,device=device, multiclass=args.multiclass,emb_mod='mean',node_layer=args.node_layer,text_dim=args.textdim)
        model.count_dict = data_dict['count_dict'] 
        model.text_dict = data_dict['text_dict'] 
        model.doc_embeds = doc_embeds 
     
    model_name = model.__class__.__name__
    token = model_name + 'lr'+str(args.lr) + 'wd'+str(args.weight_decay) + 'dp' + str(args.dropout) + 'sl' + str(args.seq_len) + 'h' + str(args.horizon) + 'hd' + str(args.n_hidden) + 'p'+str(args.patience) + 'tr'+str(args.train) 
    if args.shuffle:
        token += '-s'
    if args.weight_loss:
        token += '-wl'
    if args.spl:
        token += '-sp'
    if args.eid >= 0:
        token += '-e'+str(args.eid)
    if args.model in ['cmf'] and args.node_layer > 1:
        token += '-nl'+str(args.node_layer)
    if args.cri == 'loss':
        token += '-los'
    if args.textdim != 64:
        token += '-td'+str(args.textdim)
    
    print('model:', model_name)
    print('token:', token)
    os.makedirs('models/{}/{}'.format(args.dataset, token), exist_ok=True)
    result_file = 'results/{}/{}.csv'.format(args.dataset,token)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)

    if use_cuda:
        model.cuda()
    model.graph_dict = data_dict['graph_dict']
    model.time_of_locs = data_dict['time_of_locs']

    return model, optimizer, result_file, token 



def train(data_loader, data):
    model.train()
    total_loss = 0
    t0 = time.time()
    for i, batch in enumerate(data_loader):
        time_set, loc_set, y  = batch
        loss, _, _ , _= model(time_set, loc_set, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    t1 = time.time()
    reduced_loss = total_loss / (data.len / args.batch_size)
    print("Train Epoch {:03d} | Loss {:.6f} | time {:.2f}".format(
        epoch, reduced_loss, t1 - t0))
    return reduced_loss

 
@torch.no_grad()
def evaluate(data_loader, data, set_name='val'):
    model.eval()
    total_loss = 0
    t0 = time.time()
    pred_list = []
    y_list = []
    raw_pred_list = []
    raw_y_list = []
    for i, batch in enumerate(data_loader):
        time_set, loc_set, y = batch
        loss, pred, y, _  = model(time_set, loc_set, y) 
        total_loss += loss.item()
        raw_pred_list +=  pred.cpu().tolist()
        raw_y_list += y.cpu().tolist() 
        if data_dict['num_class'] > 1:
            true_rank, prob_rank = model.evaluate(pred, y)
            pred_list +=  prob_rank
            y_list += true_rank
        
    reduced_loss = total_loss / (data.len / args.batch_size)
    
    if data_dict['num_class'] > 1:
        eval_dict = evaluation_metrics(y_list, pred_list, raw_y_list, raw_pred_list, args.metric,args.multiclass)  
    else:
        eval_dict = evaluation_bi_metrics(raw_y_list, raw_pred_list) 

    t1 = time.time()
    # print("Valid Epoch {:03d} | Loss {:.6f} | time {:.2f}".format(
    #     epoch, reduced_loss, t1 - t0))
    return reduced_loss, eval_dict


for i in range(args.loop):
    print('i =', i, args.dataset)
    model, optimizer, result_file, token = prepare(args)
    model_state_file = 'models/{}/{}/{}.pth'.format(args.dataset, token, i)
    if i == 0 and os.path.exists(result_file):  # if result_file exist
        os.remove(result_file)

    # if not os.path.exists(model_state_file):
    bad_counter = 0
    loss_small = float('inf')
    try:
        print('begin training the predictor...')
        for epoch in range(0, args.epochs):
            epoch_start_time = time.time()
            train_loss = train(train_loader, train_data)
            valid_loss, eval_dict = evaluate(val_loader, val_data, 'val')
            if args.cri == 'loss':
                eval_metric = valid_loss
            else:
                eval_metric = 1-eval_dict[early_stop_crietira]
            if eval_metric < loss_small:
                loss_small = eval_metric
                bad_counter = 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print('Eval Epoch {:03d} train_loss: {:.6f}  valid_loss: {:.6f}'.format(epoch, train_loss, valid_loss))
                # print('{} '.format(args.metric) + '|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
                print('|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        print("training done")
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early, epoch',epoch)

    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    f = open(result_file,'a')
    wrt = csv.writer(f)

    print("Test using best epoch: {}".format(checkpoint['epoch']))
    val_loss, eval_dict = evaluate(val_loader, val_data, 'val')
    # print('{} '.format(args.metric) + '|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
    # val_res = [val_loss] + [eval_dict[k] for k in eval_dict]
    # print('test')
    _, eval_dict = evaluate(test_loader, test_data, 'test')
    print('{} '.format(args.metric) + '|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
    test_res = [eval_dict[k] for k in eval_dict]
    wrt.writerow([val_loss] + [0] + test_res)
    f.close()
     
if args.loop > 1:
    with open(result_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        arr = []
        for row in csv_reader:
            arr.append(list(map(float, row))) 
    arr = np.array(arr)
    arr = np.nan_to_num(arr)
    line_count = arr.shape[0]
    mean = [round(float(v),3) for v in arr.mean(0)]
    std = [round(float(v),3) for v in arr.std(0)]
    res = [str(mean[i]) +' ' + str(std[i]) for i in range(len(mean))]
    print(res)

    all_res_file = 'results/{}/res_stat.csv'.format(args.dataset)
    f = open(all_res_file,'a')
    wrt = csv.writer(f)
    wrt.writerow([token] + [line_count] + res)
    f.close()
