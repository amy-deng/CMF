import numpy as np
import os
import random
from math import log
import scipy.sparse as sp
from scipy import stats
import collections
import dgl
from dgl.data.utils import save_graphs,load_graphs

import torch
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score, hamming_loss, accuracy_score, roc_auc_score
from sklearn.metrics import jaccard_score

'''
load data
'''
def load_loc_data(args):
    with open(args.path + args.dataset + '/loc_entity2id.txt') as f:
        num_ents = sum(1 for _ in f)
    with open(args.path + 'cameo.txt') as f:
        num_rels = sum(1 for _ in f)
    with open(args.path + args.dataset + '/loc2id.txt') as f:
        num_locs = sum(1 for _ in f)
    r = {'num_ents':num_ents, 'num_rels':num_rels, 'num_locs':num_locs}
    print(r)
    
    
    data_file = '/data_graph.bin'
    label_file = '/data_label.pkl'
    count_file = '/data_count.pkl'
    glist, _ = load_graphs(args.path + args.dataset + data_file)

    with open(args.path + args.dataset + label_file, 'rb') as f: # lists
        label_dict = pickle.load(f)
    with open(args.path + args.dataset + count_file, 'rb') as f: # lists
        count_data = pickle.load(f)
    

    locs = label_dict['loc']
    times = label_dict['time']
 
    key = list(zip(locs, times))

    r['count_dict'] = dict(zip(key,count_data))
    print('#glist =',len(glist))
    time_of_locs = []
    for iloc in range(num_locs):
        time_of_locs.append(list(set(sorted([t for l,t in key if l == iloc]))))
    r['time_of_locs'] = time_of_locs
    graph_dict = dict(zip(key,glist))
    r['graph_dict'] = graph_dict
    if args.model in ['glean','gleanfull','dyngcn']:
        wglist, _ = load_graphs(args.path + args.dataset + '/xwg_data.bin')
        r['word_graph_dict'] = dict(zip(key,wglist))

    r['text_dict'] = dict(zip(key,label_dict['text_id']))
 
    if 'gdelt_US' in args.dataset:
        with open(args.path + args.dataset + '/label-covid19-{}-.08.pkl'.format(args.horizon),'rb') as f: # -2 used to be
            covid19_label = pickle.load(f) 
        labels = covid19_label['binary']
        covid19_label_dict = collections.Counter(labels)
        class_weight = [covid19_label_dict[k]/len(labels) for k in sorted(covid19_label_dict)]
        class_weight = torch.FloatTensor(class_weight)
        r['num_class'] = len(covid19_label_dict)
        labels = torch.LongTensor(labels)#.unsqueeze(-1)
    else:
        labels = label_dict['label'] 
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        labels = torch.FloatTensor(labels)
        class_weight = labels.mean(0)
        print('all class_weight',class_weight)
        if args.eid >= 0:
            target = [args.eid]
        else:
            target = [13]
        print('target event index =',target)
        labels = labels[:,target]
        r['num_class'] = labels.shape[-1] 
        class_weight = class_weight[target]
             
    print('class_weight',class_weight)
    r['class_weight'] = class_weight

    if args.spl and r['num_class'] == 1:
        print('=============down sampling==============')
        if not os.path.exists(args.path + args.dataset + '/down-sampling{}.pkl'.format(args.eid)):
            l = []
            thr = 0.3
            print(len(labels),'n labels')
            for i,label in enumerate(labels):
                if label[0].item() == 0:
                    if random.random() > thr:
                        l.append(i)
                else:
                    l.append(i)
            with open(args.path + args.dataset + '/down-sampling{}.pkl'.format(args.eid),'wb') as f:
                pickle.dump(l,f) 

        with open(args.path + args.dataset + '/down-sampling{}.pkl'.format(args.eid),'rb') as f:
            label_indices = pickle.load(f) 
        labels = labels[label_indices]
        class_weight = labels.mean(0)
        r['class_weight'] = class_weight
        print('class_weight after down sampling',class_weight,'#',len(label_indices))
        times = np.array(times)[label_indices]
        locs = np.array(locs)[label_indices]
 
    cut1 = int(len(times) * args.train)
    cut2 = int(len(times) * (args.train+args.val))
    print('train',cut1,'val',cut2-cut1)
    indices = list(range(len(times)))
    if args.shuffle:
        c = list(zip(times, locs, indices))
        random.Random(42).shuffle(c) # fix train, val and test sets  
        times, locs, indices = zip(*c) 
        indices = list(indices)
    times = torch.tensor(list(times))
    locs = torch.tensor(list(locs))
    r['train_time'], r['train_loc'] = times[:cut1], locs[:cut1]  
    r['train_y'] = labels[indices[:cut1]]
    r['val_time'], r['val_loc'] = times[cut1:cut2], locs[cut1:cut2]  
    r['test_time'], r['test_loc'] = times[cut2:], locs[cut2:]  
    r['val_y'] = labels[indices[cut1:cut2]]
    r['test_y'] = labels[indices[cut2:]]
    return r
 

'''
Loss function
'''
def mutual_infomation_loss(pred, target, reduction='mean'):
    mi_l = []
    for i in range(pred.size(-1)): # pred (b, C)
        mi_l.append(mutual_information(pred[:,i], target[:,i]))
    if reduction == 'mean':
        return sum(mi_l) / len(mi_l)
    elif reduction == 'sum':
        return sum(mi_l) 
    else:
        return mi_l


def mutual_information(x, y):
    sum_mi = 0
    x_value_list = torch.unique(x)
    y_value_list = torch.unique(y)
    Px = torch.tensor([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = torch.tensor([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = torch.tensor([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*torch.log( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi

   
'''
Evaluation metrics
'''
def evaluation_bi_metrics(true_l, prob_l):
    metric = 'binary'
    r = {} 
    true_l = np.array(true_l).reshape(-1)
    prob_l = np.array(prob_l).reshape(-1)
    pred_l = (prob_l > 0.5) * 1.0
    # r['precision'] = precision_score(true_l, pred_l, average=metric)
    # r['recall'] = recall_score(true_l, pred_l, average=metric)
    r['f1'] = f1_score(true_l, pred_l, average=metric)
    # r['f2'] = fbeta_score(true_l, pred_l, average=metric, beta=2)
    r['accuracy'] =  accuracy_score(true_l, pred_l)
    try:
        r['auc'] = roc_auc_score(true_l, prob_l)
    except:
        pass
    return r
 

def evaluation_metrics(true_rank_l, prob_rank_l, raw_y_list, raw_pred_list, metric='weighted',multiclass=False):
    r = {}
    m = MultiLabelBinarizer().fit(true_rank_l)
    m_actual = m.transform(true_rank_l)
    m_predicted = m.transform(prob_rank_l)
    y = np.array(raw_y_list) 
        
    prob = np.array(raw_pred_list) 
    r['acc'] = 1 - hamming_loss(m_actual, m_predicted)
    # r['w-prec'] = precision_score(m_actual, m_predicted, average=metric)
    # r['w-rec'] = recall_score(m_actual, m_predicted, average=metric)
    r['w-f1'] = f1_score(m_actual, m_predicted, average=metric)
    # r['w-f2'] = fbeta_score(m_actual, m_predicted, average=metric, beta=2)
    if multiclass:
        r['w-auc-ovo'] = roc_auc_score(y, prob, average=metric, multi_class='ovo')
        r['w-auc-ovr'] = roc_auc_score(y, prob, average=metric, multi_class='ovr')
    else:
        r['w-auc'] = roc_auc_score(y, prob, average=metric)
    return r


def evaluate_explainer_basic(p, pred, x_p, x_pred, metric='binary'):
    r = {}
    try:
        auc = roc_auc_score(np.array(pred), np.array(x_p))
        r['auc'] = auc
    except:
        pass
    acc = accuracy_score(pred, x_pred)
    r['accuracy'] = acc 
    try:
        r['prec'] = precision_score(pred, x_pred, average=metric)
        r['rec'] = recall_score(pred, x_pred, average=metric)
        r['f1'] = f1_score(pred, x_pred, average=metric)
    except:
        pass 
    return r
