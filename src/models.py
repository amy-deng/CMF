import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from modules import *
import time
import math
import random
import numpy as np

 
class CMF(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, num_class, class_weight, 
                dropout=0.2, seq_len=5, maxpool=1, use_edge_node=0, use_gru=1, 
                attn='', weight_loss=False, n_label=20, emb_dim=300,device=torch.device('cpu'),
                multiclass=False, emb_mod='lstm',node_layer=1,text_dim=64):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.num_class = num_class
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
        self.class_weight = class_weight
        self.weight_loss = weight_loss
        self.device = device
        self.n_label = num_class
        self.doc_embeds = None
        self.multiclass = multiclass
        self.text_dict = None
        self.count_dict = None
        self.doc_emb_list = None
        self.sent_embs_dict = None
        self.graph_dict = None
        self.time_of_locs = None # available time for each location, a list of list
        self.aggregator= multilevel_learning(h_dim, emb_dim, dropout, num_ents, num_rels, seq_len, maxpool, attn, self.n_label, self.device, self.multiclass, emb_mod,node_layer,text_dim)
        if use_gru:
            self.encoder = nn.GRU(2*h_dim, h_dim, batch_first=True, num_layers=1, bidirectional=False)
        else:
            self.encoder = nn.RNN(2*h_dim, h_dim, batch_first=True, num_layers=1, bidirectional=False)
        self.out_layer = nn.Sequential(nn.Linear(1*h_dim, h_dim),nn.ReLU(),nn.Linear(h_dim, num_class)) # nn.Dropout(dropout) after relu
        self.temp_nn = nn.Linear(3*h_dim, h_dim)
        self.threshold = 0.5
        if self.multiclass:
            if self.weight_loss:
                self.criterion = nn.CrossEntropyLoss(weight=1-self.class_weight)
            else:
                self.criterion = nn.CrossEntropyLoss()
            self.out_func = nn.Softmax(dim=-1)
        else:
            self.criterion = F.binary_cross_entropy
            self.out_func = torch.sigmoid
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, time_set, loc_set, y):
        pred, idx, _ = self.get_pred_embeds(time_set, loc_set, self.graph_dict)
        y_true = y[idx]
        if self.multiclass:
            loss = self.criterion(pred, y_true)
            pred = F.softmax(pred,-1)
        else:
            pred = F.sigmoid(pred)
            if self.weight_loss:
                loss = self.criterion(pred, y_true,reduction='none') # should be 1 0 vector TODO
                weight = y_true/(2*self.class_weight) + (1-y_true)/(2*(1-self.class_weight))
                loss = torch.mean(loss * weight)
            else:
                # loss = F.cross_entropy(opred, torch.argmax(y_true, dim=1), reduction='sum')
                loss = self.criterion(pred, y_true)
        return loss, pred, y_true, idx
 
    def get_pred_embeds(self, time_set, loc_set, graph_dict):
        sorted_t, idx = time_set.sort(0, descending=True)  #TODO
        embed_seq_tensor, len_non_zero = self.aggregator(sorted_t, loc_set[idx], self.ent_embeds, 
                                    self.rel_embeds,
                                    graph_dict,
                                    self.text_dict,
                                    self.count_dict,
                                    self.doc_embeds,
                                    self.doc_emb_list,
                                    self.time_of_locs)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                            len_non_zero,
                                                            batch_first=True,
                                                            enforce_sorted=False)
        _, feature = self.encoder(packed_input)
        feature = feature.squeeze(0)
        feature = torch.cat((feature, torch.zeros(len(time_set) - len(feature), feature.size(-1)).to(self.device)), dim=0)
        pred = self.out_layer(feature)
        return pred, idx, feature
 
    def get_pred_embeds_w_temporal(self, time_set, loc_set, graph_dict):
        sorted_t, idx = time_set.sort(0, descending=True)  #TODO
        embed_seq_tensor, len_non_zero = self.aggregator(sorted_t, loc_set[idx], self.ent_embeds, 
                                    self.rel_embeds,
                                    graph_dict,
                                    self.text_dict,
                                    self.count_dict,
                                    self.doc_embeds,
                                    self.doc_emb_list,
                                    self.time_of_locs)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                            len_non_zero,
                                                            batch_first=True,
                                                            enforce_sorted=False)
        _, feature = self.encoder(packed_input)
        feature = feature.squeeze(0)
        feature = torch.cat((feature, torch.zeros(len(time_set) - len(feature), feature.size(-1)).to(self.device)), dim=0)
        pred = self.out_layer(feature)
        return pred, idx, feature, embed_seq_tensor, len_non_zero
 
    def evaluate(self, pred, y): # y is a binary vector
        if self.multiclass:
            _, y_pred_tags = torch.max(pred, dim = -1)
            nonzero_y_idx = y.unsqueeze(-1).cpu().tolist()
            nonzero_pred_idx = y_pred_tags.unsqueeze(-1).cpu().tolist()
        else:
            sorted_pred, sorted_pred_idx = pred.sort(1, descending=True)
            sorted_pred = torch.where(sorted_pred > self.threshold, sorted_pred, torch.zeros(sorted_pred.size()).to(self.device))
            nonzero_pred_len = torch.count_nonzero(sorted_pred, 1)
            sorted_pred_idx = sorted_pred_idx.cpu().tolist()
            nonzero_pred_len = nonzero_pred_len.cpu().tolist()
            nonzero_pred_idx = []
            for i in range(len(sorted_pred_idx)):
                nonzero_pred_idx.append(sorted_pred_idx[i][:nonzero_pred_len[i]])
            sorted_y, sorted_y_idx = y.sort(1, descending=True)
            nonzero_y_len = torch.count_nonzero(sorted_y, 1)
            sorted_y_idx = sorted_y_idx.cpu().tolist()
            nonzero_y_len = nonzero_y_len.cpu().tolist()
            nonzero_y_idx = []
            for i in range(len(sorted_y_idx)):
                nonzero_y_idx.append(sorted_y_idx[i][:nonzero_y_len[i]])
        return nonzero_y_idx, nonzero_pred_idx

    def explain(self, time_set, loc_set, y, label_idx):
        with torch.no_grad():
            pred, idx, feature, embed_seq_tensor, len_non_zero = self.get_pred_embeds_w_temporal(time_set, loc_set, self.graph_dict)
        embed_seq_tensor = torch.cat((embed_seq_tensor, torch.zeros(len(time_set) - len(embed_seq_tensor), self.seq_len, embed_seq_tensor.size(-1)).to(self.device)), dim=0)
        ref_embeds = torch.cat((feature.unsqueeze(1).repeat(1,self.seq_len,1),embed_seq_tensor),dim=-1)
        ref_embeds = self.temp_nn(ref_embeds)
        sorted_t, idx = time_set.sort(0, descending=True) 
        embed_seq_tensor, len_non_zero, g_list, edge_scores_list, saved_doc_scores = self.aggregator.explain(
                                                        sorted_t, loc_set[idx], 
                                                        self.ent_embeds, 
                                                        self.rel_embeds,
                                                        self.graph_dict, 
                                                        self.text_dict,
                                                        self.count_dict,
                                                        self.doc_embeds,
                                                        self.doc_emb_list,
                                                        self.time_of_locs,
                                                        ref_embeds, label_idx)

        with torch.no_grad(): # using explanatory graphs
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True,
                                                               enforce_sorted=False)
            _, x_feature = self.encoder(packed_input)
            x_feature = x_feature.squeeze(0)
            x_feature = torch.cat((x_feature, torch.zeros(len(time_set) - len(x_feature), x_feature.size(-1)).to(self.device)), dim=0)
            x_pred = self.out_layer(x_feature)

        r = {}
        r['len_non_zero'] = len_non_zero
        r['g_list'] = g_list
        r['edge_scores_list'] = edge_scores_list
        r['saved_doc_scores'] = saved_doc_scores
        r['time_set'] = time_set[idx]
        r['loc_set'] = loc_set[idx]
        r['y'] = y[idx] 
        if self.multiclass:
            _, pred_label = torch.max(pred, dim = -1)
            _, x_pred_label = torch.max(x_pred, dim = -1)
            # loss = - mutual_information(pred_label, x_pred_label)  
            loss = - mutual_infomation_loss(pred_label, x_pred_label)  
            
            r['label'] = 'multiclass'
            r['pred'] = pred
            r['pred_label'] = pred_label
            r['x_pred'] = x_pred
            r['x_pred_label'] = x_pred_label
            return loss, r
        else:
            pred = F.sigmoid(pred)
            x_pred = F.sigmoid(x_pred)
            pred_label = (pred > 0.5).float() * 1
            x_pred_label = (x_pred > 0.5).float() * 1
            x = pred_label[:,label_idx]
            y1 = x_pred_label[:,label_idx]
            # loss = - mutual_information(x, y1) # + doc_entropy + evt_entropy
            loss = - mutual_infomation_loss(x, y1)  

            r['label'] = 'multilabel' + str(label_idx)
            r['pred'] = pred[:,label_idx]
            r['pred_label'] = pred_label[:,label_idx]
            r['x_pred'] = x_pred[:,label_idx]
            r['x_pred_label'] = x_pred_label[:,label_idx] 
            return loss, r
 