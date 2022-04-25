import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import dgl
import dgl.function as fn
import math
from sparsemax import *
import time

 

class EventMessagePassingEdge(nn.Module):
    def __init__(self, in_hid, out_hid, ext_dim, n_label, bias=True,
                 activation=None, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.fc1 = nn.Linear(3*in_hid, in_hid, bias=bias)
        self.activation = activation
        self.device = device
        self.n_label = n_label
        self.in_hid = in_hid
        self.fc2 = nn.Linear(1*in_hid + ext_dim, out_hid, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.xfc1 = nn.Linear(3*in_hid, in_hid, bias=bias)
        self.xfc2_list = nn.ModuleList([nn.Linear(2 * in_hid, 1, bias=False) for i in range(self.n_label)])

    def edge_finder(self,g, ext_feature, ref_embeds, doc_scores, label_idx=0):
        def apply_node(nodes):
            h_emp = nodes.data['h_tmp'].sum(-1)
            nodes.data.pop('h_tmp')
            node_scores = (h_emp!=0.0)*1.0
            node_mask = node_scores.unsqueeze(-1).repeat(1,self.in_hid)
            return {'h':nodes.data['h'] * node_mask,'h_mask':node_mask}

        def apply_edge(edges):
            evt = torch.cat([edges.src['h'], edges.data['e_h'], edges.dst['h']], dim=1)
            e_h = self.xfc1(evt)
            eh_w_ref = torch.cat((e_h, self.ref_embeds),dim=-1)
            e_h = torch.tanh(self.xfc2_list[label_idx](eh_w_ref))
            edge_mask = []
            self.edge_scores_list = []
            act_func = Sparsemax(0, self.device)
            e_by_graph = torch.split(e_h, self.batch_num_edges.cpu().tolist())
            doc_scores = torch.split(self.doc_scores.unsqueeze(-1), self.batch_num_edges.cpu().tolist())
            for i in range(len(e_by_graph)):
                edge_scores = act_func(e_by_graph[i] + doc_scores[i])
                self.edge_scores_list.append(edge_scores.cpu().detach().numpy())
                edge_scores = torch.tanh(edge_scores / 1e-12)
                edge_mask.append(edge_scores.repeat(1,self.in_hid))
                
            edge_mask = torch.cat(edge_mask,dim=0)
            return {'e_mask':edges.data['e_h'] * edge_mask}

   
        self.ext_feature = ext_feature
        self.ref_embeds = ref_embeds
        self.doc_scores = doc_scores
        self.batch_num_edges = g.batch_num_edges()
        g.apply_edges(apply_edge) 
        g.update_all(fn.v_mul_e('h', 'e_mask', 'm'), fn.sum('m', 'h_tmp'),apply_node)
        return self.edge_scores_list 
        
    def forward(self, g, ext_feature):  
        def apply_edge(edges):
            evt = torch.cat([edges.src['h'], edges.data['e_h'], edges.dst['h']], dim=1)
            e_h = self.fc1(self.dropout(evt))
            eh_w_doc = self.dropout(torch.cat((e_h, self.ext_feature),dim=-1))
            e_h = torch.relu(self.fc2(eh_w_doc))
            e_h = self.dropout(e_h)
            if self.activation is not None:
                e_h = self.activation(e_h)
            return {'e_h': e_h}
        
        def apply_edge2(edges): 
            e_h = self.fc3(edges.data['e_h'])
            e_h = self.dropout(e_h)
            # if self.activation is not None:
            #     e_h = self.activation(e_h)
            return {'e_h': e_h}

        self.ext_feature = ext_feature
        g.apply_edges(apply_edge) 
 
class EventMessagePassingNode(nn.Module):
    def __init__(self, in_hid, out_hid, bias=True,
                 activation=None, self_loop=False, dropout=0.2, emb_dim=300, device=torch.device('cpu')):
        super().__init__()
        self.activation = activation
        self.self_loop = self_loop
        self.device = device
        self.node_nn = nn.Linear(in_hid, out_hid, bias=bias) # w@f(e_s,e_r) inverse
        self.dropout = nn.Dropout(dropout)
        if self.self_loop:
            self.loop_nn = nn.Linear(out_hid, out_hid, bias=bias) 
    
    def forward(self, g):  
        def apply_func(nodes):
            h = self.node_nn(nodes.data['h'])
            h = h * nodes.data['norm']
            if self.self_loop:
                h = self.loop_nn(g.ndata['h'])
            if self.dropout is not None:
                h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}
        g.update_all(fn.v_mul_e('h', 'e_h', 'm'), fn.sum('m', 'h'), apply_func) 
         