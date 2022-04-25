import torch.nn as nn
import numpy as np
import dgl
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from propagations import *
import time
from collections import Counter
from sparsemax import *

 

class DocEmbLearning(nn.Module): # memory issue
    def __init__(self, h_dim, emb_dim, dropout, seq_len=10, device=torch.device('cpu') ):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.emb_dim = emb_dim
        # self.sent_learning = nn.LSTM(emb_dim, h_dim, batch_first=True, num_layers=1, bidirectional=True)
        self.sent_learning = nn.GRU(emb_dim, h_dim, batch_first=True, num_layers=1, bidirectional=True)
        self.out_layer = nn.Linear(2*h_dim, h_dim)
        self.device = device

    def forward(self, doc_emb_list):
        maxlen = 20
        # all_
        sent_embs_tensor = torch.zeros(len(doc_emb_list), maxlen, self.emb_dim)
        len_non_zero = []
        for i, emb in enumerate(doc_emb_list):
            length = len(emb)
            len_non_zero.append(length)
            sent_embs_tensor[i, range(length), :] = torch.FloatTensor(emb)
        
        sent_embs_tensor = sent_embs_tensor.to(self.device)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(sent_embs_tensor,
                                                            len_non_zero,
                                                            batch_first=True,
                                                            enforce_sorted=False)
        output, hn = self.sent_learning(packed_input) 
        hn = hn.permute(1,0,2).contiguous()
        sent_embs_learned = hn.view(hn.size(0), -1)
        sent_embs_learned = self.out_layer(sent_embs_learned)
        return sent_embs_learned

class DocEmbAverage(nn.Module):
    def __init__(self, h_dim, emb_dim, dropout, seq_len=10, device=torch.device('cpu') ):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.emb_dim = emb_dim
        self.out_layer = nn.Linear(emb_dim, h_dim)
        self.device = device

    def forward(self, doc_emb_list):
        maxlen = 20
        sent_embs_tensor = torch.zeros(len(doc_emb_list), self.emb_dim)
        for i, emb in enumerate(doc_emb_list):
            sent_embs_tensor[i, :] = torch.from_numpy(emb.mean(0))
        sent_embs_learned = self.out_layer(sent_embs_tensor.to(self.device))
        return sent_embs_learned

 
class multilevel_learning(nn.Module):
    def __init__(self, h_dim, emb_dim, dropout, num_nodes, num_rels, seq_len=10, 
    maxpool=1, attn='', n_label=20, device=torch.device('cpu'), multiclass=False, 
    emb_mod='lstm',node_layer=1,text_dim=64):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.emb_dim = emb_dim # sentence embedding dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.device = device
        self.n_label = n_label
        self.grads = {}
        self.up_grad = True
        self.multiclass = multiclass
        if self.multiclass:
            self.n_label = 1
        self.emb_mod = emb_mod
        self.learned_emb_dim = emb_dim
        if emb_mod == 'lstm':
            self.DocEmbLearning = DocEmbLearning(h_dim, emb_dim, dropout, seq_len, device)
            self.learned_emb_dim = text_dim
        elif emb_mod == 'mean':
            pass
        self.prop_edge = EventMessagePassingEdge(h_dim,h_dim,h_dim,n_label, bias=True,
                                                activation=F.relu, dropout=dropout,device=self.device)
        self.prop_node1 = EventMessagePassingNode(h_dim,h_dim,activation=F.relu, dropout=dropout,device=self.device)
        self.node_layer = node_layer
        if self.node_layer > 1:
            self.prop_node2 = EventMessagePassingNode(h_dim,h_dim,activation=F.relu, dropout=dropout,device=self.device)

        self.count_nn = nn.Linear(292, h_dim)
        self.count_bn = nn.BatchNorm1d(h_dim, affine=True)
        self.count_doc_bn = nn.BatchNorm1d(h_dim, affine=True)
        self.count_doc_nn = nn.Linear(text_dim+h_dim, h_dim)
        self.doc_attn = ScaledDotProductAttention(self.learned_emb_dim, text_dim, text_dim ** 0.5)
        self.sent_attn1 = nn.Linear(emb_dim+h_dim,h_dim)
        self.sent_attn2_list = nn.ModuleList([nn.Linear(h_dim,1) for i in range(self.n_label)])
        self.doc_bi_nn = nn.Bilinear(emb_dim,h_dim,1,bias=False)
    
    def forward(self, time_set, loc_set, ent_embeds, rel_embeds, graph_dict, text_dict, count_dict, doc_embeds, doc_emb_list, time_of_locs):
        key_list = []
        len_non_zero = []
        nonzero_idx = torch.nonzero(time_set, as_tuple=False).view(-1)
        time_set = time_set[nonzero_idx]   
        time_set = time_set.cpu().numpy()
        loc_set = loc_set.cpu().numpy()
        for i in range(len(time_set)):
            tim = time_set[i]
            loc = loc_set[i]
            try:
                length = time_of_locs[loc].index(tim)
            except:
                length = self.seq_len if tim >= self.seq_len else tim
            if self.seq_len <= length:
                hist_time = time_of_locs[loc][length - self.seq_len:length]
                key_list.append(list(zip([loc]*len(hist_time),hist_time)))
                len_non_zero.append(self.seq_len)
            else:
                hist_time = time_of_locs[loc][:length]
                key_list.append(list(zip([loc]*len(hist_time),hist_time)))
                len_non_zero.append(length)
        if 0 in len_non_zero:
            len_non_zero = len_non_zero[:len_non_zero.index(0)]
            key_list = key_list[:len(len_non_zero)]
         
        all_hist = [item for sublist in key_list for item in sublist]
        g_list = [graph_dict[key] for key in all_hist]
        batched_g = dgl.batch(g_list) 
        batched_g = batched_g.to(self.device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        rel_data = batched_g.edata['r'].to(self.device)
        batched_g.edata['e_h'] = rel_embeds.index_select(0, rel_data.view(-1))
        doc_ids = [list(text_dict[key]) for key in all_hist]
        docs_len = list(map(len, doc_ids))
        u_doc_ids = [list(set(v)) for v in doc_ids]
        u_docs_len = list(map(len, u_doc_ids))
        u_doc_ids_day = [j for sub in u_doc_ids for j in sub] # for trace back
        doc_ids_all = [j for sub in doc_ids for j in sub] # for trace back
        day_idx = []
        for i, row in enumerate(doc_ids):
            day_idx.append([u_doc_ids[i].index(v) for v in row])
         
       
        if self.emb_mod == 'mean':
            doc_embeds = doc_embeds[u_doc_ids_day].cpu()
        elif self.emb_mod == 'lstm':
            doc_embeds = self.DocEmbLearning(doc_emb_list[u_doc_ids_day]).cpu()
        u_doc_by_day = torch.split(doc_embeds, u_docs_len)
        max_doc_day = max(u_docs_len)
        doc_emb_by_day = torch.zeros(len(u_docs_len),max_doc_day,doc_embeds.size(-1))
        for i,row in enumerate(u_doc_by_day):
            doc_emb_by_day[i,range(u_docs_len[i]),:] = row
        doc_emb_by_day = doc_emb_by_day.to(self.device)
        doc_emb_by_day_attn, _ = self.doc_attn(doc_emb_by_day) # self attention
        doc_emb_by_day_attn = doc_emb_by_day_attn.to(self.device)
        all_doc_emb = torch.zeros(len(doc_ids_all),doc_emb_by_day_attn.size(-1))
        start_idx = 0 # day
        for i, row in enumerate(day_idx):
            all_doc_emb[start_idx:start_idx+len(row),:] = doc_emb_by_day_attn[i,row,:]
            start_idx += len(row)
        all_doc_emb = all_doc_emb.to(self.device)

        start_idx = 0
        count_features = np.zeros((len(doc_ids_all),292))
        for i in range(len(all_hist)):
            count_features[start_idx:start_idx+docs_len[i],:] = count_dict[all_hist[i]]
            start_idx += docs_len[i]
        count_features = torch.FloatTensor(count_features).to(self.device)

        count_features_res = self.count_nn(count_features)
        count_features = count_features_res
        count_features = self.count_bn(torch.relu(count_features))
        count_doc_inp = self.dropout(torch.cat((count_features,all_doc_emb),dim=-1))
        count_doc_feature  = torch.relu(self.count_doc_nn(count_doc_inp))
        count_doc_feature += count_features_res
        count_doc_feature = self.count_doc_bn(count_doc_feature)
        self.prop_edge(batched_g, count_doc_feature)
        self.prop_node1(batched_g)
        if self.node_layer > 1:
            self.prop_node2(batched_g)

        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')

        global_edge_info = torch.cat((global_node_info, global_edge_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim) 
        embed_seq_tensor = embed_seq_tensor.to(self.device)
        start_idx = 0
        for i, times in enumerate(key_list):
            embed_seq_tensor[i, range(len_non_zero[i]), :] = global_edge_info[start_idx:start_idx+len_non_zero[i]]
            start_idx = start_idx+len_non_zero[i]
 
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero
 
    def explain(self, time_set, loc_set, ent_embeds, rel_embeds, graph_dict, text_dict, count_dict, doc_embeds, doc_emb_list, time_of_locs, ref_embeds, label_idx):
        key_list = []
        len_non_zero = []
        nonzero_idx = torch.nonzero(time_set, as_tuple=False).view(-1)
        time_set = time_set[nonzero_idx]   
        time_set = time_set.cpu().numpy()
        loc_set = loc_set.cpu().numpy()
        for i in range(len(time_set)):
            tim = time_set[i]
            loc = loc_set[i]
            try:
                length = time_of_locs[loc].index(tim)
            except:
                length = self.seq_len if tim >= self.seq_len else tim
            if self.seq_len <= length:
                hist_time = time_of_locs[loc][length - self.seq_len:length]
                key_list.append(list(zip([loc]*len(hist_time),hist_time)))
                len_non_zero.append(self.seq_len)
            else:
                hist_time = time_of_locs[loc][:length]
                key_list.append(list(zip([loc]*len(hist_time),hist_time)))
                len_non_zero.append(length)
        if 0 in len_non_zero:
            len_non_zero = len_non_zero[:len_non_zero.index(0)]
            key_list = key_list[:len(len_non_zero)]
         
        all_hist = [item for sublist in key_list for item in sublist]
        g_list = [graph_dict[key] for key in all_hist]
        batched_g = dgl.batch(g_list) 
        batched_g = batched_g.to(self.device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        rel_data = batched_g.edata['r'].to(self.device)
        batched_g.edata['e_h'] = rel_embeds.index_select(0, rel_data.view(-1))

        doc_ids = [list(text_dict[key]) for key in all_hist]
        docs_len = list(map(len, doc_ids))
        u_doc_ids = [list(set(v)) for v in doc_ids]
        u_docs_len = list(map(len, u_doc_ids))
        u_doc_ids_day = [j for sub in u_doc_ids for j in sub] # for trace back
        doc_ids_all = [j for sub in doc_ids for j in sub] # for trace back
        day_idx = []
        for i, row in enumerate(doc_ids):
            day_idx.append([u_doc_ids[i].index(v) for v in row])

        with torch.no_grad():
            if self.emb_mod == 'mean':
                doc_embeds = doc_embeds[u_doc_ids_day].cpu()
            elif self.emb_mod == 'lstm':
                doc_embeds = self.DocEmbLearning(doc_emb_list[u_doc_ids_day]).cpu()
        ref_embeds = ref_embeds.view(-1, 1, ref_embeds.size(-1)).to(torch.device('cpu'))

        u_doc_by_day = torch.split(doc_embeds, u_docs_len)
        max_doc_day = max(u_docs_len)
        doc_emb_by_day = torch.zeros(len(u_docs_len),max_doc_day,doc_embeds.size(-1))
        ref_embeds_repeat = torch.zeros(len(u_docs_len),max_doc_day,ref_embeds.size(-1))

        for i,row in enumerate(u_doc_by_day):
            doc_emb_by_day[i,range(u_docs_len[i]),:] = row
            # print(row.shape,'row')
            ref_embeds_repeat[i,range(u_docs_len[i]),:] = ref_embeds[i].repeat(u_docs_len[i],1)
        # print(doc_emb_by_day.shape) 
        doc_emb_by_day = doc_emb_by_day.to(self.device)
        ref_embeds_repeat = ref_embeds_repeat.to(self.device)

        doc_ref_concat = torch.cat((ref_embeds_repeat,doc_emb_by_day),dim=-1)
        doc_scores = self.sent_attn2_list[label_idx](torch.tanh(self.sent_attn1(doc_ref_concat)))

        doc_scores = doc_scores.squeeze(-1)
        act_func = Sparsemax(-1, self.device)
        float_doc_scores = act_func(doc_scores)
        doc_scores = torch.tanh(float_doc_scores / 1e-12)
        doc_mask = doc_scores.unsqueeze(-1).repeat(1,1,doc_emb_by_day.size(-1))

        doc_emb_by_day = doc_emb_by_day * doc_mask
         
        with torch.no_grad():
            doc_emb_by_day_attn, _ = self.doc_attn(doc_emb_by_day) # self attention

        doc_emb_by_day_attn = doc_emb_by_day_attn.to(self.device)
        all_doc_emb = torch.zeros(len(doc_ids_all),doc_emb_by_day_attn.size(-1))
        all_doc_scores = torch.zeros((len(doc_ids_all)))

        all_ref_embeds = torch.zeros(len(doc_ids_all),ref_embeds.size(-1))
        start_idx = 0 # day
        for i, row in enumerate(day_idx):
            all_doc_emb[start_idx:start_idx+len(row),:] = doc_emb_by_day_attn[i,row,:]
            all_doc_scores[start_idx:start_idx+len(row)] = float_doc_scores[i,row]
            all_ref_embeds[start_idx:start_idx+len(row),:] = ref_embeds[i].repeat(len(row),1)
            start_idx += len(row)
        all_doc_emb = all_doc_emb.to(self.device)
        all_ref_embeds = all_ref_embeds.to(self.device)
        all_doc_scores = all_doc_scores.to(self.device)
        saved_doc_scores = all_doc_scores.cpu().detach().numpy()

        with torch.no_grad():
            start_idx = 0
            count_features = np.zeros((len(doc_ids_all),292))
            for i in range(len(all_hist)):
                count_features[start_idx:start_idx+docs_len[i],:] = count_dict[all_hist[i]]
                start_idx += docs_len[i]
            count_features = torch.FloatTensor(count_features).to(self.device)
            count_features_res = self.count_nn(count_features)
            count_features = count_features_res
            count_features = self.count_bn(torch.relu(count_features))
            count_doc_inp = self.dropout(torch.cat((count_features,all_doc_emb),dim=-1))
            count_doc_feature  = torch.relu(self.count_doc_nn(count_doc_inp))
            count_doc_feature += count_features_res
            count_doc_feature = self.count_doc_bn(count_doc_feature)
         
        edge_scores_list = self.prop_edge.edge_finder(batched_g, count_doc_feature, all_ref_embeds, all_doc_scores, label_idx)
         
        with torch.no_grad():
            self.prop_edge(batched_g, count_doc_feature)
            self.prop_node1(batched_g)
            if self.node_layer > 1:
                self.prop_node2(batched_g)

        batched_g.edata['e_h'] = batched_g.edata['e_h'] * batched_g.edata['e_mask'] 
        batched_g.ndata['h'] = batched_g.ndata['h'] * batched_g.ndata['h_mask'] 

        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')
         
        global_edge_info = torch.cat((global_node_info, global_edge_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim) 
        embed_seq_tensor = embed_seq_tensor.to(self.device)
        start_idx = 0
        for i, times in enumerate(key_list):
            embed_seq_tensor[i, range(len_non_zero[i]), :] = global_edge_info[start_idx:start_idx+len_non_zero[i]]
            start_idx = start_idx+len_non_zero[i]
 
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero, g_list, edge_scores_list, saved_doc_scores 

 
     
 
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, hi, ho, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        n_head = 1
        self.w_qs = nn.Linear(hi, ho, bias=False)
        self.w_ks = nn.Linear(hi, ho, bias=False)
        self.w_vs = nn.Linear(hi, ho, bias=False)

    def forward(self, inp, mask=None):
        q = self.w_qs(inp)
        k = self.w_ks(inp)
        v = self.w_vs(inp)
        attn = torch.matmul(q / self.temperature, k.transpose(-1, -2)) # transpose last two dims

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general', query_dimensions=None):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general','add']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            if query_dimensions is None:
                self.linear_in = nn.Linear(dimensions, dimensions, bias=True)
            else:
                self.linear_in = nn.Linear(query_dimensions, dimensions, bias=True)

        if self.attention_type == 'add':
            if query_dimensions is None:
                self.linear_in = nn.Linear(2* dimensions, dimensions//2, bias=True)
                self.v = nn.Parameter(torch.Tensor(dimensions//2, 1))
            else:
                self.linear_in = nn.Linear(query_dimensions+dimensions, int(query_dimensions+dimensions), bias=True)
                self.v = nn.Parameter(torch.Tensor(int(query_dimensions+dimensions), 1))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        
        query_len = context.size(1)
        # print(query.size(),context.size(), query_len)
        if self.attention_type == 'add':
            querys = query.repeat(1,query_len,1) # [B,H] -> [T,B,H]
            feats = torch.cat((querys, context), dim=-1) # [B,T,H*2]
            energy = self.tanh(self.linear_in(feats)) # [B,T,H*2] -> [B,T,H]
            # compute attention scores
            v = self.v.t().repeat(batch_size,1,1) # [H,1] -> [B,1,H]
            energy = energy.permute(0,2,1)#.contiguous() #[B,H,T]
            attention_weights = torch.bmm(v, energy) # [B,1,H]*[B,H,T] -> [B,1,T]
            # weight values
            mix = torch.bmm(attention_weights, context)#.squeeze(1) # [B,1,T]*[B,T,H] -> [B,H]
            # concat -> (batch_size * output_len, 2*dimensions)
            combined = torch.cat((mix, query), dim=2)
            combined = combined.view(batch_size * output_len, 2 * dimensions)
            # Apply linear_out on every 2nd dimension of concat
            # output -> (batch_size, output_len, dimensions)
            output = self.linear_out(combined).view(batch_size, output_len, dimensions)
            output = self.tanh(output)
            return output, attention_weights
            
        elif self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)
        return output, attention_weights
