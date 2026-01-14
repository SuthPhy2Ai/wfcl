import math
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import e3nn
import e3nn.o3
import e3nn.nn

logger = logging.getLogger(__name__)


class cry_config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)




class CausalSelfAttention_masked_for_formula(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)


        self.adj_proj1 = nn.Linear(12, 2 * config.n_head, bias=False)
        self.adj_act = nn.GELU()
        self.adj_proj2 = nn.Linear(2 * config.n_head, config.n_head, bias=False)
        self.adj_weight = nn.Parameter(torch.tensor(50.0))
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, x_padding_judge, adj, is_formula=None):
        B, T, C = x.size()

        x_padding_judge = 1.0 - x_padding_judge
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=2)
        x_padding_judge = x_padding_judge @ x_padding_judge.transpose(-2, -1)
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=1)
        x_padding_judge = torch.tile(x_padding_judge, [1, self.n_head, 1, 1])

        ############# 额外cls ##############################
        # if is_formula:
        #     x_padding_judge[:, :, 7:, 0] = 0.0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        ######## use adj to fix att  ####################
        att_fix = self.adj_act(self.adj_proj1(adj))
        att_fix = self.adj_proj2(att_fix)
        att_fix = torch.einsum('ijkl->iljk', att_fix)

        ##########Dropkey#####################
        att_full = torch.ones_like(att)
        att_full = self.attn_drop(att_full)
        x_padding_judge = x_padding_judge * att_full
        ############################################
        att = att.masked_fill(x_padding_judge == 0, -1e9)
        ############  output 接口  ###################
        self.att_score = att.detach()

        att = F.softmax(att, dim=-1)

        att = att + self.adj_weight * att_fix
        # att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y



class Transformer_encoder_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn1 = CausalSelfAttention_masked_for_formula(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, x_padding_judge, adj, is_formula):
        x = x + self.attn1(self.ln1(x), x_padding_judge, adj, is_formula)
        x = x + self.mlp(self.ln2(x))
        return x




class Transformer_point_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge):
        for block in self.blocks:
            x = block(x, x_padding_judge, is_formula=False)
        return x







class Transformer_formula_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge, adj):
        for block in self.blocks:
            x = block(x, x_padding_judge, adj, is_formula=True)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model)  # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1)  # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *  # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)  # size = [batch, L, d_model]
        return x  # size = [batch, L, d_model]







# pointNet based on Convolution, T-NET naming is not accurate
class tNet(nn.Module):
    """
    The PointNet structure in the orginal PointNet paper:
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et. al. 2017
    """
    def __init__(self, config):
        super(tNet, self).__init__()

        self.activation_func = F.relu
        self.num_units = config.embeddingSize

        self.conv1 = nn.Conv1d(config.numberofVars+config.numberofYs, self.num_units, 1)
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        #self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(config.numberofVars+config.numberofYs)
        #self.input_layer_norm = nn.LayerNorm(config.numberofPoints)

        self.bn1 = nn.BatchNorm1d(self.num_units)
        self.bn2 = nn.BatchNorm1d(2 * self.num_units)
        self.bn3 = nn.BatchNorm1d(4 * self.num_units)
        self.bn4 = nn.BatchNorm1d(2 * self.num_units)
        self.bn5 = nn.BatchNorm1d(self.num_units)

    def forward(self, x):
        """
        :param x: [batch, #features, #points]
        :return:
            logit: [batch, embedding_size]
        """
        x = self.input_batch_norm(x)
        x = self.activation_func(self.bn1(self.conv1(x)))
        x = self.activation_func(self.bn2(self.conv2(x)))
        x = self.activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * self.num_units

        x = self.activation_func(self.bn4(self.fc1(x)))
        x = self.activation_func(self.bn5(self.fc2(x)))
        #x = self.fc2(x)

        return x




class points_emb(nn.Module):
    def __init__(self, config, in_channels=3):
        super(points_emb, self).__init__()
        emb_every = config.n_embd // in_channels
        emb_remainder = config.n_embd % in_channels
        self.fc1 = nn.Linear(1, emb_every)
        self.fc2 = nn.Linear(1, emb_every)
        self.fc3 = nn.Linear(1, emb_every + emb_remainder)



    def forward(self, xyz):
        xyz = xyz.transpose(dim0=1, dim1=2)
        out1 = self.fc1(xyz[:,:,0:1])
        out2 = self.fc2(xyz[:,:,1:2])
        out3 = self.fc3(xyz[:,:,2:3])
        points_emb = torch.cat([out1, out2, out3], dim=2)
        return points_emb



class seq_emb(nn.Module):
    def __init__(self, config):
        super(seq_emb, self).__init__()
        self.fc1 = nn.Linear(1, config.n_embd)

    def forward(self, seq):
        seq = seq.unsqueeze(dim=2)
        seq_emb = self.fc1(seq)
        return seq_emb



class project_mlp(nn.Module):
    def __init__(self, config):
        super(project_mlp, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.fc2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.act_fun = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.act_fun(x))
        return self.drop(x)



class project_mlp_formula(nn.Module):
    def __init__(self, config):
        super(project_mlp_formula, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.fc2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.act_fun = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.act_fun(x))
        return self.drop(x)


def swish(x, beta=1.0):
    return x * torch.sigmoid(beta * x)



class CRY_ENCODER(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        embeddingSize = config.n_embd


        self.block_size = config.block_size


        self.cgcnn_fc1 = nn.Linear(64*2+12, 64*2)
        self.cgcnn_act1 = nn.Sigmoid()
        self.cgcnn_act2 = nn.Softplus()
        self.cgcnn_bn11 = nn.BatchNorm1d(2*64)
        self.cgcnn_bn12 = nn.BatchNorm1d(64)

        self.cgcnn_fc2 = nn.Linear(64*2+12, 64*2)
        self.cgcnn_bn21 = nn.BatchNorm1d(2*64)
        self.cgcnn_bn22 = nn.BatchNorm1d(64)



        # self.ln_gru = nn.LayerNorm(embeddingSize)
        # self.dropkey = nn.Dropout(config.attn_pdrop)
        # self.gru = nn.GRU(bidirectional=True, hidden_size=embeddingSize, batch_first=True, input_size=embeddingSize)
        # self.att_score = nn.Linear(2 * embeddingSize, 1, bias=False)


        # self.soap_fc1 = nn.Linear(651525, 2 * embeddingSize)
        # self.soap_act = nn.GELU()
        # self.soap_fc2 = nn.Linear(2 * embeddingSize, embeddingSize)


        # input embedding stem
        self.element_embd = nn.Embedding(config.block_size, 64)# embeddingSize - 3*11*11
        self.convert_to_trans = nn.Linear(64, embeddingSize - 3*11*11)
        
        self.pos_emb = nn.Linear(embeddingSize, embeddingSize)

        self.points_emb = points_emb(config, in_channels=3)

        self.drop = nn.Dropout(config.embd_pdrop)

        # self.fc_project_formula = torch.nn.Linear(config.n_embd, config.n_embd)
        # self.fc_project_points = torch.nn.Linear(config.n_embd, config.n_embd)
        self.fc_project_formula = project_mlp_formula(config)
        self.fc_project_points = project_mlp(config)



        # transformer
        # self.blocks_unmask = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.Trans_element_encoder = Transformer_formula_encoder(config)
        # self.block = Block(config)
        # decoder head
        self.ln_before_aggregate = nn.LayerNorm(config.n_embd)
        self.ln_after_aggregate = nn.LayerNorm(2 * config.n_embd)

        ########### for gru aggregation  ####################
        # self.out1 = nn.Linear(2 * config.n_embd, 4 * config.n_embd)
        # self.out_act = nn.ReLU()
        # self.out2 = nn.Linear(4 * config.n_embd, 640)

        ########### for mean aggregation  ####################
        self.proj_energy = nn.Linear(config.n_embd, 640)

        # self.out1 = nn.Linear(640, 4 * 640)
        # self.out_act = nn.ReLU()
        # self.out2 = nn.Linear(4 * 640, 1)

        ########### e3nn aggregation  #######################
        irreps_in = "128x0e + 128x1o"
        irreps_out = "384x0e"
        self.linear_layer1 = e3nn.o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

        self.linear1_layer2 = e3nn.o3.Linear(irreps_in='128x0e', irreps_out='256x0e')
        self.non_linearity = e3nn.nn.Activation(irreps_in='256x0e', acts=[swish])
        self.linear2_layer2 = e3nn.o3.Linear(
            irreps_in='256x0e', irreps_out=e3nn.o3.Irreps("384x0e")
        )

        self.scalar = nn.Linear(1, 1)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))



    def forward(self, batch):
        

        idx = batch['nodes']
        adj = batch['distance']
        node_extra = batch['node_extra']
        
        formula_mask = idx == 0
        formula_mask = formula_mask.float()
        mask = 1 - formula_mask
        # zeros = torch.zeros(b, 1, device=formula_mask.device)
        # formula_mask = torch.cat([zeros, formula_mask], axis=1)
        # forward the GPT model
        nodes = self.element_embd(idx)  # each index maps to a (learnable) vector -> b x length x embedding
        # input_embedding = self.pos_emb(token_embeddings)  # [:, :t, :] # each position maps to a (learnable) vector
        b, t, e = nodes.size()
        
        nodes_i = nodes.unsqueeze(2).expand(b, t, t, e)
        nodes_j = nodes.unsqueeze(1).expand(b, t, t, e)

        nodes_res = nodes
        nodes = torch.cat([nodes_i, nodes_j, adj], dim=-1)
        nodes = self.cgcnn_fc1(nodes)
        nodes = self.cgcnn_bn11(nodes.view(-1, 2 * e)).view(b, t, t, 2 * e)
        nodes_filter, nodes_core = nodes.chunk(2, dim=-1)
        nodes_filter = self.cgcnn_act1(nodes_filter)
        nodes_core = self.cgcnn_act2(nodes_core)

        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)

        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)

        nodes = torch.sum(nodes_filter * nodes_core, dim=2)
        nodes = self.cgcnn_bn12(nodes.view(-1, e)).view(b, t, e)
        nodes = nodes_res + self.cgcnn_act2(nodes)





        nodes_i = nodes.unsqueeze(2).expand(b, t, t, e)
        nodes_j = nodes.unsqueeze(1).expand(b, t, t, e)

        nodes_res = nodes
        nodes = torch.cat([nodes_i, nodes_j, adj], dim=-1)
        nodes = self.cgcnn_fc2(nodes)
        nodes = self.cgcnn_bn21(nodes.view(-1, 2 * e)).view(b, t, t, 2 * e)
        nodes_filter, nodes_core = nodes.chunk(2, dim=-1)
        nodes_filter = self.cgcnn_act1(nodes_filter)
        nodes_core = self.cgcnn_act2(nodes_core)

        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)
        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)

        nodes = torch.sum(nodes_filter * nodes_core, dim=2)
        nodes = self.cgcnn_bn22(nodes.view(-1, e)).view(b, t, e)
        nodes = nodes_res + self.cgcnn_act2(nodes)

        nodes = nodes.masked_fill(mask.unsqueeze(-1).expand(b, t ,e) == 0, 0.0)

        x = self.convert_to_trans(nodes)
        # soap = self.soap_act(self.soap_fc1(soap))
        # soap = self.soap_fc2(soap)

        # x = x + self.pos_emb(node_extra)
        x = torch.cat([x, node_extra], axis=-1)
        x = self.pos_emb(x)
        x = self.Trans_element_encoder(x, formula_mask, adj)



        x = self.ln_before_aggregate(x)

        x = self.proj_energy(x)


        ############## gru aggregation  ###################
        # x, _ = self.gru(self.ln_gru(x))
        # att_score = self.att_score(x)
        # mask = 1.0 - formula_mask.float()
        # mask = self.dropkey(mask)
        # att_score = torch.where(mask==0.0, torch.ones_like(mask) * -1e10, att_score.squeeze(-1))
        # att_weights = torch.softmax(att_score, dim=1)
        # x = torch.sum(att_weights.unsqueeze(-1) * x, dim=1)


        ############## mean aggregation  ###################
        output_mask = formula_mask.unsqueeze(-1).repeat(1, 1, 640)
        x = x.masked_fill(output_mask == 1, 0.0)
        
        # hidden = x

        # x = x.sum(axis=1)


        # # x = self.ln_after_aggregate(x)
        # x = self.out_act(self.out1(x))
        # pred = self.out2(x).squeeze()


        ############ mace aggregation  ###################
        x1 = x[:, :, :512]
        x2 = x[:, :, 512:]
        es1 = self.linear_layer1(x1).squeeze()
        es2 = self.linear2_layer2(self.non_linearity(self.linear1_layer2(x2))).squeeze()

        es = es1 + es2
        #### 补一个用batch.batch做汇聚 ###############
        padding_mask = formula_mask.unsqueeze(-1)  # 扩展维度用于广播 [B, T, 1]
        es = es * (1 - padding_mask)  # 将padding位置置零
        es_sum = es.sum(dim=1)       # 沿序列维度求和 [B, D]

        return es_sum
