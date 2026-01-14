
import math
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
# from fastkan import FastKANLayer
# from model_cegann import CEGAN
# from model_cgcnn import CrystalGraphConvNet

logger = logging.getLogger(__name__)


class CLIPConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)



class PointNetConfig:
    """ base PointNet config """

    def __init__(self, embeddingSize, numberofPoints,
                 **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points


        for k, v in kwargs.items():
            setattr(self, k, v)








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



# class seq_emb(nn.Module):
#     def __init__(self, config):
#         super(seq_emb, self).__init__()
#         self.fc1 = nn.Conv1d(1, 8, kernel_size=3, stride=1)
#         self.bn1 = nn.BatchNorm1d(8)
#         self.fc2 = nn.Conv1d(8, 64, kernel_size=3, stride=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Conv1d(64, config.n_embd, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm1d(config.n_embd)
#         # self.fc4 = nn.Conv1d(128, config.n_embd, kernel_size=3, stride=1)
#         # self.bn4 = nn.BatchNorm1d(config.n_embd)
#         self.fc5 = nn.Linear(config.n_embd, config.n_embd)
#         self.fc_res1 = nn.Conv1d(1, 64, kernel_size=5)
#         self.bn_res1 = nn.BatchNorm1d(64)
#         self.fc_res2 = nn.Conv1d(64, config.n_embd, kernel_size=5)
#         self.bn_res2 = nn.BatchNorm1d(config.n_embd)
#         self.max_pool = nn.MaxPool1d(3)
#         self.act_fun = nn.ReLU()
#         self.act_tanh = nn.GELU()
#         self.drop = nn.Dropout(config.resid_pdrop)
#     def forward(self, seq):
#         seq = seq.unsqueeze(dim=1)
#         seq_res = seq
#         seq = self.act_fun(self.bn1(self.fc1(seq)))
#         seq = self.act_fun(self.bn2(self.fc2(seq)))
#         seq = self.act_fun(self.bn_res1(self.fc_res1(seq_res)) + seq)
#         seq_res = seq
#         seq = self.act_fun(self.bn3(self.fc3(seq)))
#         # seq = self.act_fun(self.bn4(self.fc4(seq)))
#         seq = self.act_fun(self.bn_res2(self.fc_res2(seq_res)) + seq)
#         seq = self.max_pool(seq)
#         seq = seq.squeeze()
#         seq = self.drop(seq)
#         seq = self.act_tanh(self.fc5(seq))

#         return seq



class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class seq_emb(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=128, num_blocks=[2, 2, 2, 2]):
        super(seq_emb, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks[0])
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, embedding_dim)
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * BasicBlock1D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )
        layers = []
        layers.append(BasicBlock1D(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# class CoPE(nn.Module):
#     def __init__(self, npos_max, head_dim):
#         super().__init__()
#         self.npos_max = npos_max
#         self.pos_emb = nn.Parameter(
#             torch.zeros(1, head_dim, npos_max)
#         )

#     def forward(self, query, attn_logits):
#         # compute positions
#         gates = torch.sigmoid(attn_logits)
#         pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
#         pos = pos.clamp(max=self.npos_max - 1)
#         # interpolate from integer positions
#         pos_ceil = pos.ceil().long()
#         pos_floor = pos.floor().long()
#         logits_int = torch.matmul(query, self.pos_emb)
#         logits_ceil = logits_int.gather(-1, pos_ceil)
#         logits_floor = logits_int.gather(-1, pos_floor)
#         w = pos - pos_floor  # Interpolation factor
#         return logits_ceil * w + logits_floor * (1 - w)


# def random_mask(rng, x, mask_ratio, bias=None):
#     """
#     x: [N, L, L, C] input

#     """
#     N, L1, L2, C = x.shape
#     x = x.view(-1, L2, C)
#     N, L, _ = x.shape  # batch, length, dim
#     len_keep = int(L * (1 - mask_ratio))

#     # Generating uniform noise
#     noise = torch.rand(N, L, generator=rng)

#     # Apply bias if provided
#     if bias is not None:
#         noise += bias

#     # Calculate indices to keep
#     ids_shuffle = torch.argsort(noise, dim=1).to(x.device)
#     ids_restore = torch.argsort(ids_shuffle, dim=1).to(x.device)

#     # Mask creation
#     mask = torch.ones(N, L, dtype=torch.bool, device=x.device)
#     mask.scatter_(1, ids_shuffle[:, :len_keep], False)

#     # Applying mask
#     x_masked = x.clone()  # Creating a copy to preserve original data
#     x_masked[mask] = 0  # Applying mask

#     x_masked = x_masked.view(-1, L1, L2, C).to(x.device)
    
#     return x_masked, mask, ids_restore



class CLIP(nn.Module):


    def __init__(self, config, pointNetConfig=None, cry_encoder=None):
        super().__init__()

        self.config = config
        self.pointNetConfig = pointNetConfig
        self.pointNet = None

        self.cry_encoder = cry_encoder
        embeddingSize = config.n_embd
        self.block_size = config.block_size

        # self.input_deal = nn.Linear(360, embeddingSize)
        # self.fc_input = nn.Linear(embeddingSize, embeddingSize)
        # self.ln_gru1 = nn.LayerNorm(config.n_embd)
        # self.dropkey1 = nn.Dropout(config.attn_pdrop)
        # self.gru1 = nn.GRU(bidirectional=True, hidden_size=embeddingSize, batch_first=True, input_size=embeddingSize)
        # self.att_score = nn.Linear(2 * embeddingSize, 1, bias=False)

        # self.ln_gru2 = nn.LayerNorm(2 * config.n_embd)
        # self.dropkey1 = nn.Dropout(config.attn_pdrop)
        # self.gru2 = nn.GRU(bidirectional=True, hidden_size=embeddingSize, batch_first=True, input_size=2 * embeddingSize)
        # self.att_score2 = nn.Linear(2 * embeddingSize, 1, bias=False)
        # self.key = nn.Linear(2 * embeddingSize, embeddingSize)
        # self.query = nn.Linear(2 * embeddingSize, embeddingSize)
        # self.cope = CoPE(config.block_size, config.n_embd)



        self.pos_emb_wf = PositionalEncoding(embeddingSize, dropout=0.1, max_len=self.pointNetConfig.numberofPoints)

        self.penalty_labels = torch.eye(config.block_size)

        self.wf_emb = seq_emb(embedding_dim=config.n_embd)

        # self.drop = nn.Dropout(config.embd_pdrop)

        self.mlp_predict = nn.Sequential(
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(config.n_embd, 64),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(64, pointNetConfig.numberofPoints),
        )

        self.fc_project_formula = nn.Linear(config.n_embd, config.n_embd)
        # self.fc_project_formula = nn.Linear(config.n_embd, config.n_embd)
        self.fc_project_wf = nn.Linear(config.n_embd, config.n_embd)

        # self.fc_project_formula = FastKANLayer(512, config.n_embd)
        # self.fc_project_formula_cgcnn = FastKANLayer(128, config.n_embd)
        # self.fc_project_points = FastKANLayer(config.n_embd, config.n_embd)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))# 0.07
        self.register_buffer('total_labels', torch.arange(30000))
        
        self.rng = torch.Generator()
        self.rng.manual_seed(42)  # 使用常见的种子数字42


        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.ln_f = nn.LayerNorm(config.n_embd)
        self.ln_wf = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.startswith('gru1.'):
                    decay.add(fpn)
                elif pn.startswith('gru2.'):
                    decay.add(fpn)
                elif pn.endswith('grid'):
                    no_decay.add(fpn)
                elif mn.endswith('cope'):
                    no_decay.add(fpn)

        no_decay.add('logit_scale')


        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)


        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, batch):


        wf = batch['wf']
        b, _ = batch['nodes'].size()

        x = self.cry_encoder(batch)
        formula_embedding_final = self.fc_project_formula(x)


        formula_embedding_final = formula_embedding_final / formula_embedding_final.norm(dim=-1, keepdim=True)


        wf_embeddings_final = self.wf_emb(wf)
        wf_embeddings_final = self.ln_wf(wf_embeddings_final)

        wf_embeddings_final = self.fc_project_wf(wf_embeddings_final)

        wf_embeddings_final = wf_embeddings_final / wf_embeddings_final.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_wf = logit_scale * wf_embeddings_final @ formula_embedding_final.t()
        logits_per_formula = logits_per_wf.t()

        labels = self.total_labels[:b]


        if wf.shape[0] == b:
            loss = (F.cross_entropy(logits_per_wf, labels) +
                    F.cross_entropy(logits_per_formula, labels)) / 2
        else:
            loss = 0.0

        return loss, logits_per_wf


