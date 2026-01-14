import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

import e3nn.o3
import e3nn.nn


import glob
import os
import numpy as np
import copy as cpy
import torch
import random
import pandas as pd
import multiprocessing as mp
from tqdm import  tqdm
from torch.utils.data import DataLoader, random_split, Subset
# from torch.utils.data import DataLoader
from ase.db import connect
from src.data.dataset import CharDataset, MyCollator
from src.models.clip_model import CLIP, CLIPConfig, PointNetConfig
from src.models.crystal_encoder import cry_config, CRY_ENCODER

from pathlib import Path
from typing import Union

import math
import time
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import AdamW
from transformers import (
    # AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)




# set the random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# config
device='cuda:1'
numEpochs = 1000 # number of epochs to train the GPT+PT model
embeddingSize = 384 # the hidden dimension of the representation of both GPT and PT
batchSize = 64 # batch size of training data
decimals = 4 # decimals
n_layers = 2
n_heads = 4
numPoints = 9
blockSize = 95
num_workers = 2
dataInfo = 'Layers_{}Heads_{}EmbeddingSize{}'.format(n_layers, n_heads, embeddingSize)
addr = './checkpoints/' # where to save model
bestLoss = None # if there is any model to load as pre-trained one
fName = '{}_phonon_moco.txt'.format(dataInfo)
ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])




# filtered_ids = pd.read_csv('./filtered_idx.csv')['ids'].tolist()

# db = connect('/home/wy/bader/MoreH_924.db')
db = connect('./data/processed/structures.db')


rows = []
for row in db.select():
    rows.append(row)

dataset = CharDataset(rows)


total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = int(total_size * 0.1)
# 确保训练集、测试集和验证集的总和等于数据集的总大小
validation_size = total_size - train_size - test_size

indices = torch.randperm(total_size).tolist()
# indices = [i for i in range(total_size)]

train_indices, val_indices, test_indices = indices[: train_size], indices[train_size: train_size + validation_size], indices[train_size + validation_size :]
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)


mconf = cry_config(blockSize,
                  n_layer=n_layers, n_head=n_heads, n_embd=embeddingSize)
cry_encoder = CRY_ENCODER(mconf)



pconf = PointNetConfig(embeddingSize=embeddingSize,
                       numberofPoints=numPoints)
mconf = CLIPConfig(blockSize,
                  n_layer=n_layers, n_head=n_heads, n_embd=embeddingSize)
model = CLIP(mconf, pconf, cry_encoder)
train_loader = DataLoader(train_dataset,
                    batch_size=batchSize,
                    num_workers=8,
                    collate_fn=MyCollator(mysql_url=db),
                    shuffle=True,
                    drop_last=False)
val_loader = DataLoader(val_dataset,
                    batch_size=2*batchSize,
                    num_workers=4,
                    collate_fn=MyCollator(mysql_url=db),
                    shuffle=True,
                    drop_last=False)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9,0.95))


num_train_epochs = numEpochs
num_warmup_epochs = 0
output_dir = './checkpoints'
model.train()

# 假设 train_loader 和 val_loader 均已定义
num_update_steps_per_epoch = math.ceil(len(train_loader) / 1)
max_train_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps,
)
start_epoch = 0
completed_steps = 0

print_loss = []
completed_steps = 0
best_loss = float('inf')

for epoch in range(start_epoch, num_train_epochs):
    model.train()
    train_loss_sum = 0.0
    t0 = time.time()

    pbar = tqdm(train_loader, total=len(train_loader))
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, logits = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss_sum += loss.item()
        print_loss.append(loss.item())
        lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Epoch {epoch+1} Step {step}: loss {loss.item():.4f}, lr {lr:.2e}")

        completed_steps += 1

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model(batch)
            val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}  time: {time.time()-t0:.1f}s")

    # 保存最优
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        print("Saving best model…")
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "best_contra.pt"))