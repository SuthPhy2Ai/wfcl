"""简单推断测试脚本"""
import os
import sys
import glob

# 检查文件
print('=== 检查文件 ===')
pts = glob.glob('**/*.pt', recursive=True)
print('Checkpoint files:', pts[:5] if pts else 'None')
dbs = glob.glob('**/*.db', recursive=True)
print('Database files:', dbs[:5] if dbs else 'None')

# 导入模块
print('\n=== 导入模块 ===')
from model import CLIP, CLIPConfig, PointNetConfig
from cry_encoder import cry_config, CRY_ENCODER
from utils import CharDataset, MyCollator
print('All modules imported successfully!')

# 创建模型
print('\n=== 创建模型 ===')
import torch

embeddingSize = 384
n_layers = 2
n_heads = 4
numPoints = 9
blockSize = 95

cry_conf = cry_config(blockSize, n_layer=n_layers, n_head=n_heads, n_embd=embeddingSize)
cry_encoder = CRY_ENCODER(cry_conf)

pconf = PointNetConfig(embeddingSize=embeddingSize, numberofPoints=numPoints)
mconf = CLIPConfig(blockSize, n_layer=n_layers, n_head=n_heads, n_embd=embeddingSize)

model = CLIP(mconf, pconf, cry_encoder)
print(f'Model created! Parameters: {sum(p.numel() for p in model.parameters()):,}')

# 尝试加载checkpoint
print('\n=== 加载Checkpoint ===')
ckpt_files = glob.glob('**/*.pt', recursive=True)
if ckpt_files:
    ckpt_path = ckpt_files[0]
    print(f'Loading: {ckpt_path}')
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint)
        print('Checkpoint loaded successfully!')
    except Exception as e:
        print(f'Failed to load checkpoint: {e}')
else:
    print('No checkpoint found, using random weights')

# 测试前向传播
print('\n=== 测试前向传播 ===')
import numpy as np

batch_size = 2
max_len = 10

dummy_batch = {
    'nodes': torch.randint(1, 50, (batch_size, max_len)),
    'wf': torch.randn(batch_size, numPoints),
    'distance': torch.randn(batch_size, max_len, max_len, 12),
    'node_extra': torch.randn(batch_size, max_len, 3*11*11),
}

model.eval()
with torch.no_grad():
    loss, logits = model(dummy_batch)
    print(f'Loss: {loss.item():.4f}')
    print(f'Logits shape: {logits.shape}')
    print('Forward pass successful!')

print('\n=== 测试完成 ===')
