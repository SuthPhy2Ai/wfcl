# 方法论

**概述**

我们的方法论以一个多模态对比学习框架为核心，旨在预测二维材料的功函数。其核心原则是学习一个共享的嵌入空间，在该空间中，材料的晶体结构表示与其对应的电子波函数表示能够对齐。为实现这一目标，我们训练了两个独立的编码器——一个用于晶体图，另一个用于波函数序列。通过优化这两个编码器的输出，使得匹配的（晶体-波函数）对之间的相似性最大化，而非匹配对的相似性最小化。此架构的灵感源于 CLIP 框架，并针对材料科学应用进行了适配。

**晶体结构编码器**

晶体结构编码器 $f_{\text{cry}}(\cdot)$ 是一个深度图神经网络，它将一个晶体图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 映射到一个固定维度的向量表示 $\mathbf{z}^{\text{cry}} \in \mathbb{R}^{D}$。每个原子 $i \in \mathcal{V}$ 的输入特征包含其原子序数 $a_i$，每条边 $(i,j) \in \mathcal{E}$ 的输入是原子间距离 $d_{ij}$。

**输入表示**

对于一个包含 $T$ 个原子的晶体，其初始节点特征由一个嵌入层生成：$\mathbf{h}_i^{(0)} = E_{\text{elem}}[a_i]$，其中 $E_{\text{elem}} \in \mathbb{R}^{V \times D_{\text{elem}}}$ 是一个针对 $V$ 种不同元素的可学习嵌入矩阵。原子间距离则通过高斯基函数进行扩展，以捕捉局部环境信息：

$$ \mathbf{e}_{ij} = \exp\left(-\frac{(\mathbf{d}_{ij} - \boldsymbol{\mu})^2}{\boldsymbol{\gamma}^2}\right) $$

其中，$\boldsymbol{\mu}$ 和 $\boldsymbol{\gamma}$ 是固定参数，定义了在 8.0 Å 截断距离内的 12 个高斯滤波器的中心和宽度。此外，我们为每个原子计算一个局部几何特征向量 $\mathbf{x}_i^{\text{extra}} \in \mathbb{R}^{363}$，该向量编码了原子 $i$、其 11 个最近邻原子 $j$ 以及这些邻居各自的 11 个最近邻原子 $k$ 之间的距离和角度信息。

**门控图卷积层**

我们采用两个晶体图卷积网络（CGCNN）风格的层来更新节点表示。原子 $i$ 在第 $l$ 层的更新规则如下：

$$ \mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{Softplus}\left( \sum_{j \in \mathcal{N}(i)} \sigma(\mathbf{z}_{ij}^{(l)}) \odot g(\mathbf{z}_{ij}^{(l)}) \right) $$

其中 $\mathcal{N}(i)$ 是原子 $i$ 的邻居集合，$\mathbf{z}_{ij}^{(l)} = W_f^{(l)}[\mathbf{h}_i^{(l)} \| \mathbf{h}_j^{(l)} \| \mathbf{e}_{ij}] + \mathbf{b}_f^{(l)}$，$g(\cdot)$ 是一个激活函数（Softplus）。特征向量被分为两半，其中一半经过 Sigmoid 门控函数 $\sigma(\cdot)$ 处理，用于调节另一半。$W_f$ 和 $\mathbf{b}_f$ 是可学习的参数。

**带邻接偏置的 Transformer**

节点的特征随后与局部几何特征 $\mathbf{x}_i^{\text{extra}}$ 融合，并输入到一个多头自注意力 Transformer 中。原子 $i$ 和 $j$ 之间的注意力分数通过一个可学习的邻接偏置进行修正：

$$ \alpha_{ij} = \frac{\exp(\frac{(\mathbf{q}_i^T \mathbf{k}_j)}{\sqrt{d_k}} + B_{ij})}{\sum_{k=1}^T \exp(\frac{(\mathbf{q}_i^T \mathbf{k}_k)}{\sqrt{d_k}} + B_{ik})} $$

其中 $\mathbf{q}_i$ 和 $\mathbf{k}_j$ 分别是查询和键向量，偏置项 $B_{ij} = \phi(\mathbf{e}_{ij})$ 由一个小型多层感知机 $\phi$ 从高斯距离特征中计算得出。

**等变聚合**

Transformer 输出的最终节点表示被投影到一个 640 维空间，然后使用 `e3nn` 库中的 $O(3)$ 等变层聚合成一个单一的图级别嵌入。这些特征被拆分并通过并行的等变线性层进行处理，这些层作用于 `0e`（标量）和 `1o`（伪向量）类型的不可约表示（irreps），从而确保旋转不变性。最终得到的原子级特征进行求和，生成最终的晶体嵌入 $\mathbf{z}^{\text{cry}}$。

**波函数编码器**

波函数编码器 $f_{\text{wf}}(\cdot)$ 是一个一维残差网络（ResNet），它处理电子势序列 $\mathbf{w} \in \mathbb{R}^{L}$ 以生成一个嵌入向量 $\mathbf{z}^{\text{wf}} \in \mathbb{R}^{D}$。该架构由一个卷积主干和四个残差阶段组成。每个残差块定义为：

$$ \mathbf{w}^{(l+1)} = \text{ReLU}(\mathcal{F}(\mathbf{w}^{(l)}, \{W_i^{(l)}\}) + \mathbf{w}^{(l)}) $$

其中 $\mathcal{F}$ 代表两个带批量归一化（BatchNorm）和 ReLU 的一维卷积。通道深度逐级增加（64, 128, 256, 512），而空间维度通过阶段之间的步进卷积（strided convolutions）进行缩减。一个最终的全局平均池化层和一个线性投影层生成嵌入向量 $\mathbf{z}^{\text{wf}}$。

**对比学习目标**

给定一个批次（batch）的 $N$ 个（晶体，波函数）对，我们计算它们的嵌入向量 $\{\mathbf{z}_i^{\text{cry}}, \mathbf{z}_i^{\text{wf}}\}_{i=1}^N$。这些嵌入向量经过线性投影头和 L2 归一化，得到 $\{\hat{\mathbf{p}}_i^{\text{cry}}, \hat{\mathbf{p}}_i^{\text{wf}}\}_{i=1}^N$。第 $i$ 个波函数与第 $j$ 个晶体之间的相似度是其余弦相似度 $s_{ij} = (\hat{\mathbf{p}}_i^{\text{wf}})^T \hat{\mathbf{p}}_j^{\text{cry}}$。

模型通过最小化一个基于相似度分数的对称交叉熵损失进行训练：

$$ \mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij} / \tau)} + \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ji} / \tau)} \right] $$

其中 $\tau$ 是一个可学习的温度参数，用于缩放 logits。

**训练细节**

该模型在一个包含 1,899 种二维材料的数据集上进行训练，数据集按 80% 训练、10% 验证和 10% 测试的比例划分。我们使用 AdamW 优化器，学习率为 $1 \times 10^{-4}$，并采用余弦退火学习率调度策略。嵌入维度 $D$ 设置为 384，批大小为 64。训练共进行 1000 个周期（epochs），并保存验证损失最低的模型检查点用于后续推理。

