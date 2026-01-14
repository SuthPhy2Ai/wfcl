# Method



Our methodology is centered on a multi-modal contrastive learning framework designed to predict the work function of 2D materials. The core principle is to learn a shared embedding space where the representations of a material's crystal structure and its corresponding work function profile are aligned. This is achieved by training two independent encoders—one for the crystal graph and one for the work function profile sequence—whose outputs are optimized to maximize the similarity between corresponding pairs while minimizing it for non-corresponding pairs. The architecture is inspired by the CLIP framework, adapted for materials science applications.

## Crystal Structure Encoder

The crystal structure encoder, $f_{\text{cry}}(\cdot)$, is a deep graph neural network that maps a crystal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ to a fixed-dimensional vector representation $\mathbf{z}^{\text{cry}} \in \mathbb{R}^{D}$. The input features for each atom $i \in \mathcal{V}$ consist of its atomic number $a_i$, and the input for each edge $(i,j) \in \mathcal{E}$ is the interatomic distance $d_{ij}$.


For a crystal with $T$ atoms, the initial node features are derived from an embedding layer: $\mathbf{h}_i^{(0)} = E_{\text{elem}}[a_i]$, where $E_{\text{elem}} \in \mathbb{R}^{V \times D_{\text{elem}}}$ is a learnable embedding matrix for $V$ unique elements. Interatomic distances are expanded using a Gaussian basis function to capture local environment information:

$$ \mathbf{e}_{ij} = \exp\left(-\frac{(\mathbf{d}_{ij} - \boldsymbol{\mu})^2}{\boldsymbol{\gamma}^2}\right) $$

where $\boldsymbol{\mu}$ and $\boldsymbol{\gamma}$ are fixed parameters defining the centers and widths of 12 Gaussian filters over a cutoff distance of 8.0 Å. Additionally, a local geometric feature vector $\mathbf{x}_i^{\text{extra}} \in \mathbb{R}^{363}$ is computed for each atom, encoding the distances and angles between atom $i$, its 11 nearest neighbors $j$, and their respective 11 nearest neighbors $k$.


Two CGCNN-inspired layers update the node representations. The update for atom $i$ at layer $l$ is given by:

$$ \mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{Softplus}\left( \sum_{j \in \mathcal{N}(i)} \sigma(\mathbf{z}_{ij}^{(l)}) \odot g(\mathbf{z}_{ij}^{(l)}) \right) $$

where $\mathcal{N}(i)$ is the set of neighbors of atom $i$, $\mathbf{z}_{ij}^{(l)} = W_f^{(l)}[\mathbf{h}_i^{(l)} \| \mathbf{h}_j^{(l)} \| \mathbf{e}_{ij}] + \mathbf{b}_f^{(l)}$, and $g(\cdot)$ is an activation function (Softplus). The feature vector is split into two halves, with one processed by a sigmoid gate $\sigma(\cdot)$ that modulates the other half. $W_f$ and $\mathbf{b}_f$ are learnable parameters.



The node features are then fused with the local geometric features $\mathbf{x}_i^{\text{extra}}$ and processed by a multi-head self-attention Transformer. The attention score between atoms $i$ and $j$ is modified with a learnable adjacency bias:

$$ \alpha_{ij} = \frac{\exp(\frac{(\mathbf{q}_i^T \mathbf{k}_j)}{\sqrt{d_k}} + B_{ij})}{\sum_{k=1}^T \exp(\frac{(\mathbf{q}_i^T \mathbf{k}_k)}{\sqrt{d_k}} + B_{ik})} $$

where $\mathbf{q}_i$ and $\mathbf{k}_j$ are the query and key vectors, and the bias $B_{ij} = \phi(\mathbf{e}_{ij})$ is computed by a small MLP $\phi$ from the Gaussian distance features.


The final node representations from the Transformer are projected to a 640-dimensional space and then aggregated into a single graph-level embedding using $O(3)$-equivariant layers from the `e3nn` library. The features are split and processed through parallel equivariant linear layers operating on irreducible representations (irreps) of types `0e` (scalar) and `1o` (pseudo-vector), ensuring rotational invariance. The resulting atomic features are summed to produce the final crystal embedding $\mathbf{z}^{\text{cry}}$.

## Work Function Profile Encoder

The work function profile encoder, $f_{\text{prof}}(\cdot)$, is a 1D Residual Network (ResNet) that processes the work function profile sequence $\mathbf{w} \in \mathbb{R}^{L}$ to produce an embedding $\mathbf{z}^{\text{prof}} \in \mathbb{R}^{D}$. The architecture consists of a convolutional stem followed by four residual stages. Each residual block is defined as:

$$ \mathbf{w}^{(l+1)} = \text{ReLU}(\mathcal{F}(\mathbf{w}^{(l)}, \{W_i^{(l)}\}) + \mathbf{w}^{(l)}) $$

where $\mathcal{F}$ represents two 1D convolutions with BatchNorm and ReLU. The channel depth is progressively increased (64, 128, 256, 512) and spatial dimensions are reduced via strided convolutions between stages. A final global average pooling layer and a linear projection produce the embedding $\mathbf{z}^{\text{prof}}$.

## Contrastive Learning 

Given a batch of $N$ (crystal, work function profile) pairs, we compute their embeddings $\{\mathbf{z}_i^{\text{cry}}, \mathbf{z}_i^{\text{prof}}\}_{i=1}^N$. These are passed through linear projection heads and L2-normalized to obtain $\{\hat{\mathbf{p}}_i^{\text{cry}}, \hat{\mathbf{p}}_i^{\text{prof}}\}_{i=1}^N$. The similarity between the $i$-th work function profile and the $j$-th crystal is the cosine similarity $s_{ij} = (\hat{\mathbf{p}}_i^{\text{prof}})^T \hat{\mathbf{p}}_j^{\text{cry}}$.

The model is trained to minimize a symmetric cross-entropy loss over the similarity scores:

$$ \mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij} / \tau)} + \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ji} / \tau)} \right] $$

where $\tau$ is a learnable temperature parameter that scales the logits.

## Training 

The model was trained on a dataset of 1,899 2D materials, split into 80% for training, 10% for validation, and 10% for testing. We used the AdamW optimizer with a learning rate of $1 \times 10^{-4}$ and a cosine annealing schedule. The embedding dimension $D$ was set to 384, and the batch size was 64. Training was conducted for 1000 epochs, with the model checkpoint corresponding to the lowest validation loss being saved for inference.

