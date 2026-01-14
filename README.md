# 2D Materials Work Function Prediction

Deep learning model for predicting work functions of 2D crystalline materials using attention-based neural networks with interpretability analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a CLIP-inspired contrastive learning framework to predict the work function of 2D materials from their crystal structures. The model combines a crystal structure encoder with attention mechanisms to learn meaningful representations of atomic configurations and their relationship to electronic properties.

**Key Features:**
- **Attention-based architecture**: Multi-head self-attention mechanism captures atomic interactions
- **Interpretability**: Extract and visualize which atoms the model considers important
- **Comprehensive analysis**: Statistical validation of learned patterns (surface vs. bulk, electronegativity effects)
- **Trained model included**: Pre-trained checkpoint on 100+ 2D materials

**Scientific Contribution:**
- Novel application of contrastive learning to materials property prediction
- Systematic interpretability analysis revealing model decision-making process
- Statistical evidence that model learns chemically meaningful patterns

## Architecture

The model consists of three main components:

1. **Crystal Encoder** (`CRY_ENCODER`): Processes atomic structures using graph convolutions and transformer blocks with attention mechanisms
2. **Work Function Encoder**: ResNet-based encoder for work function profiles
3. **Contrastive Learning**: CLIP-style framework matching crystal structures to their work function profiles

**Key Technical Details:**
- Embedding dimension: 384
- Transformer layers: 2
- Attention heads: 4
- Training: Contrastive loss with cosine similarity
- Attention mechanism: Saves attention scores for interpretability analysis

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd 2d_work_func

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### Training

```bash
# Train the model from scratch
python src/train.py
```

The training script will:
- Load 2D material structures from the database
- Split data into train/validation/test sets (80/10/10)
- Train for 1000 epochs with cosine learning rate scheduling
- Save the best model to `checkpoints/best_contra.pt`

### Inference

```python
import torch
from src.models.clip_model import CLIP, CLIPConfig, PointNetConfig
from src.models.crystal_encoder import CRY_ENCODER, cry_config

# Load trained model
model = CLIP(config, pointnet_config, cry_encoder)
checkpoint = torch.load('checkpoints/best_contra.pt')
model.load_state_dict(checkpoint)
model.eval()

# Make predictions
# ... (see notebooks/test.ipynb for full example)
```

## Dataset

The dataset contains 100+ 2D crystalline materials from the Materials Project database (MIP2D). Each material includes:

- Crystal structure (atomic positions, lattice parameters)
- Work function profiles (Z-axis potential)
- Material properties (composition, space group)

**Data Format:**
- Raw structures: POSCAR format
- Work functions: Text files with Z-coordinate and potential values
- Database: LMDB format for efficient loading

See [data/README.md](data/README.md) for detailed dataset documentation.

## Training

### Configuration

Key hyperparameters (in `src/train.py`):

```python
embeddingSize = 384      # Hidden dimension
n_layers = 2             # Transformer layers
n_heads = 4              # Attention heads
batchSize = 64           # Batch size
numEpochs = 1000         # Training epochs
learning_rate = 1e-4     # Initial learning rate
```

### Training Process

The model is trained using:
- **Optimizer**: AdamW with weight decay
- **Learning rate schedule**: Cosine annealing
- **Loss**: Contrastive loss (cross-entropy on similarity matrix)
- **Validation**: Monitor validation loss, save best checkpoint

## Interpretability Analysis

### Extracting Attention Scores

```bash
# Extract attention for a specific material (e.g., MoS2)
python scripts/interpretability/extract_attention.py --formula MoS2
```

This generates:
- Attention heatmaps showing which atoms receive high attention
- Gradient-based saliency maps
- Ranking of top important atoms

**Example outputs** are available in [examples/interpretability/](examples/interpretability/).

### Statistical Analysis

Comprehensive statistical analysis of attention patterns:

```bash
# Analyze surface vs. bulk attention patterns
python scripts/interpretability/analyze_surface.py \
    --attn_dir interpretability/all_attention_results \
    --db data/processed/structures.db \
    --out interpretability/surface_attention_report
```

**Key Findings:**
- Halogen elements (F, Cl, Br, I) show 92.2% surface attention concentration
- Model learns chemically meaningful patterns without explicit supervision
- Attention correlates with electronegativity for surface atoms

See [docs/statistical_analysis_report.md](docs/statistical_analysis_report.md) for detailed analysis.

## Project Structure

```
2d_work_func/
├── src/                    # Core source code
│   ├── models/            # Neural network models
│   │   ├── clip_model.py         # CLIP architecture
│   │   └── crystal_encoder.py    # Crystal structure encoder
│   ├── data/              # Data loading utilities
│   │   └── dataset.py            # Dataset and collator
│   └── train.py           # Training script
├── scripts/               # Analysis and utility scripts
│   ├── interpretability/  # Attention analysis scripts
│   └── statistical_analysis/  # Statistical analysis scripts
├── notebooks/             # Jupyter notebooks
│   ├── train.ipynb               # Training notebook
│   ├── test.ipynb                # Testing notebook
│   └── prepare_dataset.ipynb     # Dataset preparation
├── data/                  # Dataset directory
│   ├── raw/              # Raw material structures
│   └── processed/        # Processed database files
├── checkpoints/           # Trained model checkpoints
├── examples/              # Example outputs
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
└── README.md             # This file
```

## Citation

If you use this code or model in your research, please cite:

```bibtex
@software{2d_work_function_prediction,
  title={Deep Learning for 2D Materials Work Function Prediction},
  author={Research Team},
  year={2025},
  url={https://github.com/your-repo/2d_work_func}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Materials Project for providing the 2D materials database (MIP2D)
- PyTorch and e3nn teams for excellent deep learning frameworks
- ASE (Atomic Simulation Environment) for materials structure handling

## Documentation

- [Methodology](docs/methodology.md) - Detailed research methodology
- [Attention Extraction Guide](docs/attention_extraction_guide.md) - Tutorial on extracting attention scores
- [Statistical Analysis Report](docs/statistical_analysis_report.md) - Comprehensive analysis of attention patterns
- [Dataset Documentation](data/README.md) - Dataset structure and format

## Contact

For questions or issues, please open an issue on GitHub or contact the research team.
