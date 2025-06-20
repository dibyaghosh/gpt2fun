# GPT Pretraining with JAX/Flax

A high-performance implementation for training GPT-style transformer language models from scratch using JAX/Flax with distributed computing support.

## Overview

This codebase provides a complete pipeline for pretraining large language models on text datasets. It features efficient distributed training, comprehensive logging, and checkpointing capabilities designed for research and production use.

## Architecture

### Core Files

- **`src/run.py`** - Main training script with distributed setup, training loop, and logging
- **`src/transformer.py`** - GPT transformer architecture implementation  
- **`src/llm.py`** - Language model wrapper with embeddings and output head
- **`src/dataset.py`** - Efficient data loading for large datasets with sharding support
- **`src/utils.py`** - Training utilities (optimizers, checkpointing, training state)
- **`src/download_fineweb100B.py`** - Script to download FineWeb-100B dataset

### Model Features

- **Standard GPT Architecture**: Multi-head attention, feedforward layers, layer normalization
- **Configurable Size**: Adjustable layers, heads, embedding dimensions
- **Research Hooks**: Custom perturbation points for research experiments
- **Mixed Precision**: bfloat16 training for memory efficiency

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For TPU support, ensure JAX TPU libraries are available
```

### Basic Training

```bash
python src/run.py \
  --model.n_layer=12 \
  --model.n_head=12 \
  --model.n_embd=768 \
  --batch_size=512 \
  --num_tokens=1000000000 \
  --save_dir=./checkpoints
```

### Configuration Options

#### Model Configuration
- `--model.n_layer`: Number of transformer layers (default: 12)
- `--model.n_head`: Number of attention heads (default: 12) 
- `--model.n_embd`: Embedding dimension (default: 768)
- `--model.vocab_size`: Vocabulary size (default: 50304)
- `--model.dropout`: Dropout rate (default: 0.0)

#### Training Configuration
- `--batch_size`: Global batch size (default: 512)
- `--per_device_batch_size`: Batch size per device (default: 16)
- `--seq_len`: Sequence length (default: 1024)
- `--num_tokens`: Total tokens to train on
- `--optimizer.lr`: Learning rate (default: 6e-4)
- `--optimizer.weight_decay`: Weight decay (default: 1e-1)

#### Data & Checkpointing
- `--data_dir`: Path to tokenized data directory
- `--save_dir`: Directory for saving checkpoints
- `--save_interval`: Steps between checkpoint saves
- `--from_pretrained`: Load from existing checkpoint

## Data Format

The system expects preprocessed tokenized data in binary format:
- Header: `[magic_number, version, num_tokens]` (3 x uint32)
- Tokens: Array of uint16 token IDs
- Magic number: `20240520` for format validation

## Distributed Training

The codebase automatically handles distributed training across multiple devices:

```bash
# Multi-GPU training
python src/run.py --batch_size=2048 --per_device_batch_size=16

# The system will automatically:
# - Distribute data across devices
# - Perform gradient accumulation as needed
# - Synchronize updates across processes
```

## Monitoring

Training metrics are logged to Weights & Biases, including:
- Loss and accuracy
- Learning rate schedules  
- Gradient and parameter norms
- Layer-wise activation statistics
- Training throughput

## Advanced Features

### Custom Perturbations
The transformer includes research-oriented perturbation hooks at key points:
- Input to each block
- After attention sublayer
- After MLP sublayer

### Gradient Accumulation
Automatic gradient accumulation when `batch_size > per_device_batch_size * num_devices`.

### Flexible Optimizers
Support for various optimization strategies:
- AdamW with weight decay
- Learning rate warmup and decay
- Gradient clipping
- Optional weight decay scheduling

## Dependencies

Key dependencies include:
- **JAX**: Numerical computing and automatic differentiation
- **Flax**: Neural network library built on JAX
- **Optax**: Gradient processing and optimization
- **Orbax**: Checkpointing and serialization
- **Weights & Biases**: Experiment tracking
- **Google Cloud Storage**: Data storage integration

## Hardware Support

Optimized for:
- **TPUs**: Google Cloud TPU v3/v4/v5
- **GPUs**: Multi-GPU setups via JAX distributed training
- **CPUs**: Development and small-scale experiments

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]