# LLF-Training

Parametric baseline models for [Lead-Lag Forecasting on Social Platforms](https://github.com/kimzemian/LLF). This is a submodule of the main [LLF](https://github.com/kimzemian/LLF) repository.

Datasets are available at [lead-lag-forecasting.github.io](https://lead-lag-forecasting.github.io/).

## Models

Three architectures are implemented in `baseline_models.py`:

| Model | Description |
|---|---|
| **MLP** | Multi-layer perceptron with Tanh activations and Kaiming initialization |
| **LSTM** | LSTM encoder using the final hidden state for prediction |
| **Transformer** | Transformer encoder with learnable positional embeddings and average pooling |

All models take concatenated multi-channel time series as input and predict a scalar target.

## Datasets

Defined in `baseline_datasets.py`:

| Class | Channels | Use |
|---|---|---|
| `ArxivDataset` | Citations + Accesses | Standard ArXiv benchmarking |
| `ArxivDatasetAblation` | Single channel (configurable) | Ablation studies |
| `GitHubDataset` | Forks + Stars + Pushes | GitHub benchmarking |

All inputs are log-transformed (`log1p`) and standardized with a global mean/std scaler.

## Usage

### Training

```bash
python main_bench.py \
  --model_name transformer \
  --data_root /path/to/arxiv/data \
  --input_horizon 365 \
  --hidden_size 256 \
  --num_layers 3 \
  --lr 5e-4 \
  --epochs 5 \
  --batch_size 128
```

For ablation (single-channel input):

```bash
python main_bench.py --ablation --ablation_name accesses
```

### Hyperparameter Sweep

```bash
bash sweep.sh
```

Runs a grid search over learning rates, weight decays, optimizers, warmup schedules, and LR schedulers. Results are logged to [Weights & Biases](https://wandb.ai/).

## Metrics

Evaluated on validation and test splits:

- **log MAE** — Mean absolute error on log-transformed predictions
- **absolute MAE** — MAE on original scale
- **relative MAE@k** — MAE / true label for samples with label >= k (k=1, 5)

## Project Structure

```
├── main_bench.py            # Main training script (ArXiv + GitHub)
├── baseline_models.py       # MLP, LSTM, Transformer architectures
├── baseline_datasets.py     # Dataset loaders with ablation support
├── config.py                # Hyperparameter search space
├── sweep.sh                 # Grid search launcher
├── eval_mae.py              # MAE evaluation
├── log_to_df.py             # W&B experiment result aggregation
└── baseline_preprocess.py   # HF dataset to CSV conversion
```

## Requirements

- PyTorch
- Transformers
- Datasets (HuggingFace)
- Weights & Biases (`wandb`)
