# C-GPT: A Transformer Chess Engine

A decoder-only transformer trained on millions of chess games to predict next moves through autoregressive sequence modeling. The model learns to play chess purely from move sequences in standard algebraic notation (SAN), with no explicit knowledge of chess rules or board state.

## Project Structure

```
c-gpt/
├── configs/                   # TOML experiment configurations
│   ├── bishop.toml
│   └── knight.toml
├── scripts/
│   ├── preprocess.py          # Data pipeline: HuggingFace -> tokenized tensors
│   ├── train.py               # Training loop with checkpointing and evaluation
│   └── checks.py              # Sanity checks
├── src/
│   └── cgpt/
│       ├── model_base.py      # GPT model architecture
│       ├── evaluate.py        # Stockfish evaluation framework
│       ├── storage.py         # Local and GCS storage backends
│       ├── paths.py           # Project path constants
│       └── tests.py           # Tests
├── data/                      # Tokenized tensors and vocab mappings
├── models/                    # Saved checkpoints and evaluation results
├── configs/                   # Experiment configs
├── notebooks/                 # Development and analysis notebooks
├── misc/
│   └── san_strings/           # Chess move vocabulary
├── cloud_instance_setup.sh    # GCP VM setup script
└── pyproject.toml             # Package and dependency configuration
```

## Setup

### Local (Windows/Mac)

```bash
git clone <repo-url>
cd c-gpt
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

pip install -e .
```

Stockfish must be installed separately. On Windows, download from https://stockfishchess.org/download/ and set the path in your config TOML.

### Cloud (GCP)

```bash
git clone 
cd c-gpt
./cloud_instance_setup.sh
```

This installs Stockfish via apt, creates a virtual environment, and installs the package with cloud dependencies (`google-cloud-storage`).

On Linux, Stockfish is available on the system PATH after install, so set `stockfish_path = "stockfish"` in your config.

## Usage

All scripts are run from the project root with a config file as the argument.

**Preprocess data:**
```bash
python scripts/preprocess.py configs/bishop.toml
```

**Train:**
```bash
python scripts/train.py configs/bishop.toml
```

**Run checks:**
```bash
python scripts/checks.py configs/bishop.toml
```

## Configuration

Experiments are configured via TOML files in `configs/`. Example:

```toml
[model]
n_embd = 256
n_head = 8
n_layer = 8
block_size = 180
dropout = 0.2
save_name = "knight"

[training]
batch_size = 64
learning_rate = 3e-4
max_iters = 50000
device = "cuda"
continue_training = 0
# ...

[data]
dataset_name = "main_set"
min_elo = 2100

[evaluation]
stockfish_path = "stockfish"
# ...

[storage]
save_cloud = 0

[cloud]
bucket_name = "cgpt-main"
```

## Architecture

Decoder-only transformer with causal masking, following the GPT architecture. Move-level tokenization in standard algebraic notation. See the source in `src/cgpt/model_base.py` for details.

## Limitations

- No explicit board state representation forces the model to reconstruct position from move history
- Tactical depth is limited at 15M parameters
- The model plays at a beginner level, understanding piece movement and openings but lacking strategic depth

## Acknowledgments

- Architecture based on [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and the ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video
- Training data from the [angeluriot/chess_games](https://huggingface.co/datasets/angeluriot/chess_games) dataset
- Evaluation powered by [Stockfish](https://stockfishchess.org/)