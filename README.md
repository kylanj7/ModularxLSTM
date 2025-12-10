# Modular xLSTM Training Pipeline

A compact, maintainable xLSTM (Extended Long Short-Term Memory) implementation with WandB integration. Designed for easy adaptation to different dataset types including text generation, time-series prediction, and more.

## Features

- ✅ **Compact & Maintainable** - ~300 lines of well-documented code
- ✅ **Modular Architecture** - Easy to swap datasets and customize
- ✅ **WandB Integration** - Automatic experiment tracking
- ✅ **Both sLSTM & mLSTM** - Alternating scalar and matrix LSTM blocks
- ✅ **Production Ready** - Gradient clipping, checkpointing, validation loops

## Installation
```bash
pip install torch pandas numpy wandb tqdm
```

## Quick Start

### 1. Text Generation (Default)
```python
# Prepare JSONL data
# train.jsonl: {"text": "Your training text here"}
# val.jsonl: {"text": "Your validation text here"}

# Run training
python xlstm_pipeline.py
```

### 2. Stock Price Prediction

Follow the conversion guide below to adapt for time-series data.

## Project Structure
```
├── xlstm_pipeline.py       # Main pipeline script
├── train.jsonl / train.csv # Training data
├── val.jsonl / val.csv     # Validation data
├── checkpoints/            # Model checkpoints
└── README.md              # This file
```

## Configuration

All hyperparameters are in the `Config` class:
```python
class Config:
    # Model Architecture
    vocab_size = 50000      # For text: vocabulary size
    embed_dim = 512         # Hidden dimension
    num_blocks = 6          # Number of xLSTM blocks
    block_types = ['slstm', 'mlstm']  # Alternating cell types
    
    # Training
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 10
    max_length = 512        # Sequence length
    grad_clip = 1.0
    
    # WandB
    use_wandb = True
    wandb_project = "xlstm-training"
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## Converting to Stock Price Prediction

This guide shows how to adapt the pipeline from text generation to stock market prediction.

### Step-by-Step Conversion

#### 1. Add Required Imports

**Add to the imports section:**
```python
import pandas as pd
import numpy as np
```

#### 2. Replace Dataset Class

**Find and replace `SimpleDataset` with:**
```python
class StockDataset(Dataset):
    """Load CSV stock data and create sequences for time-series prediction"""
    
    def __init__(self, path, seq_length=50, prediction_horizon=1, normalize=True):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        
        # Load and clean data
        df = pd.read_csv(path)
        df = df.sort_values('Date')  # Ensure chronological order
        
        # Clean price columns (remove $ and commas)
        price_cols = ['Close/Last', 'Open', 'High', 'Low']
        for col in price_cols:
            df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Select features: Close, Volume, Open, High, Low
        features = df[['Close/Last', 'Volume', 'Open', 'High', 'Low']].values
        
        # Normalize
        if normalize:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0)
            features = (features - self.mean) / (self.std + 1e-8)
        
        # Create sequences
        self.sequences = []
        for i in range(len(features) - seq_length - prediction_horizon + 1):
            input_seq = features[i:i + seq_length]
            target_seq = features[i + seq_length:i + seq_length + prediction_horizon]
            self.sequences.append((input_seq, target_seq))
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.float32),
            'labels': torch.tensor(target_seq, dtype=torch.float32)
        }
    
    def __len__(self):
        return len(self.sequences)
```

#### 3. Update Config Class

**Replace these parameters in `Config`:**
```python
class Config:
    # Model Architecture
    num_features = 5        # Changed from: vocab_size = 50000
    embed_dim = 512
    num_blocks = 6
    block_types = ['slstm', 'mlstm']
    
    # Training
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 10
    seq_length = 50         # Changed from: max_length = 512
    prediction_horizon = 1  # NEW: How many days ahead to predict
    normalize_data = True   # NEW: Whether to normalize features
    grad_clip = 1.0
    
    # Paths
    train_path = "train.csv"     # Changed from: "train.jsonl"
    val_path = "val.csv"         # Changed from: "val.jsonl"
    checkpoint_dir = "checkpoints"
    
    # WandB
    use_wandb = True
    wandb_project = "xlstm-stock-prediction"  # Changed project name
    wandb_run_name = None
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

#### 4. Update Model Input/Output Layers

**In the `xLSTM` class, replace:**
```python
# OLD (for text):
def __init__(self, config):
    super().__init__()
    self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
    self.blocks = nn.ModuleList([...])
    self.norm = nn.LayerNorm(config.embed_dim)
    self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    self.head.weight = self.embed.weight  # weight tying

def forward(self, x):
    x = self.embed(x)
    for block in self.blocks:
        x, _ = block(x, None)
    return self.head(self.norm(x))
```

**NEW (for stock):**
```python
def __init__(self, config):
    super().__init__()
    self.input_proj = nn.Linear(config.num_features, config.embed_dim)  # Changed
    self.blocks = nn.ModuleList([
        xLSTMBlock(config.embed_dim, config.block_types[i % len(config.block_types)])
        for i in range(config.num_blocks)
    ])
    self.norm = nn.LayerNorm(config.embed_dim)
    self.head = nn.Linear(config.embed_dim, config.num_features)  # Changed
    # Removed weight tying

def forward(self, x):
    x = self.input_proj(x)  # Changed
    for block in self.blocks:
        x, _ = block(x, None)
    x = self.head(self.norm(x))
    return x[:, -1:, :]  # Return only last timestep prediction
```

#### 5. Change Loss Function

**In `Trainer.__init__`, replace:**
```python
# OLD:
self.criterion = nn.CrossEntropyLoss()

# NEW:
self.criterion = nn.MSELoss()  # or nn.L1Loss() for MAE
```

#### 6. Update Training Loop

**In `train_epoch`, replace the loss calculation:**
```python
# OLD (for text):
logits = self.model(inputs)
loss = self.criterion(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1)
)

# NEW (for stock):
predictions = self.model(inputs)
loss = self.criterion(predictions, labels)
```

#### 7. Update Validation Loop

**In `validate`, replace:**
```python
# OLD (for text):
logits = self.model(inputs)
loss = self.criterion(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1)
)

# NEW (for stock):
predictions = self.model(inputs)
loss = self.criterion(predictions, labels)
```

#### 8. Update Main Function

**Replace the data loading section:**
```python
# OLD:
def simple_tokenizer(text):
    return [ord(c) % Config.vocab_size for c in text]

def main():
    config = Config()
    train_dataset = SimpleDataset(config.train_path, simple_tokenizer, config.max_length)
    val_dataset = SimpleDataset(config.val_path, simple_tokenizer, config.max_length)

# NEW:
def main():
    config = Config()
    train_dataset = StockDataset(
        config.train_path, 
        seq_length=config.seq_length,
        prediction_horizon=config.prediction_horizon,
        normalize=config.normalize_data
    )
    val_dataset = StockDataset(
        config.val_path,
        seq_length=config.seq_length,
        prediction_horizon=config.prediction_horizon,
        normalize=config.normalize_data
    )
```

### CSV Data Format

Your stock CSV should look like this:
```csv
Date,Close/Last,Volume,Open,High,Low
8/1/2025,$154.27,61286990,$155.05,$158.19,$151.06
7/31/2025,$158.35,45342610,$159.99,$160.89,$156.73
7/30/2025,$158.61,40261680,$157.37,$159.38,$156.56
```

**Requirements:**
- Must have columns: `Date`, `Close/Last`, `Volume`, `Open`, `High`, `Low`
- Prices can have `$` symbols (will be cleaned automatically)
- Sort by date (oldest to newest)

### Running Stock Prediction
```bash
python xlstm_pipeline.py
```

Monitor training in WandB dashboard for:
- Training loss
- Validation loss
- Model predictions vs actual values

---

## Adaptation Template for Other Datasets

Want to use this for audio, images, or other data? Follow this template:

### 1. Identify Your Data Characteristics

| Question | Text Example | Stock Example | Your Data |
|----------|--------------|---------------|-----------|
| Input type? | Discrete tokens | Continuous values | ? |
| Input shape? | `[batch, seq_len]` | `[batch, seq_len, features]` | ? |
| Output type? | Next token | Next value(s) | ? |
| Task type? | Classification | Regression | ? |
| Loss function? | CrossEntropyLoss | MSELoss | ? |

### 2. Modify Components

**Dataset Class:**
- Load your data format (CSV, JSONL, images, etc.)
- Implement appropriate preprocessing
- Return `{'input_ids': X, 'labels': Y}` format

**Config:**
- Set input dimensions (`num_features` or `vocab_size`)
- Set sequence parameters
- Add dataset-specific parameters

**Model:**
- Input layer: `nn.Embedding` (discrete) or `nn.Linear` (continuous)
- Output layer: Match your prediction dimensionality
- Weight tying: Only for text generation

**Trainer:**
- Loss function: Match your task (Classification/Regression/Other)
- Metrics: Add task-specific metrics

### 3. Test End-to-End
```python
# Quick shape test
config = Config()
dataset = YourDataset(config.train_path)
batch = dataset[0]
model = xLSTM(config)

# Check shapes
output = model(batch['input_ids'].unsqueeze(0))
print(f"Input shape: {batch['input_ids'].shape}")
print(f"Output shape: {output.shape}")
print(f"Label shape: {batch['labels'].shape}")
```
## Model Architecture

### xLSTM Blocks

The pipeline uses two types of LSTM cells:

**sLSTM (Scalar LSTM)**
- Exponential gating with stabilization
- Memory normalization
- Better for local patterns

**mLSTM (Matrix LSTM)**  
- Covariance-based memory updates
- Attention-like mechanisms
- Better for long-range dependencies

### Typical Configuration
```
Input → Embedding/Projection
  ↓
[sLSTM Block] → LayerNorm → Residual
  ↓
[mLSTM Block] → LayerNorm → Residual
  ↓
... (alternating)
  ↓
Output Projection → Predictions
```

---

## Advanced Usage

### Custom Tokenizer (Text)
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def custom_tokenizer(text):
    return tokenizer.encode(text, max_length=512, truncation=True)

# Use in dataset
dataset = SimpleDataset(path, custom_tokenizer, max_length=512)
```

### Multi-Step Prediction (Stock)
```python
# In Config:
prediction_horizon = 5  # Predict next 5 days

# Model output shape will be: [batch, 5, num_features]
```

### Custom Normalization
```python
# In StockDataset, replace normalization:
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
features = scaler.fit_transform(features)
```

### Resume Training
```python
# In main():
trainer = Trainer(model, train_loader, val_loader, config)
trainer.load_checkpoint("checkpoints/checkpoint_epoch_5.pt")
trainer.train()
```

## License

MIT License - Feel free to use and modify for your projects.


