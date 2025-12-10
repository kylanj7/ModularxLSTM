"""
Modular xLSTM Training Pipeline - Text Generation Version
Easy to maintain and adapt to different datasets
"""

import torch
import torch.nn as nn
import json
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# ==================== CONFIGURATION ====================

class Config:
    """Single config class for everything"""
    # Model Architecture
    vocab_size = 50000      # Vocabulary size for tokenization
    embed_dim = 512         # Hidden dimension size
    num_blocks = 6          # Number of xLSTM blocks
    block_types = ['slstm', 'mlstm']  # Alternating cell types
    dropout = 0.1           # Dropout rate
    
    # Training Hyperparameters
    batch_size = 16         # Batch size
    learning_rate = 3e-4    # Learning rate
    num_epochs = 10         # Number of training epochs
    max_length = 512        # Maximum sequence length
    grad_clip = 1.0         # Gradient clipping threshold
    
    # File Paths
    train_path = "train.jsonl"      # Path to training data (JSONL with 'text' field)
    val_path = "val.jsonl"          # Path to validation data
    checkpoint_dir = "checkpoints"  # Directory to save checkpoints
    
    # Weights & Biases Configuration
    use_wandb = True                    # Enable/disable W&B logging
    wandb_project = "xlstm-training"    # W&B project name
    wandb_run_name = None               # W&B run name (auto-generated if None)
    
    # Device Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== DATASET ====================

class SimpleDataset(Dataset):
    """
    Dataset for loading and processing text data from JSONL files.
    
    Features:
    - Loads JSONL files with 'text' field
    - Tokenizes text on-the-fly
    - Creates input-label pairs for next-token prediction
    
    Args:
        path: Path to JSONL file
        tokenizer_fn: Function to tokenize text
        max_length: Maximum sequence length
    """
    
    def __init__(self, path, tokenizer_fn, max_length=512):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                text = json.loads(line)['text']
                tokens = tokenizer_fn(text)[:max_length+1]
                if len(tokens) > 1:
                    self.data.append(tokens)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.data)

# ==================== MODEL COMPONENTS ====================

class sLSTMCell(nn.Module):
    """
    Scalar LSTM Cell with exponential gating and stabilization.
    
    Features:
    - Exponential gating mechanism
    - Numerical stabilization with m_t
    - Memory normalization with n_t
    
    State: (h, c, n, m) - 4 tensors
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Gate projections
        self.W_i = nn.Linear(dim*2, dim)  # Input gate
        self.W_f = nn.Linear(dim*2, dim)  # Forget gate
        self.W_o = nn.Linear(dim*2, dim)  # Output gate
        self.W_z = nn.Linear(dim*2, dim)  # Cell input
    
    def forward(self, x, state):
        # Initialize state if None
        if state is None:
            batch_size = x.size(0)
            device = x.device
            h = torch.zeros(batch_size, self.dim, device=device)
            c = torch.zeros(batch_size, self.dim, device=device)
            n = torch.zeros(batch_size, self.dim, device=device)
            m = torch.zeros(batch_size, self.dim, device=device)
            state = (h, c, n, m)
        
        h, c, n, m = state
        combined = torch.cat([x, h], -1)
        
        # Compute gates
        i_t = self.W_i(combined)
        f_t = self.W_f(combined)
        o_t = self.W_o(combined)
        z = torch.tanh(self.W_z(combined))
        
        # Stabilized exponential gating
        m_new = torch.max(f_t + m, i_t)
        i = torch.exp(i_t - m_new)
        f = torch.exp(f_t + m - m_new)
        
        # Update cell state with normalization
        c_new = f * c + i * z
        n_new = f * n + i
        h_new = torch.sigmoid(o_t) * (c_new / (n_new + 1e-6))
        
        return h_new, (h_new, c_new, n_new, m_new)

class mLSTMCell(nn.Module):
    """
    Matrix LSTM Cell with covariance-based memory update.
    
    Features:
    - Covariance matrix memory (C)
    - Query-Key-Value mechanism
    - Attention-like memory retrieval
    
    State: (C, n, m) - 3 tensors
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # QKV projections
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_i = nn.Linear(dim, dim)
        self.W_f = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
    
    def forward(self, x, state):
        # Initialize state if None
        if state is None:
            batch_size = x.size(0)
            device = x.device
            C = torch.zeros(batch_size, self.dim, self.dim, device=device)
            n = torch.zeros(batch_size, self.dim, device=device)
            m = torch.zeros(batch_size, self.dim, device=device)
            state = (C, n, m)
        
        C, n, m = state
        
        # Compute QKV
        q = self.W_q(x)
        k = self.W_k(x) / (self.dim ** 0.5)
        v = self.W_v(x)
        i_t = self.W_i(x)
        f_t = self.W_f(x)
        
        # Stabilized exponential gating
        m_new = torch.max(f_t + m, i_t)
        i = torch.exp(i_t - m_new).unsqueeze(-1)
        f = torch.exp(f_t + m - m_new).unsqueeze(-1)
        
        # Update covariance matrix
        C_new = f * C + i * torch.bmm(v.unsqueeze(-1), k.unsqueeze(-2))
        n_new = f.squeeze(-1) * n + i.squeeze(-1) * k
        
        # Retrieve from memory
        h = torch.sigmoid(self.W_o(x)) * torch.bmm(C_new, q.unsqueeze(-1)).squeeze(-1)
        
        return h, (C_new, n_new, m_new)

class xLSTMBlock(nn.Module):
    """
    xLSTM Block with pre-normalization and residual connections.
    
    Components:
    - LayerNorm before cell and FFN
    - sLSTM or mLSTM cell
    - Feed-forward network (FFN)
    - Residual connections
    """
    
    def __init__(self, dim, cell_type, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cell = sLSTMCell(dim) if cell_type == 'slstm' else mLSTMCell(dim)
        self.cell_type = cell_type
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, state):
        # Process sequence timestep by timestep
        batch_size, seq_len, dim = x.shape
        
        # Reset state (each block maintains its own)
        state = None
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, dim]
            normed = self.norm1(x_t)
            h, state = self.cell(normed, state)
            
            # Add residual and FFN
            out = x_t + h
            out = out + self.ffn(self.norm2(out))
            outputs.append(out)
        
        # Stack outputs back to [batch, seq_len, dim]
        x = torch.stack(outputs, dim=1)
        
        return x, state

class xLSTM(nn.Module):
    """
    Complete xLSTM model for text generation.
    
    Architecture:
    - Token embedding layer
    - Multiple xLSTM blocks (alternating sLSTM and mLSTM)
    - Output projection to vocabulary
    - Weight tying between embedding and output
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # xLSTM blocks
        self.blocks = nn.ModuleList([
            xLSTMBlock(
                config.embed_dim,
                cell_type=config.block_types[i % len(config.block_types)],
                dropout=config.dropout
            )
            for i in range(config.num_blocks)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        # Embed tokens
        x = self.embedding(x)
        x = self.embed_dropout(x)
        
        # Process through xLSTM blocks
        for block in self.blocks:
            x, _ = block(x, None)
        
        # Project to vocabulary
        x = self.output_norm(x)
        logits = self.lm_head(x)
        
        return logits

# ==================== TRAINER ====================

class Trainer:
    """
    Trainer class for xLSTM model.
    
    Features:
    - Training and validation loops
    - Gradient clipping
    - Learning rate scheduling (optional)
    - Checkpointing
    - WandB logging
    - Metric tracking and plotting
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_perplexity': []
        }
        
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        # Initialize WandB
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={
                    "vocab_size": config.vocab_size,
                    "embed_dim": config.embed_dim,
                    "num_blocks": config.num_blocks,
                    "block_types": config.block_types,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "num_epochs": config.num_epochs,
                    "max_length": config.max_length,
                }
            )
            wandb.watch(self.model, log="all", log_freq=100)
    
    def plot_training_history(self):
        """Plot training metrics"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot perplexity
        ax2.plot(epochs, self.history['val_perplexity'], 'g-s', label='Val Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Validation Perplexity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.close()
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")):
            inputs = batch['input_ids'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            # Forward pass
            logits = self.model(inputs)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log to WandB every 10 steps
            if self.config.use_wandb and step % 10 == 0:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + step
                })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            inputs = batch['input_ids'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            logits = self.model(inputs)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_perplexity = self.validate()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_perplexity'].append(val_perplexity)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Perplexity={val_perplexity:.2f}")
            
            # Log to WandB
            if self.config.use_wandb:
                wandb.log({
                    "train_loss_epoch": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
                    "epoch": epoch + 1
                })
            
            # Save checkpoint
            checkpoint_path = f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            
            # Log checkpoint to WandB
            if self.config.use_wandb:
                wandb.save(checkpoint_path)
        
        # Plot training history at the end
        self.plot_training_history()
        
        if self.config.use_wandb:
            wandb.log({"training_history": wandb.Image('training_history.png')})
            wandb.finish()

# ==================== MAIN ====================

def simple_tokenizer(text):
    """
    Simple character-level tokenizer.
    Replace with your actual tokenizer (e.g., HuggingFace, SentencePiece)
    """
    return [ord(c) % Config.vocab_size for c in text]

def main():
    config = Config()
    
    # Load data
    train_dataset = SimpleDataset(config.train_path, simple_tokenizer, config.max_length)
    val_dataset = SimpleDataset(config.val_path, simple_tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = xLSTM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == "__main__":
    main()
