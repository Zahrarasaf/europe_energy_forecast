import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np

class EnergyTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Input projection
        self.input_projection = nn.Linear(
            config.input_size, 
            config.TRANSFORMER_PARAMS["d_model"]
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.TRANSFORMER_PARAMS["d_model"],
            config.TRANSFORMER_PARAMS["sequence_length"]
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.TRANSFORMER_PARAMS["d_model"],
            nhead=config.TRANSFORMER_PARAMS["nhead"],
            dropout=config.TRANSFORMER_PARAMS["dropout"],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            config.TRANSFORMER_PARAMS["num_layers"]
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(config.TRANSFORMER_PARAMS["d_model"], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.forecast_horizon)
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer_encoder(x)
        
        # Use only the last output for forecasting
        x = x[:, -1, :]
        
        # Output projection
        return self.output_layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100
        )
        return [optimizer], [scheduler]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
