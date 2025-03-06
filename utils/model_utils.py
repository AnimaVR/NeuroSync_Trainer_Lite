# model_utils.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import torch
import torch.optim as optim

from utils.model import Seq2Seq, Encoder, Decoder, Loss

def prepare_training_components(config, model):
    criterion = Loss(delta=config['delta'], w1=config['w1'], w2=config['w2'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return float(epoch) / float(max(1, config['warmup_epochs']))
        return max(0.0, float(config['n_epochs'] - epoch) / float(max(1, config['n_epochs'] - config['warmup_epochs'])))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
       
    return criterion, optimizer, scheduler

def build_model(config, device):
    encoder = Encoder(config['input_dim'], config['hidden_dim'], config['n_layers'], config['num_heads'], config['dropout'])
    decoder = Decoder(config['output_dim'], config['hidden_dim'], config['n_layers'], config['num_heads'], config['dropout'])
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model


def load_model(model_path, config, device):
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    dropout = config['dropout']
    num_heads = config['num_heads']

    encoder = Encoder(config['input_dim'], hidden_dim, n_layers, num_heads, dropout)
    decoder = Decoder(config['output_dim'], hidden_dim, n_layers, num_heads, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model

def save_final_model(model, final_model_path='out/model.pth'):
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        print(f"Initializing {m} with normal distribution")
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def count_parameters(model):
    """Count and print the number of parameters in a model."""
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {param_count}")
    return param_count


def calculate_gradient_norm(model):
    """Calculate and return the gradient norm for the model."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
