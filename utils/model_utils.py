# model_utils.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils.model import Seq2Seq, Encoder, Decoder, Loss

def prepare_training_components(config, model):
    criterion = Loss()

    no_decay = [
        "bias",
        "layer_norm.weight", "layer_norm.bias",  
        ".norm1.weight", ".norm1.bias",         
        ".norm2.weight", ".norm2.bias",
        ".norm3.weight", ".norm3.bias",          
    ]

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # (optional) debug print
    if config.get("debug_no_decay", False):
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            bucket = "no_decay" if any(nd in name for nd in no_decay) else " decay "
            print(f"{name:50s} → {bucket}")

    optimizer_grouped_parameters = [
        {"params": decay_params,    "weight_decay": config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
    )

    def lr_lambda(epoch):
        if epoch < config["warmup_epochs"]:
            return float(epoch) / float(max(1, config["warmup_epochs"]))
        return max(
            0.0,
            float(config["n_epochs"] - epoch)
            / float(max(1, config["n_epochs"] - config["warmup_epochs"]))
        )
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
    
def _sync_models(models):
    """
    Synchronizes parameters from the primary model (models[0]) to all other models
    and zeros their gradients.
    """
    for m in models[1:]:
        for p0, p_other in zip(models[0].parameters(), m.parameters()):
            p_other.data.copy_(p0.data.to(p_other.device))
    # Zero gradients for models[1:].
    for m in models[1:]:
        for p in m.parameters():
            if p.grad is not None:
                p.grad.zero_()

