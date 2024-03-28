import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

class InterruptException(Exception):
    pass

def extract_layers_vit(model, layers):     
    from timm.models.vision_transformer import PatchEmbed, Block, Attention, Mlp
    for layer in model.children():
        if type(layer) in {nn.Sequential, PatchEmbed, Block, Attention, Mlp}:
            extract_layers_vit(layer, layers)
        if not list(layer.children()):
            layers.append(layer)

def extract_layers_swin(model, layers):     
    from timm.models.swin_transformer import WindowAttention, SwinTransformerBlock, PatchMerging, PatchEmbed, BasicLayer, Mlp
    for layer in model.children():
        if type(layer) in {nn.Sequential, WindowAttention, SwinTransformerBlock, PatchMerging, PatchEmbed, BasicLayer, Mlp}:
            extract_layers_swin(layer, layers)
        if not list(layer.children()):
            layers.append(layer)

def Layer_input(model, layer_idx, layers, data, device):

        # get data
        input_data, _ = next(data)

        layer = layers[layer_idx]

        if type(layer) == nn.Linear:
            Input_ft = InputLinear()
        else:
            raise TypeError(f'{type(layer)} is not supported')

        handle = layers[layer_idx].register_forward_hook(Input_ft)
        
        with torch.no_grad():
            try:
                model(input_data.to(device))
            except InterruptException:
                pass

            handle.remove()

        del input_data

        return Input_ft.inputs[0].permute(2,0,1).flatten(1).T

class InputLinear:
    
    def __init__(self):
        self.inputs = []

    def __call__(self, module, module_in, module_out):

        if len(module_in) != 1:
            raise TypeError('The number of layer is not one!')
    
        self.inputs.append(module_in[0])
        raise InterruptException

def freeze_layers_vit(model):
    from timm.models.vision_transformer import PatchEmbed, Block, Attention, Mlp
    for layer in model.children():
        if type(layer) in {nn.Sequential, PatchEmbed, Block, Attention, Mlp}:
            freeze_layers_vit(layer)
        else:
            if type(layer) == nn.Conv2d:
                layer.weight.requires_grad = False
                # layer.bias.requires_grad = False
            elif type(layer) == nn.Linear:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

def freeze_layers_swin(model):
    from timm.models.swin_transformer import WindowAttention, SwinTransformerBlock, PatchMerging, PatchEmbed, BasicLayer, Mlp
    for layer in model.children():
        if type(layer) in {nn.Sequential, WindowAttention, SwinTransformerBlock, PatchMerging, PatchEmbed, BasicLayer, Mlp}:
            freeze_layers_swin(layer)
        else:
            if type(layer) == nn.Conv2d:
                layer.weight.requires_grad = False
            elif type(layer) == nn.Linear:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

def BatchNrom_tuning(model, model_name, train_loader, device):
    
    if 'swin' in model_name:
        freeze_layers_swin(model)
    else:
        freeze_layers_vit(model)
    
    for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    n_epochs = 1
    lr = 1 * 1e-4
    weight_decay = 1e-6 
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=lr, weight_decay=weight_decay)
    model.train()
    print(f'\n Start re-training: {device} is available, num_epochs {n_epochs}, \
                    lr {lr}\n')

    for epoch in range(1, n_epochs+1):
        batch_losses = []
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            loss = F.cross_entropy(output, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())

    model.eval()