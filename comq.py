# from __future__ import annotations
import torch
import torch.nn as nn
import copy

from modelutils import Layer_input, extract_layers_vit, extract_layers_swin
from quant import COMQ_Layer

class COMQ():

    def __init__(self, model, data_loader, model_name,
                 bits, iters, scalar=1.0, greedy=False, device='cpu'):
        
        self.orig_model = model  
        self.data_loader_iter = iter(data_loader)
        self.bits = bits
        self.iters = iters
        self.scalar = scalar
        self.greedy = greedy
        self.device = device
        
        self.quantized_model = copy.deepcopy(self.orig_model)
        self.orig_layers = [] 
        self.quantized_layers = []
        if 'swin' in model_name:
            extract_layers_swin(self.orig_model, self.orig_layers)
            extract_layers_swin(self.quantized_model, self.quantized_layers)
            self.last_layer = [166]
        else:
            extract_layers_vit(self.orig_model, self.orig_layers)
            extract_layers_vit(self.quantized_model, self.quantized_layers)
            self.last_layer = [186]

    def quantize_model(self):

        layers_Q = [
            i for i, layer in enumerate(self.quantized_layers) 
                if type(layer) in {nn.Linear} and i not in self.last_layer
                ]
        print(f'Layer to be quantized {layers_Q}')
        print(f'Total num to quantize {len(layers_Q)-1}')     
        
        for layer_idx in layers_Q:
  
            layer_input = Layer_input(self.orig_model, layer_idx, self.orig_layers, self.data_loader_iter, self.device)

            print(f'\nQuantizing layer: {layer_idx}')

            if type(self.orig_layers[layer_idx]) == nn.Linear:

                W = self.orig_layers[layer_idx].weight.data

                Q, quantize_error, quantize_error_W = COMQ_Layer._quant_layer(W, self.iters, self.bits,
                                            layer_input, self.greedy, self.scalar, self.device)
                
                self.quantized_layers[layer_idx].weight.data = Q.float()

            print(f'The quantization error of layer {layer_idx} is {quantize_error}.')
            print(f'The weight quantization error of layer {layer_idx} is {quantize_error_W}.\n')
                
            del layer_input

        return self.quantized_model