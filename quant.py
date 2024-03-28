# from __future__ import annotations

import numpy as np
import torch

class COMQ_Layer:
    
    def _quantizer(target_val, bit_code):

        input_vector_expanded = target_val.unsqueeze(-1).expand_as(bit_code)

        differences = torch.abs(bit_code - input_vector_expanded)

        _, min_indices = torch.min(differences, dim=1)

        closest_values_efficient = bit_code[torch.arange(bit_code.size(0)), min_indices]

        return closest_values_efficient

    def _quant_layer(W, iters, bits, X, greedy=False, scalar=1.0, device='cpu'):
        
        Q = W.clone()
           
        max_w, _ = torch.max(W, dim=1)
        min_w, _ = torch.min(W, dim=1)

        delta = (max_w - min_w)/(2**bits - 1)
        starts = (min_w / delta).round()
        ends = starts + (2**bits - 1)
        bit_code = scalar * torch.stack([torch.linspace(start, end, steps=2**bits) for start, end in zip(starts, ends)]).to(device)
        bit_code = bit_code * delta.unsqueeze(1)
      
        u = torch.zeros(X.shape[0], W.shape[0]).to(device)

        if greedy:
            
            need_perm_matrix = torch.norm(X, dim=0).unsqueeze(0) * W.abs()

            perm = torch.argsort(need_perm_matrix, dim=1, descending=True)

            del need_perm_matrix

            invperm = torch.argsort(perm)

            W = W.gather(1, perm)
            
            Q = Q.gather(1, perm)

            for _ in range(iters):

                for idx, w in enumerate(W.T):
            
                    X_permed = torch.index_select(X.T, 0, perm[:,idx]).T

                    u -= X_permed* (w - Q[:, idx]).unsqueeze(0)
                    
                    w_x = X_permed * w.unsqueeze(0)
                
                    target_val = torch.sum((w_x + u) * X_permed, dim=0) / torch.sum(X_permed*X_permed, dim=0)
                
                    q = COMQ_Layer._quantizer(target_val, bit_code)
                
                    u.add_(X_permed * (w - q).unsqueeze(0))
                
                    Q[:, idx] = q

                    del X_permed

            Q = Q.gather(1, invperm)
            W = W.gather(1, invperm)

        else:

            for _ in range(iters):

                for idx, w in enumerate(W.T):

                    u -= torch.outer(X[:, idx], w - Q[:, idx])
                        
                    u_x = torch.mv(u.T, X[:, idx])

                    w_x = torch.outer(X[:, idx], w)
                
                    target_val = (u_x + torch.mv(w_x.T, X[:, idx])) / (torch.sum(X[:, idx]**2))
                
                    q = COMQ_Layer._quantizer(target_val, bit_code)

                    u.add_(torch.outer(X[:, idx], w - q))
                
                    Q[:, idx] = q

        del bit_code, u
            
        quantize_error_XW = torch.norm(X @ W.T - X @ Q.T, p='fro')
        
        quantize_error_weight = torch.norm(W - Q, p='fro')
            
        return Q, quantize_error_XW, quantize_error_weight
                
