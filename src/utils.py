import numpy as np
import torch
import ipdb as pdb
import math
from torch import linalg as LA
import wandb
import itertools
import random

########## Distance from Initialization #################

def model_dist(curr_model, init_model, pos=False, weight_only=False):
    dist= 0.0
    keys = list(curr_model.state_dict().keys())
    if not pos:
        keys = [key for key in curr_model.state_dict().keys() if 'pos_encoder' not in key]
    
    if weight_only:
        keys = [key for key in curr_model.state_dict().keys() if 'weight' in key or 'bias' in key]
    
    init_params= init_model.state_dict()
    curr_params= curr_model.state_dict()
    
    for key in keys:
        assert key in init_model.state_dict().keys()
        try:
            if 'float' in str(curr_params[key].dtype):
                x = curr_params[key] - init_params[key]
        except:
            pdb.set_trace()
        
        if len(x.size())>1:
            if len(x.size())>2:
                x = x.squeeze()
                assert len(x.size())==2
                
            norm = LA.matrix_norm(x, ord='fro')
        else:
            norm = LA.vector_norm(x, ord=2)
        
        # print(key, '\t', norm)

        dist += norm**2
    
    dist= math.sqrt(dist)
    return dist

def log_distance_from_initialization(curr_model, init_model, step):
    dist_dict = {}

    for name, param in curr_model.named_parameters():
        if param.requires_grad:
            init_param = init_model.state_dict()[name]
            if 'attn.c_attn.weight' in name:
                qkv_size = param.size(0) // 3
                dist_dict[f'query/{name}'] = (param[:qkv_size] - init_param[:qkv_size]).norm().item()
                dist_dict[f'key/{name}'] = (param[qkv_size:2*qkv_size] - init_param[qkv_size:2*qkv_size]).norm().item()
                dist_dict[f'value/{name}'] = (param[2*qkv_size:] - init_param[2*qkv_size:]).norm().item()
            elif 'attn.c_proj.weight' in name or 'attn.c_proj.bias' in name:
                dist_dict[f'attention_output/{name}'] = (param - init_param).norm().item()
            elif 'mlp.c_fc.weight' in name or 'mlp.c_fc.bias' in name:
                dist_dict[f'mlp_fc/{name}'] = (param - init_param).norm().item()
            elif 'mlp.c_proj.weight' in name or 'mlp.c_proj.bias' in name:
                dist_dict[f'mlp_proj/{name}'] = (param - init_param).norm().item()
            else:
                dist_dict[f'{name}'] = (param - init_param).norm().item()
    
    wandb.log({f"dist_from_init/{k}": v for k, v in dist_dict.items()}, step=step)





def model_sim(curr_model, init_model):
    sim_dict = {}
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    keys = list(curr_model.state_dict().keys())
    keys = [key for key in curr_model.state_dict().keys() if 'pos_encoder' not in key]
    
    

    keys = [key for key in keys if 'weight' in key]
    keys = [key for key in keys if 'attn' in key or 'mlp' in key]
    
    init_params= init_model.state_dict()
    curr_params= curr_model.state_dict()
    
    for key in keys:
        assert key in init_model.state_dict().keys()
        try:
            if 'float' in str(curr_params[key].dtype):
                # Cosine similarity between two vectors
                flat_x = curr_params[key].flatten()
                flat_y = init_params[key].flatten()
                sim = cos(flat_x, flat_y)
                sim_dict[key] = sim
        except:
            pdb.set_trace()
        
        
        
    return sim_dict