import time
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def get_optimal_ncl_acc(ys_list):
    # ys_list contains tensors of shape [bsize * wsize], one for each boolean task
    ys_combined = torch.cat(ys_list, dim=1)  # Shape: N × (M × num_tasks)

    # Count the number of -1s and 1s in each row
    counts_neg1 = (ys_combined == -1).sum(dim=1)  # Count of -1s
    counts_pos1 = (ys_combined == 1).sum(dim=1)   # Count of 1s

    # Find the most frequent value in each row
    most_frequent_count = torch.max(counts_neg1, counts_pos1)
    counts = torch.sum(most_frequent_count).item()
    
    bsize, wsize_num_tasks = ys_combined.shape  # wsize_num_tasks = wsize × num_tasks
    total_num = bsize * wsize_num_tasks
    acc = 100 * counts / total_num 
    return acc

def get_optimal_ncl_loss(ys_list, cont_ncl_loss_list=None):
    # ys_list contains tensors of shape [ncl_bsize × ncl_wsize], one for each continuous task
    # For individual evaluation, len(ys_list) == 1
    if cont_ncl_loss_list is None:
        # This branch is used only for individual task evaluation
        ys_combined = torch.cat(ys_list, dim=1) 
        # ys_combined.shape = [ncl_bsize, 1 × ncl_wsize]
        row_means = torch.mean(ys_combined, dim=1, keepdim=True)
        # row_means[i] represents the expected true label over w: E_w[f(x_i, w)]
        ncl_optimum_values = row_means.expand_as(ys_combined)
        # Expand the shape of row_means to match for MSE; broadcasting may suffice, but we expand for safety
        loss = mean_squared_error(ys_combined, ncl_optimum_values)
        return loss
    else:
        # This branch is used for mixed (all-task) evaluation
        n_task = len(ys_list)
        ncl_bsize, ncl_wsize = ys_list[0].shape
        opt_y_list = []

        for b in tqdm(range(ncl_bsize)):
            numerator = 0.0
            denominator = 0.0
            for i in range(n_task):
                for j in range(ncl_wsize):
                    numerator += (1 / cont_ncl_loss_list[i]) * ys_list[i][b, j]
                    denominator += (1 / cont_ncl_loss_list[i])
            opt_y = numerator / denominator
            opt_y_list.append(opt_y)

        opt_ys = torch.tensor(opt_y_list)
        opt_ys = opt_ys.unsqueeze(1).repeat(1, n_task * ncl_wsize)
        ys_combined = torch.cat(ys_list, dim=1)  # Shape: [ncl_bsize, ncl_wsize × n_task]
        loss = mean_squared_error(ys_combined, opt_ys)
        return loss


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()

def cross_entropy(ys_pred, ys):
    '''
    ys_pred: values in [-inf, inf]
    ys: binary targets in {-1, 1}
    '''
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)

def get_optimal_ncl_bool_loss_from_true_no_context_function(ys_list):
    # ys_list contains tensors of shape [bsize × wsize], one for each boolean task
    ys_combined = torch.cat(ys_list, dim=1)  # Shape: [N, M × num_tasks]

    # Compute the mode (most frequent value) along each row
    mode_vals, _ = torch.mode(ys_combined, dim=1)

    # Create a tensor filled with mode values
    ncl_opt_ys_combined = mode_vals.unsqueeze(1).expand_as(ys_combined)

    ncl_loss = cross_entropy(ncl_opt_ys_combined, ys_combined)
    return ncl_loss

def get_optimal_ncl_bool_loss(exp_name):
    if exp_name == 'conjunction_15':  # dim=15
        ncl_loss = 0.24639
    elif exp_name == 'disjunction_15':  # dim=15
        ncl_loss = 0.24386
    elif exp_name == 'sparse_parity_15_2':  # dim=15
        ncl_loss = 0.68954
    elif exp_name == 'sparse_parity_15_3':  # dim=15
        ncl_loss = 0.69385
    elif exp_name == 'conjunction_20':  # dim=20
        ncl_loss = 0.14085
    elif exp_name == 'disjunction_20':  # dim=20
        ncl_loss = 0.13822
    elif exp_name == 'sparse_parity_20_2':  # dim=20
        ncl_loss = 0.69102
    elif exp_name == 'sparse_parity_20_3':  # dim=20
        ncl_loss = 0.69464
    elif exp_name == 'conjunction_10':  # dim=10
        ncl_loss = 0.40751
    elif exp_name == 'disjunction_10':  # dim=10
        ncl_loss = 0.40316
    elif exp_name == 'sparse_parity_10_2':  # dim=10
        ncl_loss = 0.68295
    elif exp_name == 'sparse_parity_10_3':  # dim=10
        ncl_loss = 0.69017
    elif exp_name == 'parity_10':  # dim=10
        ncl_loss = 0.69656
    return ncl_loss