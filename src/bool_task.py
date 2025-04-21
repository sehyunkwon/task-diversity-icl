import math

import torch
import numpy as np
import pdb
import time
import random
import itertools

current_time = int(time.time())

def generate_idx_list(n_dim, k, l, seed=current_time):
    # Function implemented to fix the w_b selected at each iteration in SparseParity.
    numbers = list(range(0, n_dim))
    combinations = list(itertools.combinations(numbers, k))
    
    if seed is not None:
        # Save the current random state
        state = random.getstate()
        # Set the seed
        random.seed(seed)
        # Shuffle randomly
        random.shuffle(combinations)
        # Restore the original random state
        random.setstate(state)
    else:
        random.shuffle(combinations)
    
    _idx_list = combinations[:l]
    idx_list = [list(tup) for tup in _idx_list]
    
    return idx_list


def accuracy(ys_pred, ys):
	return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
	'''
	ys_pred: [-inf, inf]
	ys: {-1, 1}
	'''
	output = sigmoid(ys_pred)
	target = (ys + 1) / 2
	return bce_loss(output, target)

def generate_idx(i, num, idx_list, k=2):
	# assert num == len(idx_list)
	# i % num -> One of [0, 1, 2, ..., len(idx_list) -1]

	if num == 0:
		return [[j for j in range(k)]]
	idx = random.choice(idx_list)

	return idx



class Task:
	def __init__(self, n_dims, batch_size, n_points, pool_dict=None, seeds=None):
		self.n_dims = n_dims
		self.b_size = batch_size
		self.pool_dict = pool_dict
		self.seeds = seeds
		self.data = 'bool'
		self.n_points = n_points
		assert pool_dict is None or seeds is None

	def evaluate(self, xs):
		raise NotImplementedError

	@staticmethod
	def get_metric():
		raise NotImplementedError

	@staticmethod
	def get_training_metric():
		raise NotImplementedError
	

def get_task_sampler(
	n_dims, batch_size, **kwargs
):
	task_names_to_classes = {
		"conjunction": Conjunction,
		# 'teach_biconjunction': TeachBiConjunction,
		# "mono_conjunction": MonoConjunction,
		# "teach_conjunction": TeachConjunction,
		"disjunction": Disjunction,
		# "sparse_disjunction": SparseDisjunction,
		# "nearest_neighbours": NearestNeighbours,
		"parity": Parity,
		"sparse_parity": SparseParity,
		"bool_retrieval": BoolRetrieval,
		"optimal_no_context_sparse_parity": OptimalNoContextSparseParity,
		"optimal_no_context_parity": OptimalNoContextParity,
		"optimal_no_context_conjunction": OptimalNoContextConjunction,
		"optimal_no_context_disjunction": OptimalNoContextDisjunction,
		# "majority": Majority,
		# "int_halfspace": IntHalfspace,
		# "dnf": DNF,
		# "teach_dnf": TeachDNF,
		# "cnf": CNF,
		# 'sparse_thres': SparseThreshold,
	}
	if kwargs['task'] in task_names_to_classes:
		task_cls = task_names_to_classes[kwargs['task']]
		return lambda **args: task_cls(n_dims, batch_size, **kwargs)
	else:
		print("Unknown task")
		raise NotImplementedError

class SparseParity(Task):
	ncl_opt_loss = None

	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(SparseParity, self).__init__(n_dims, batch_size, n_points)
		self.ncl = OptimalNoContextSparseParity(n_dims, batch_size, n_points, **kwargs)

		self.k = kwargs['k']
		self.l = kwargs['l']
		self.idx_list = generate_idx_list(n_dims, self.k, self.l)

		# if pool_dict is None:
		wb = []
		for i in range(self.b_size):
			idx = generate_idx(i, self.l, self.idx_list, self.k)

			# idx = np.random.choice(range(self.n_dims), self.k, replace=False)
			w = np.zeros(self.n_dims)
			w[idx] = 1
			wb.append(w)
		
		wb = np.array(wb)
		self.w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
	
	def NCL_optimal_loss(self, individual_ncl_loss):
		# Initialization
		if SparseParity.ncl_opt_loss is None:
			SparseParity.ncl_opt_loss = individual_ncl_loss

	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		return xs_b

	def sample_wb(self, b_size):
		wb = []
		for i in range(b_size):
			idx = generate_idx(i, self.l, self.idx_list, self.k)
			w = np.zeros(self.n_dims)
			w[idx] = 1
			wb.append(w)
		wb = np.array(wb)
		wb = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
		return wb

	def evaluate_by_given_wb(self, xs_b, w_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		return ys_b.sign()
		
	def evaluate(self, xs_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = self.w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		# noise = torch.randint(0,2, (128, 70)) * 2 -1
		# noise = noise.to(ys_b.device)
		# ys_b = torch.cat((ys_b[:128, :], noise), dim=0)
		return ys_b.sign()

	def ncl_evaluate(self, xs_b):
		ys_b = self.ncl.evaluate(xs_b)
		return ys_b

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy

class SparseParityTemp(Task):
	ncl_opt_loss = None

	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(SparseParityTemp, self).__init__(n_dims, batch_size, n_points)

		self.k = kwargs['k']
		self.l = kwargs['l']
		self.idx_list = generate_idx_list(n_dims, self.k, self.l)
		# print('idx_list: ', self.idx_list)

		# print(f'idx_list: {self.idx_list}')

		# if pool_dict is None:
		wb = []
		for i in range(self.b_size):
			idx = generate_idx(i, self.l, self.idx_list, self.k)

			# idx = np.random.choice(range(self.n_dims), self.k, replace=False)
			w = np.zeros(self.n_dims)
			w[idx] = 1
			wb.append(w)
		
		wb = np.array(wb)
		self.w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
	
	def NCL_optimal_loss(self, individual_ncl_loss):
		# Initialization
		if SparseParity.ncl_opt_loss is None:
			SparseParity.ncl_opt_loss = individual_ncl_loss

	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		return xs_b

	def sample_wb(self, b_size):
		wb = []
		for i in range(b_size):
			idx = generate_idx(i, self.l, self.idx_list, self.k)
			w = np.zeros(self.n_dims)
			w[idx] = 1
			wb.append(w)
		wb = np.array(wb)
		wb = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
		return wb

	def evaluate_by_given_wb(self, xs_b, w_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		return ys_b.sign()
		
	def evaluate(self, xs_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = self.w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		# noise = torch.randint(0,2, (128, 70)) * 2 -1
		# noise = noise.to(ys_b.device)
		# ys_b = torch.cat((ys_b[:128, :], noise), dim=0)
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy


class OptimalNoContextDisjunction(Task):
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(OptimalNoContextDisjunction, self).__init__(n_dims, batch_size, n_points)

		self.conjunction = DisjunctionTemp(n_dims, batch_size, n_points, **kwargs)
		how_many_wb = 2**self.n_dims
		self.all_wb = torch.tensor(np.random.choice([0, 1, -1], size=(how_many_wb, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		self.ncl_opt_loss = None

		all_xs = torch.cartesian_prod(*[torch.tensor([0, 1]) for _ in range(self.n_dims)])
		all_xs = all_xs.float()*2-1
		all_xs = all_xs.view(1, 2**self.n_dims, self.n_dims)

		candidates = []
		for i in range(how_many_wb):
			w = self.all_wb[i].unsqueeze(0)  # -> w.shape = torch.Size([1, n_dim, 1])
			output = self.conjunction.evaluate_by_given_wb(all_xs, w)
			candidates.append(output)
		
		candidates = torch.stack(candidates).squeeze()  # -> candidates.shape = torch.Size([how_may_wb, 2**n_dim])
		candidates = candidates.T	# -> candidates.shape = torch.Size([2**n_dim, how_may_wb])

		majority_values = torch.sign(torch.sum(candidates, dim=1, keepdim=True))
		majority_values_flat = majority_values.view(-1)	# -> majority_values_flat:  torch.Size([1024])

		self.no_context_function = {}
		for i in range(len(majority_values_flat)):
			key = tuple(all_xs[:, i, :].squeeze().numpy().tolist())  # key has to be immutable. Therefore, convert to tuple.
			# Extract the value from the tensor
			value = majority_values_flat[i].item()
			# Add to dictionary
			self.no_context_function[key] = value
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		### original paper uses below setting ###
		# for b in range(b_size):
		# 	wb, k = self.w_b[b], self.kw[b]            
		# 	pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
		# 	nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
		# 	for i in range(n_points):
		# 		if np.random.choice([0, 1], p=[0.7, 0.3]):
		# 			xs_b[b, i, pidx] = +1.0
		# 			xs_b[b, i, nidx] = -1.0
		# 			assert (xs_b[b, i, :] @ wb).squeeze() >= k

		return xs_b
	
	def evaluate(self, xs_b):
		ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1])

		for i in range(len(xs_b)):
			for j in range(xs_b.shape[1]):
				key = tuple(xs_b[i, j, :].numpy())
				
				if key in self.no_context_function:
					value = self.no_context_function[key]
				else:
					raise ValueError(f"{key} is invalid input.")
    
				ys_b[i, j] = value

		return ys_b

	def evaluate_by_given_wb(self, xs_b, w_b):
		return self.evaluate(xs_b)
	
	def sample_wb(self, b_size):
		return None

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	
class OptimalNoContextConjunction(Task):
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(OptimalNoContextConjunction, self).__init__(n_dims, batch_size, n_points)

		self.conjunction = ConjunctionTemp(n_dims, batch_size, n_points, **kwargs)
		how_many_wb = 2**self.n_dims
		self.all_wb = torch.tensor(np.random.choice([0, 1, -1], size=(how_many_wb, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		self.ncl_opt_loss = None

		all_xs = torch.cartesian_prod(*[torch.tensor([0, 1]) for _ in range(self.n_dims)])
		all_xs = all_xs.float()*2-1
		all_xs = all_xs.view(1, 2**self.n_dims, self.n_dims)

		candidates = []
		for i in range(how_many_wb):
			w = self.all_wb[i].unsqueeze(0)  # -> w.shape = torch.Size([1, n_dim, 1])
			output = self.conjunction.evaluate_by_given_wb(all_xs, w)
			candidates.append(output)
		
		candidates = torch.stack(candidates).squeeze()  # -> candidates.shape = torch.Size([how_may_wb, 2**n_dim])
		candidates = candidates.T	# -> candidates.shape = torch.Size([2**n_dim, how_may_wb])

		majority_values = torch.sign(torch.sum(candidates, dim=1, keepdim=True))
		majority_values_flat = majority_values.view(-1)	# -> majority_values_flat:  torch.Size([1024])

		self.no_context_function = {}
		for i in range(len(majority_values_flat)):
			key = tuple(all_xs[:, i, :].squeeze().numpy().tolist())  # key has to be immutable. Therefore, convert to tuple.
			# Extract the value from the tensor
			value = majority_values_flat[i].item()
			# Add to dictionary
			self.no_context_function[key] = value
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		### original paper uses below setting ###
		# for b in range(b_size):
		# 	wb, k = self.w_b[b], self.kw[b]            
		# 	pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
		# 	nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
		# 	for i in range(n_points):
		# 		if np.random.choice([0, 1], p=[0.7, 0.3]):
		# 			xs_b[b, i, pidx] = +1.0
		# 			xs_b[b, i, nidx] = -1.0
		# 			assert (xs_b[b, i, :] @ wb).squeeze() >= k

		return xs_b
	
	def evaluate(self, xs_b):
		ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1])

		for i in range(len(xs_b)):
			for j in range(xs_b.shape[1]):
				key = tuple(xs_b[i, j, :].numpy())
				
				if key in self.no_context_function:
					value = self.no_context_function[key]
				else:
					raise ValueError(f"{key} is invalid input.")
    
				ys_b[i, j] = value

		return ys_b

	def evaluate_by_given_wb(self, xs_b, w_b):
		return self.evaluate(xs_b)

	def sample_wb(self, b_size):
		return None

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	

class Conjunction(Task):
	ncl_opt_loss = None
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(Conjunction, self).__init__(n_dims, batch_size, n_points)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		# k = int(n_dims/3)
		# if pool_dict is None:
		self.ncl = OptimalNoContextConjunction(n_dims, batch_size, n_points, **kwargs)
		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
	
	def NCL_optimal_loss(self, individual_ncl_loss):
		# Initialization
		if Conjunction.ncl_opt_loss is None:
			Conjunction.ncl_opt_loss = individual_ncl_loss
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		### original paper uses below setting ###
		# for b in range(b_size):
		# 	wb, k = self.w_b[b], self.kw[b]            
		# 	pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
		# 	nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
		# 	for i in range(n_points):
		# 		if np.random.choice([0, 1], p=[0.7, 0.3]):
		# 			xs_b[b, i, pidx] = +1.0
		# 			xs_b[b, i, nidx] = -1.0
		# 			assert (xs_b[b, i, :] @ wb).squeeze() >= k

		return xs_b

	def sample_wb(self, b_size):
		wb = torch.tensor(np.random.choice([0, 1, -1], size=(b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		return wb
	
	def evaluate_by_given_wb(self, xs_b, w_b):
		w_b = w_b.to(xs_b.device)
		kw = torch.norm(w_b, p=1, dim=1) - 1
		ys_b = (xs_b @ w_b).squeeze() - kw
		return ys_b.sign()
		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - self.kw
		return ys_b.sign()

	def ncl_evaluate(self, xs_b):
		ys_b = self.ncl.evaluate(xs_b)
		return ys_b

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy

class ConjunctionTemp(Task):
	ncl_opt_loss = None
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(ConjunctionTemp, self).__init__(n_dims, batch_size, n_points)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		# k = int(n_dims/3)
		# if pool_dict is None:
		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
	
	def NCL_optimal_loss(self, individual_ncl_loss):
		# Initialization
		if ConjunctionTemp.ncl_opt_loss is None:
			ConjunctionTemp.ncl_opt_loss = individual_ncl_loss
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		### original paper uses below setting ###
		# for b in range(b_size):
		# 	wb, k = self.w_b[b], self.kw[b]            
		# 	pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
		# 	nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
		# 	for i in range(n_points):
		# 		if np.random.choice([0, 1], p=[0.7, 0.3]):
		# 			xs_b[b, i, pidx] = +1.0
		# 			xs_b[b, i, nidx] = -1.0
		# 			assert (xs_b[b, i, :] @ wb).squeeze() >= k

		return xs_b

	def sample_wb(self, b_size):
		wb = torch.tensor(np.random.choice([0, 1, -1], size=(b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		return wb
	
	def evaluate_by_given_wb(self, xs_b, w_b):
		w_b = w_b.to(xs_b.device)
		kw = torch.norm(w_b, p=1, dim=1) - 1
		ys_b = (xs_b @ w_b).squeeze() - kw
		return ys_b.sign()
		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	
	
class Parity(Task):
	ncl_opt_loss = None
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(Parity, self).__init__(n_dims, batch_size, n_points)
		self.n_dims = n_dims
		funcs = np.random.choice(2**n_dims, size = batch_size)
		all_subsets  = self.generate_subsets(n_dims)
		self.w_b = torch.zeros(size= (batch_size, n_dims, 1))
		# Approximate 35% of indices will be 1
		# self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
		for i in range(batch_size):
			self.w_b[i, all_subsets[funcs[i]]] = 1
	
	def NCL_optimal_loss(self, individual_ncl_loss):
		# Initialization
		if Parity.ncl_opt_loss is None:
			Parity.ncl_opt_loss = individual_ncl_loss
		
	
	def sample_xs(self, n_points, b_size):
		# Input distribution is uniform over {-1, 1}^n_dims
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		return xs_b

	def sample_wb(self, b_size):
		funcs = np.random.choice(2**self.n_dims, size = b_size)
		all_subsets  = self.generate_subsets(self.n_dims)
		wb = torch.zeros(size= (b_size, self.n_dims, 1))
		# self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
		for i in range(b_size):
			wb[i, all_subsets[funcs[i]]] = 1
		return wb

	def evaluate_by_given_wb(self, xs_b, w_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		return ys_b.sign()
		
	def evaluate(self, xs_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = self.w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		# noise = torch.randint(0,2, (128, 70)) * 2 -1
		# noise = noise.to(ys_b.device)
		# ys_b = torch.cat((ys_b[:128, :], noise), dim=0)
		return ys_b.sign()


	def generate_subsets(self, n):
		subsets = []
		for i in range(2**n):
			subset = [j for j in range(n) if (i & 1 << j)]
			subsets.append(subset)
		return subsets
	
	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	
	
class OptimalNoContextParity(Task):
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(OptimalNoContextParity, self).__init__(n_dims, batch_size, n_points)

		self.parity = Parity(n_dims, batch_size, n_points, **kwargs)

		self.n_dims = n_dims
		funcs = np.random.choice(2**n_dims, size = 2**n_dims)
		all_subsets  = self.parity.generate_subsets(n_dims)
		self.all_wb = torch.zeros(size= (2**n_dims, n_dims, 1))

		for i in range(len(all_subsets)):
			self.all_wb[i, all_subsets[funcs[i]]] = 1

		self.ncl_opt_loss = None
		
		all_xs = torch.cartesian_prod(*[torch.tensor([0, 1]) for _ in range(self.n_dims)])
		all_xs = all_xs.float()*2-1
		all_xs = all_xs.view(1, 2**self.n_dims, self.n_dims)

		candidates = []
		for i in range(len(all_subsets)):
			w = self.all_wb[i].unsqueeze(0)
			output = self.parity.evaluate_by_given_wb(all_xs, w)
			candidates.append(output)
		
		candidates = torch.stack(candidates)
		candidates = candidates.T

		majority_values = torch.sign(torch.sum(candidates, dim=1, keepdim=True))
		majority_values_flat = majority_values.view(-1)

		self.no_context_function = {}
		for i in range(len(majority_values_flat)):
			key = tuple(all_xs[:, i, :].squeeze().numpy()) 
			value = majority_values_flat[i].item()   
			self.no_context_function[key] = value

	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		return xs_b
		
	def evaluate(self, xs_b):
		ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1])

		for i in range(len(xs_b)):
			for j in range(xs_b.shape[1]):
				key = tuple(xs_b[i, j, :].numpy()) 

				if key in self.no_context_function:
					value = self.no_context_function[key]
				else:
					raise ValueError(f"{key} is invalid input.")
				
				ys_b[i, j] = value

		return ys_b

	def evaluate_by_given_wb(self, xs_b, w_b):
		return self.evaluate(xs_b)
	
	def sample_wb(self, b_size):
		return None

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	
	
class OptimalNoContextSparseParity(Task):
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(OptimalNoContextSparseParity, self).__init__(n_dims, batch_size, n_points)

		self.sparse_parity = SparseParityTemp(n_dims, batch_size, n_points, **kwargs)
	
		self.k = kwargs['k']
		self.l = kwargs['l']
		self.idx_list = generate_idx_list(n_dims, self.k, self.l)
		
		all_wb = []
		for i in range(len(self.idx_list)):
			idx = self.idx_list[i]
			w = np.zeros(self.n_dims)
			w[idx] = 1
			all_wb.append(w)
		
		all_wb = np.array(all_wb)
		self.all_wb = torch.tensor(all_wb, dtype=torch.float).unsqueeze(2)

		all_xs = torch.cartesian_prod(*[torch.tensor([0, 1]) for _ in range(self.n_dims)])
		all_xs = all_xs.float()*2-1
		all_xs = all_xs.view(1, 2**self.n_dims, self.n_dims)

		candidates = []
		for i in range(len(self.idx_list)):
			w = self.all_wb[i].unsqueeze(0)
			output = self.sparse_parity.evaluate_by_given_wb(all_xs, w)
			candidates.append(output)
		
		candidates = torch.stack(candidates)
		candidates = candidates.T
		
		majority_values = torch.sign(torch.sum(candidates, dim=1, keepdim=True))
		majority_values_flat = majority_values.view(-1)

		self.no_context_function = {}
		for i in range(len(majority_values_flat)):
			key = tuple(all_xs[:, i, :].squeeze().numpy())
			value = majority_values_flat[i].item()
			self.no_context_function[key] = value

		self.ncl_opt_loss = None
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		return xs_b
		
	def evaluate(self, xs_b):
		ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1])

		for i in range(len(xs_b)):
			for j in range(xs_b.shape[1]):
				key = tuple(xs_b[i, j, :].numpy())
				
				if key in self.no_context_function:
					value = self.no_context_function[key]
				else:
					raise ValueError(f"{key} is invalid input.")
    
				ys_b[i, j] = value

		return ys_b

	def evaluate_by_given_wb(self, xs_b, w_b):
		return self.evaluate(xs_b)
	
	def sample_wb(self, b_size):
		return None

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	

class Disjunction(Task):
	ncl_opt_loss = None
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(Disjunction, self).__init__(n_dims, batch_size, n_points)
		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
  
		# self.ncl = OptimalNoContextDisjunction(n_dims, batch_size, n_points, **kwargs)

	def NCL_optimal_loss(self, individual_ncl_loss):
		# Initialization
		if Disjunction.ncl_opt_loss is None:
			Disjunction.ncl_opt_loss = individual_ncl_loss
			
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		# Manipulate the input to create negative examples to make a more balanced dataset
		# for b in range(b_size):
		# 	wb, k = self.w_b[b], self.kw[b]            
		# 	pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
		# 	nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
		# 	for i in range(n_points):
		# 		if np.random.choice([0, 1], p=[0.7, 0.3]):
		# 			xs_b[b, i, pidx] = -1.0
		# 			xs_b[b, i, nidx] = +1.0
		# 			assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

		return xs_b
		
	def sample_wb(self, b_size):
		wb = torch.tensor(np.random.choice([0, 1, -1], size=(b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		return wb
	
	def evaluate_by_given_wb(self, xs_b, w_b):
		w_b = w_b.to(xs_b.device)
		kw = torch.norm(w_b, p=1, dim=1) - 1
		ys_b = (xs_b @ w_b).squeeze() + kw
		return ys_b.sign()
		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() + self.kw
		return ys_b.sign()

	def ncl_evaluate(self, xs_b):
		ys_b = self.ncl.evaluate(xs_b)
		return ys_b

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy

def generate_kv(num_keys, n_dim, num_values=4, seed=current_time):
    original_state = torch.get_rng_state()

    torch.manual_seed(seed)

    all_combinations = torch.tensor(list(itertools.product([-1, 1], repeat=n_dim)))
    total_combinations = all_combinations.size(0)

    indices = torch.randperm(total_combinations)[:num_keys]
    key = all_combinations[indices]
    value_list = []

    for _ in range(num_values):
        value = torch.randint(0, 2, (num_keys,)) * 2 - 1
        value_list.append(value)

    torch.set_rng_state(original_state)  

    return key, value_list

class DisjunctionTemp(Task):
	ncl_opt_loss = None
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(DisjunctionTemp, self).__init__(n_dims, batch_size, n_points)
		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		self.kw = torch.norm(self.w_b, p=1, dim=1) - 1

	def NCL_optimal_loss(self, individual_ncl_loss):
		# Initialization
		if DisjunctionTemp.ncl_opt_loss is None:
			DisjunctionTemp.ncl_opt_loss = individual_ncl_loss
			
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		# Manipulate the input to create negative examples to make a more balanced dataset
		# for b in range(b_size):
		# 	wb, k = self.w_b[b], self.kw[b]            
		# 	pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
		# 	nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
		# 	for i in range(n_points):
		# 		if np.random.choice([0, 1], p=[0.7, 0.3]):
		# 			xs_b[b, i, pidx] = -1.0
		# 			xs_b[b, i, nidx] = +1.0
		# 			assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

		return xs_b
		
	def sample_wb(self, b_size):
		wb = torch.tensor(np.random.choice([0, 1, -1], size=(b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		return wb
	
	def evaluate_by_given_wb(self, xs_b, w_b):
		w_b = w_b.to(xs_b.device)
		kw = torch.norm(w_b, p=1, dim=1) - 1
		ys_b = (xs_b @ w_b).squeeze() + kw
		return ys_b.sign()
		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() + self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	

class BoolRetrieval(Task):
	def __init__(self, n_dims, batch_size, n_points, **kwargs):
		super(BoolRetrieval, self).__init__(n_dims, batch_size, n_points)
		self.n_dims = n_dims
		self.num_keys = kwargs['num_keys']
		self.num_values = kwargs['num_values']

		keys, value_list = generate_kv(self.num_keys, self.n_dims, self.num_values)
		self.keys = keys
		self.value_list = value_list

		self.ncl_opt_loss = None
        
	def sample_xs_and_ys(self, n_points, b_size):
		random_indices = torch.randint(0, self.num_keys, (b_size , n_points))
		xs_b = self.keys[random_indices]

		random_int = random.randint(0, self.num_values-1)
		ys_b = self.value_list[random_int][random_indices]

		for i in range(b_size):
			query_pos = np.random.randint(0 , n_points)
			xs_b[i][-1,:] = xs_b[i][query_pos, :]
			ys_b[i][-1] = ys_b[i][query_pos]

		return xs_b, ys_b
	
	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy


# class TeachBiConjunction(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(TeachBiConjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)

# 		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
# 		self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
	
# 	def sample_xs(self, n_points, b_size):
# 		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

# 		for b in range(b_size):
# 			wb, k = self.w_b[b], self.kw[b]            
# 			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
# 			for i in range(n_points):
# 				if np.random.choice([0, 1], p=[0.6, 0.4]):
# 					xs_b[b, i, pidx] = +1.0
# 					xs_b[b, i, nidx] = -1.0
# 					assert (xs_b[b, i, :] @ wb).squeeze() >= k
		
# 		# Adding teaching sequence in the beginning of samples

# 		for b in range(b_size):
# 			wb = self.w_b[b].squeeze()
# 			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
# 			ex  = len(pidx) + len(nidx) + 2
# 			new_ex  = wb.repeat(ex, 1)  # new_ex shape: (ex, n_dims)

# 			for i in range(self.n_dims):
# 				if i not in pidx and i not in nidx:
# 					new_ex[0, i] = -1.0

# 			for i in range(self.n_dims):
# 				if i not in pidx and i not in nidx:
# 					new_ex[1, i] = 1.0

# 			for k in range(2, ex):
# 				for i in range(self.n_dims):
# 					if i not in pidx and i not in nidx:
# 						new_ex[k, i] = -1.0


# 			cx = 2
# 			for i in range(len(pidx)):
# 				new_ex[cx, pidx[i]] = -1.0
# 				cx += 1
			
# 			for i in range(len(nidx)):
# 				new_ex[cx, nidx[i]] = 1.0
# 				cx += 1
			
# 			assert cx == ex
			

# 			# idx = torch.randperm(len(new_ex))
# 			# new_ex = new_ex[idx]
# 			xs_b[b, 0:ex, :] = new_ex

# 		return xs_b

		
# 	def evaluate(self, xs_b):
# 		w_b = self.w_b.to(xs_b.device)
# 		ys_b = (xs_b @ w_b).squeeze() - self.kw
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy


# class MonoConjunction(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
# 		super(MonoConjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)

# 		self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[2/3, 1/3]), dtype=torch.float)
# 		self.kw = self.w_b.sum(dim=1) - 1
	
# 	def sample_xs(self, n_points, b_size):

# 		xs_b = torch.tensor(np.random.choice([0, 1], size=(b_size, n_points, self.n_dims), p=[1-self.p, self.p]), dtype=torch.float)*2-1 

# 		return xs_b

		
# 	def evaluate(self, xs_b):
# 		w_b = self.w_b.to(xs_b.device)
# 		ys_b = (xs_b @ w_b).squeeze() - self.kw
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy
	



# class TeachConjunction(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(TeachConjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)
# 		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
# 		self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)
# 		self.kw = self.w_b.sum(dim=1) - 1
	
# 	def sample_xs(self, n_points, b_size):
# 		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

# 		for b in range(b_size):
# 			wb, k = self.w_b[b], self.kw[b]
# 			tidx = [i for i in range(self.n_dims) if wb[i] == 1]
# 			for i in range(n_points):
# 				if np.random.choice([0, 1], p=[0.6, 0.4]):
# 					xs_b[b, i, tidx] = +1.
# 					assert (xs_b[b, i, :] @ wb).squeeze() >= k
		
# 		# Adding teaching sequence in the beginning of samples
# 		for b in range(b_size):
# 			wb = self.w_b[b].squeeze()
# 			tidx = [i for i in range(self.n_dims) if wb[i] == 1]
# 			ex  = len(tidx) + 1
# 			new_ex  = wb.repeat(ex, 1)
# 			for i in range(len(tidx)):
# 				cx = i+1
# 				new_ex[cx, tidx[i]] = 0
			
# 			new_ex = new_ex * 2 - 1

# 			# idx = torch.randperm(len(new_ex))
# 			# new_ex = new_ex[idx]
# 			xs_b[b, 0:ex, :] = new_ex
			

# 		return xs_b

		
# 	def evaluate(self, xs_b):
# 		w_b = self.w_b.to(xs_b.device)
# 		ys_b = (xs_b @ w_b).squeeze() - self.kw
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy


# class SparseDisjunction(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(SparseDisjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)

# 		self.k = 4
# 		if pool_dict is None:
# 			wb = []
# 			for i in range(self.b_size):
# 				idx = np.random.choice(range(self.n_dims), self.k, replace=False)
# 				w = np.zeros(self.n_dims)
# 				w[idx] = 1
# 				wb.append(w)
			
# 			wb = np.array(wb)
# 			self.w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
# 			self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
# 			self.xs_b = None
			
			
			
# 		else:
# 			assert 'w' in pool_dict
# 			indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
# 			self.w_b = pool_dict["w"][indices]
# 			self.kw = pool_dict["kw"][indices]
# 			self.xs_b = pool_dict["xs"][indices]
			
	
# 	def sample_xs(self, n_points, b_size):
# 		if self.xs_b is not None:
# 			# Using pre-generated xs
# 			return self.xs_b
# 		else:
# 			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1


# 			# Manipulate the input to create negative examples to make a more balanced dataset
# 			for b in range(b_size):
# 				wb, k = self.w_b[b], self.kw[b]            
# 				pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 				nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
# 				for i in range(n_points):
# 					if np.random.choice([0, 1], p=[0.7, 0.3]):
# 						xs_b[b, i, pidx] = -1.0
# 						xs_b[b, i, nidx] = +1.0
# 						assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

# 			return xs_b


# 	# @staticmethod
# 	# def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
# 	# 	# w_b shape: (num_tasks, n_dims, 1)
# 	# 	start_time = time()
# 	# 	k = 4
# 	# 	wb = []
# 	# 	for i in range(num_tasks):
# 	# 		idx = np.random.choice(range(n_dims), k, replace=False)
# 	# 		w = np.zeros(n_dims)
# 	# 		w[idx] = 1
# 	# 		wb.append(w)
		
# 	# 	wb = np.array(wb)
# 	# 	w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
# 	# 	kw = torch.norm(w_b, p=1, dim=1) - 1


# 	# 	xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1

# 	# 	for b in range(num_tasks):
# 	# 		wb, k = w_b[b], kw[b]            
# 	# 		pidx = [i for i in range(n_dims) if wb[i] == 1.0]
# 	# 		nidx = [i for i in range(n_dims) if wb[i] == -1.0]
# 	# 		for i in range(n_points):
# 	# 			if np.random.choice([0, 1], p=[0.7, 0.3]):
# 	# 				xs_b[b, i, pidx] = -1.0
# 	# 				xs_b[b, i, nidx] = +1.0
# 	# 				assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

# 	# 	end_time = time()
# 	# 	print('Time to generate pool dict: {:.2f} mins {:.2f} secs'.format((end_time-start_time)//60, (end_time-start_time)%60))


# 	# 	return {"w": w_b, "kw": kw, "xs": xs_b}
		
# 	def evaluate(self, xs_b):
# 		w_b = self.w_b.to(xs_b.device)
# 		ys_b = (xs_b @ w_b).squeeze() + self.kw
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy



# class NearestNeighbours(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, start_idx=0):
# 		super(NearestNeighbours, self).__init__(n_dims, batch_size, pool_dict, seeds)
# 		self.start_idx = start_idx

# 		if pool_dict is None:
# 			self.xs_b = None
# 		else:
# 			indices = torch.randperm(len(pool_dict["xs"]))[:batch_size]
# 			self.xs_b = pool_dict["xs"][indices]
# 			# self.ys_b = pool_dict["ys"][indices]
# 	# def check_unique(self, xs_b):
# 	# 	temp_xs = xs_b[:, :self.start_idx, :] # bs x start_idx x n_dims
# 	# 	temp_xs_2d = temp_xs.reshape(-1, temp_xs.shape[2]) # bs * n_points x n_dims
# 	# 	_, inverse_indices = torch.unique(temp_xs_2d, dim=0, return_inverse=True)
# 	# 	inverse_indices = inverse_indices.reshape(temp_xs.shape[0], temp_xs.shape[1]) # bs x start_idx
# 	# 	for row in inverse_indices:
# 	# 		if len(torch.unique(row)) != self.start_idx:
# 	# 			return False
			
# 	# 	return True
	
# 	def sample_xs(self, n_points, b_size):

# 		# xs_b = None
# 		# unique_found = False
# 		# while(not unique_found):
# 		if self.xs_b is not None:
# 			return self.xs_b
# 		else:
# 			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
# 			# unique_found = self.check_unique(xs_b)
# 		return xs_b
		
# 	def evaluate(self, xs_b):
# 		# if self.ys_b is not None:
# 		# 	return self.ys_b
# 		# else:
# 		ys_b = torch.randint(0, 2, (xs_b.shape[0], self.start_idx), dtype= torch.float)*2-1 # bs x start_idx

# 		xs_norm = torch.norm(xs_b, dim=2, keepdim=True)
# 		xs_normalized = xs_b / xs_norm
# 		xs_T = torch.transpose(xs_normalized, 1, 2) # bs x n_dims x n_points
# 		sim_mx = torch.matmul(xs_normalized, xs_T) # bs x n_points x n_points

# 		for pt in range(1, self.start_idx): # across initial points
# 			for batch in range(xs_b.shape[0]): # across batch
# 				similarities = sim_mx[batch][pt][:pt] # consider similarities with tensors occuring before
# 				similarities = torch.round(similarities, decimals=7)
# 				selected_idx = torch.argmax(similarities)
# 				if similarities[selected_idx].item() > 0.2**self.n_dims9:
# 					# if ys_b[batch][selected_idx] != ys_b[batch][pt]:
# 					# 	pdb.set_trace()
# 					ys_b[batch][pt] = ys_b[batch][selected_idx]

# 		for pt in range(self.start_idx, xs_b.shape[1]): # across points
# 			y_vals = []
# 			for batch in range(xs_b.shape[0]): # across batch
# 				similarities = sim_mx[batch][pt][:pt] # consider similarities with tensors occuring before
# 				try:
# 					similarities = torch.round(similarities, decimals=7)
# 					selected_idx = torch.argmax(similarities)
# 				except:
# 					pdb.set_trace()
# 				y_vals.append(ys_b[batch][selected_idx].item())
# 			y_col = torch.tensor(y_vals).unsqueeze(1)
# 			ys_b = torch.cat((ys_b, y_col), dim=1)

# 		return ys_b

# 	# @staticmethod
# 	# def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
# 	# 	# w_b shape: (num_tasks, n_dims, 1)
# 	# 	start_time = time()

# 	# 	xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1
# 	# 	# ys_b = self.evaluate(xs_b)

# 	# 	end_time = time()
# 	# 	print('Time to generate pool dict: {:.2f} mins {:.2f} secs'.format((end_time-start_time)//60, (end_time-start_time)%60))

# 	# 	return {"xs": xs_b}

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy



# class SparseThreshold(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(SparseThreshold, self).__init__(n_dims, batch_size, pool_dict, seeds)

# 		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
# 		thres_bound = 3
# 		self.kw = torch.randint(-thres_bound, thres_bound, (self.b_size, 1),  dtype= torch.float) + 0.5
		
	
# 	def sample_xs(self, n_points, b_size):
# 		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

# 		return xs_b

		
# 	def evaluate(self, xs_b):
# 		w_b = self.w_b.to(xs_b.device)
# 		ys_b = (xs_b @ w_b).squeeze() - self.kw
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy






# class IntHalfspace(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(IntHalfspace, self).__init__(n_dims, batch_size, pool_dict, seeds)
# 		bound = 3
# 		self.w_b = torch.randint(-bound, bound+1, (self.b_size, self.n_dims, 1),  dtype= torch.float)

# 	def sample_xs(self, n_points, b_size):
# 		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
# 		return xs_b

# 	def evaluate(self, xs_b):
# 		w_b = self.w_b.to(xs_b.device)
# 		ys_b = (xs_b @ w_b).squeeze() - 0.5
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy





# class Majority(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(Majority, self).__init__(n_dims, batch_size, pool_dict, seeds)

# 		if pool_dict is None:
# 			self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)
# 			self.xs_b = None
# 		else:
# 			assert 'w' in pool_dict
# 			indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
# 			self.w_b = pool_dict["w"][indices]
# 			self.xs_b = pool_dict["xs"][indices]

	
# 	def sample_xs(self, n_points, b_size):
# 		if self.xs_b is not None:
# 			# Using pre-generated xs
# 			return self.xs_b
# 		else:
# 			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

# 			return xs_b


# 	# @staticmethod
# 	# def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
# 	# 	# w_b shape: (num_tasks, n_dims, 1)

# 	# 	w_b = torch.tensor(np.random.choice([0, 1], size=(num_tasks, n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)

# 	# 	xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1


# 	# 	return {"w": w_b, "xs": xs_b}

# 	def evaluate(self, xs_b):
# 		w_b = self.w_b.to(xs_b.device)
# 		ys_b = (xs_b @ w_b).squeeze() - 0.5
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy



# # Three DNF Task named DNF for simplicity. Complete DNF is hard to learn complexity-wise, so we use a 3-term DNF
# class DNF(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(DNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
# 		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
# 		self.w_b = [torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.8, 0.1, 0.1]), dtype=torch.float) for i in range(3)] # Create 3 clauses
# 		self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
	
# 	def sample_xs(self, n_points, b_size):
# 		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

# 		# Manipulate the input to create positive examples to make a more balanced dataset
# 		for b in range(b_size):
# 			cid = np.random.choice([0, 1, 2])        # Choose a clause
# 			wb, k = self.w_b[cid][b], self.kw[cid][b]
# 			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
# 			for i in range(n_points):
# 				if np.random.choice([0, 1], p=[0.65, 0.35]):
# 					xs_b[b, i, pidx] = +1.0
# 					xs_b[b, i, nidx] = -1.0
# 					assert (xs_b[b, i, :] @ wb).squeeze() >= k

# 		return xs_b

		
# 	def evaluate(self, xs_b):
# 		w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
# 		ys_bs = [(xs_b @ w_bs[i]).squeeze() - self.kw[i] for i in range(3)]
# 		ys_bs = [ys_bs[i].sign() for i in range(3)]
# 		# Combine stack three tensors into one
# 		ys_b = torch.stack(ys_bs, dim=2).max(dim=2)[0]  # 0th Index is the value, 1st index has indices
		
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy





# # Three DNF Task named DNF for simplicity. Complete DNF is hard to learn complexity-wise, so we use a 3-term DNF
# class TeachDNF(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(TeachDNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
# 		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
# 		self.w_b = [torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.8, 0.2]), dtype=torch.float) for i in range(3)] # Create 3 clauses
# 		self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
	
# 	def sample_xs(self, n_points, b_size):
# 		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

# 		# Manipulate the input to create positive examples to make a more balanced dataset
# 		for b in range(b_size):
# 			cid = np.random.choice([0, 1, 2])        # Choose a clause
# 			wb, k = self.w_b[cid][b], self.kw[cid][b]
# 			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 			for i in range(n_points):
# 				if np.random.choice([0, 1], p=[0.65, 0.35]):
# 					xs_b[b, i, pidx] = +1.0
# 					assert (xs_b[b, i, :] @ wb).squeeze() >= k
		
# 		# Adding teaching sequence in the beginning of samples
# 		for b in range(b_size):
# 			wb_f = [self.w_b[i][b].squeeze() for i in range(3)]
# 			tidxs = []
# 			for wb in wb_f:
# 				pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 				tidxs.append(pidx)
			
# 			prev_ex_len = 0
# 			for k in range(len(wb_f)):
				
# 				wb = wb_f[k]
# 				tidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 				ex = len(tidx) + 1
# 				new_ex = wb.repeat(ex, 1)
				
# 				for i in range(len(tidx)):
# 					cx = i+1
# 					try:
# 						new_ex[cx, tidx[i]] = 0
# 					except:
# 						pdb.set_trace()
				
# 				new_ex = new_ex * 2 -1

# 				xs_b[b, prev_ex_len: prev_ex_len + ex, :] = new_ex
# 				prev_ex_len += ex
# 				# xs_b[b, ex_lens[k-1]:ex_lens[k-1]+ex, :] = new_ex

		

# 		return xs_b

		
# 	def evaluate(self, xs_b):
# 		w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
# 		ys_bs = [(xs_b @ w_bs[i]).squeeze() - self.kw[i] for i in range(3)]
# 		ys_bs = [ys_bs[i].sign() for i in range(3)]
# 		# Combine stack three tensors into one
# 		ys_b = torch.stack(ys_bs, dim=2).max(dim=2)[0]  # 0th Index is the value, 1st index has indices
		
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy








# class CNF(Task):
# 	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
# 		super(CNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
# 		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
# 		self.w_b = [torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.80, 0.1, 0.1]), dtype=torch.float) for i in range(3)] # Create 3 clauses
# 		self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
	
# 	def sample_xs(self, n_points, b_size):
# 		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		
# 		# Manipulate the input to create negative examples to make a more balanced dataset
# 		for b in range(b_size):
# 			cid = np.random.choice([0, 1, 2])        # Choose a clause
# 			wb, k = self.w_b[cid][b], self.kw[cid][b]
# 			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
# 			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
# 			for i in range(n_points):
# 				if np.random.choice([0, 1], p=[0.7, 0.3]):
# 					xs_b[b, i, pidx] = -1.0
# 					xs_b[b, i, nidx] = +1.0
# 					assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

# 		return xs_b

		
# 	def evaluate(self, xs_b):
# 		w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
# 		ys_bs = [(xs_b @ w_bs[i]).squeeze() + self.kw[i] for i in range(3)]
# 		ys_bs = [ys_bs[i].sign() for i in range(3)]
# 		# Combine stack three tensors into one
# 		ys_b = torch.stack(ys_bs, dim=2).min(dim=2)[0]  # 0th Index is the value, 1st index has indices
		
# 		return ys_b.sign()

# 	@staticmethod
# 	def get_metric():
# 		return accuracy

# 	@staticmethod
# 	def get_training_metric():
# 		return cross_entropy

