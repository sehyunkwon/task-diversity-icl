import math
import random
import torch
import numpy as np
import pdb
import time

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    '''
    ys_pred: {-inf, inf}
    ys: {-1, 1}
    '''
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, n_points, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        self.data = 'gaussian'
        self.n_points = n_points
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    def NCL_optimal_loss(self):
        # output is scalar
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(n_dims,**kwargs):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "relu_regression": ReLURegression,
        "leaky_relu_regression": LeakyReLURegression,
        "logistic_regression": LogisticRegression,
        "sine_regression": SineRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "decision_tree": DecisionTree,
        "gaussian_retrieval": GaussianRetrieval,
        "optimal_no_context_linear_regression": OptimalNoContextLinearRegression,
        "optimal_no_context_quadratic_regression": OptimalNoContextQuadraticRegression,
    }
    if kwargs['task'] in task_names_to_classes:
        task_cls = task_names_to_classes[kwargs['task']]
        return lambda **args: task_cls(n_dims, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        self.n_dims = n_dims
        super(LinearRegression, self).__init__(self.n_dims, batch_size, n_points)
        
        self.scale = kwargs['scale']

        self.mu = kwargs['mu']
        # Non-zero mean gaussian
        self.w_b = torch.randn(self.b_size , self.n_dims , 1) + self.mu
        
        self.ncl_w_b = torch.zeros(self.b_size , self.n_dims , 1) + self.mu

        self.ncl_opt_loss = self.NCL_optimal_loss()

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    def ncl_evaluate(self, xs_b):
        w_b =  self.ncl_w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    def sample_wb(self, b_size):
        wb = torch.randn(b_size , self.n_dims , 1) + self.mu
        return wb
    
    def NCL_optimal_loss(self):
        # NCL optimum function is  <\mu,x>
        return self.n_dims

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class LeakyReLURegression(Task):
    ncl_opt_loss = None

    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        self.n_dims = n_dims
        super(LeakyReLURegression, self).__init__(self.n_dims, batch_size, n_points)
        
        self.scale = kwargs['scale']

        self.mu = kwargs['mu']
        # Non-zero mean gaussian
        self.w_b = torch.randn(self.b_size , self.n_dims , 1) + self.mu

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = torch.nn.functional.leaky_relu(self.scale * (xs_b @ w_b)[:, :, 0], negative_slope=0.5)
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = torch.nn.functional.leaky_relu(self.scale * (xs_b @ w_b)[:, :, 0], negative_slope=0.5)
        return ys_b
    
    def sample_wb(self, b_size):
        wb = torch.randn(b_size , self.n_dims , 1) + self.mu
        return wb
    
    def NCL_optimal_loss(self, individual_ncl_loss):
        # Initialization
        if LeakyReLURegression.ncl_opt_loss is None:
            LeakyReLURegression.ncl_opt_loss = individual_ncl_loss

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class LogisticRegression(Task):
    ncl_opt_loss = None

    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        self.n_dims = n_dims
        super(LogisticRegression, self).__init__(self.n_dims, batch_size, n_points)
        
        self.scale = kwargs['scale']

        self.mu = kwargs['mu']
        # Non-zero mean gaussian
        self.w_b = torch.randn(self.b_size , self.n_dims , 1) + self.mu

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = torch.nn.functional.sigmoid(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = torch.nn.functional.sigmoid(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def sample_wb(self, b_size):
        wb = torch.randn(b_size , self.n_dims , 1) + self.mu
        return wb
    
    def NCL_optimal_loss(self, individual_ncl_loss):
        # Initialization
        if LogisticRegression.ncl_opt_loss is None:
            LogisticRegression.ncl_opt_loss = individual_ncl_loss

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class SineRegression(Task):
    ncl_opt_loss = None

    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        self.n_dims = n_dims
        super(SineRegression, self).__init__(self.n_dims, batch_size, n_points)
        
        self.scale = kwargs['scale']

        self.mu = kwargs['mu']
        # Non-zero mean gaussian
        self.w_b = torch.randn(self.b_size , self.n_dims , 1) + self.mu

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = torch.sin(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = torch.sin(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def sample_wb(self, b_size):
        wb = torch.randn(b_size , self.n_dims , 1) + self.mu
        return wb
    
    def NCL_optimal_loss(self, individual_ncl_loss):
        # Initialization
        if SineRegression.ncl_opt_loss is None:
            SineRegression.ncl_opt_loss = individual_ncl_loss

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    

class ReLURegression(Task):
    ncl_opt_loss = None

    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        self.n_dims = n_dims
        super(ReLURegression, self).__init__(self.n_dims, batch_size, n_points)
        
        self.scale = kwargs['scale']

        self.mu = kwargs['mu']
        # Non-zero mean gaussian
        self.w_b = torch.randn(self.b_size , self.n_dims , 1) + self.mu

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = torch.nn.functional.relu(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = torch.nn.functional.relu(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def sample_wb(self, b_size):
        wb = torch.randn(b_size , self.n_dims , 1) + self.mu
        return wb
    
    def NCL_optimal_loss(self, individual_ncl_loss):
        # Initialization
        if ReLURegression.ncl_opt_loss is None:
            ReLURegression.ncl_opt_loss = individual_ncl_loss

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class ReLURegression(Task):
    ncl_opt_loss = None

    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        self.n_dims = n_dims
        super(ReLURegression, self).__init__(self.n_dims, batch_size, n_points)
        
        self.scale = kwargs['scale']

        self.mu = kwargs['mu']
        # Non-zero mean gaussian
        self.w_b = torch.randn(self.b_size , self.n_dims , 1) + self.mu

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = torch.nn.functional.relu(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = torch.nn.functional.relu(self.scale * (xs_b @ w_b)[:, :, 0])
        return ys_b
    
    def sample_wb(self, b_size):
        wb = torch.randn(b_size , self.n_dims , 1) + self.mu
        return wb
    
    def NCL_optimal_loss(self, individual_ncl_loss):
        # Initialization
        if ReLURegression.ncl_opt_loss is None:
            ReLURegression.ncl_opt_loss = individual_ncl_loss

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    

class OptimalNoContextLinearRegression(Task):
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        self.n_dims = n_dims
        super(OptimalNoContextLinearRegression, self).__init__(self.n_dims, batch_size, n_points)
        
        self.scale = kwargs['scale']

        self.mu = kwargs['mu']
        # Non-zero mean gaussian
        self.w_b = torch.zeros(self.b_size , self.n_dims , 1) + self.mu

        self.ncl_opt_loss = self.NCL_optimal_loss()

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    def sample_wb(self, b_size):
        wb = torch.zeros(b_size , self.n_dims , 1) + self.mu
        return wb
    
    def NCL_optimal_loss(self):
        # NCL optimum function is  <\mu,x>
        return 1

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class QuadraticRegression(Task):
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        super(QuadraticRegression , self).__init__(n_dims, batch_size, n_points)
        """scale: a constant by which to scale the randomly sampled weights."""
        self.scale = kwargs['scale']

        self.n_dims = n_dims
        self.mu =  kwargs['mu']

        # True parameter for quadratic regression, nonzero mean
        self.quad_w = 1/math.sqrt(self.n_dims) * ( torch.randn(self.b_size, self.n_dims , self.n_dims) + self.mu )
        self.ncl_quad_w = 1/math.sqrt(self.n_dims) * ( torch.zeros(self.b_size, self.n_dims , self.n_dims) + self.mu )
        # True parameter for quadratic regression, zero mean
        # self.quad_w = 1/math.sqrt(self.n_dims) * torch.randn(self.b_size, self.n_dims , self.n_dims)
        self.ncl_opt_loss = self.NCL_optimal_loss()

    def evaluate(self, xs_b):
        quad_w = self.quad_w.to(xs_b.device)
        tmp = (xs_b @ quad_w)
        prod = tmp * xs_b  #element-wise product
        ys_b = torch.sum(prod, dim = -1 , keepdim = False )
        return ys_b

    def ncl_evaluate(self, xs_b):
        quad_w = self.ncl_quad_w.to(xs_b.device)
        tmp = (xs_b @ quad_w)
        prod = tmp * xs_b  #element-wise product
        ys_b = torch.sum(prod, dim = -1 , keepdim = False )
        return ys_b
    
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        tmp = (xs_b @ w_b)
        prod = tmp * xs_b  #element-wise product
        ys_b = torch.sum(prod, dim = -1 , keepdim = False )
        return ys_b
    
    def sample_wb(self, b_size):
        wb = 1/math.sqrt(self.n_dims) * ( torch.randn(b_size, self.n_dims , self.n_dims) + self.mu )
        return wb

    def NCL_optimal_loss(self):
        # NCL optimum function is x^T torch.ones x.
        return self.n_dims + 2

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class OptimalNoContextQuadraticRegression(Task):
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        super(OptimalNoContextQuadraticRegression , self).__init__(n_dims, batch_size, n_points)
        """scale: a constant by which to scale the randomly sampled weights."""
        self.scale = kwargs['scale']

        self.n_dims = n_dims
        self.mu =  kwargs['mu']

        # True parameter for quadratic regression, nonzero mean
        self.quad_w = 1/math.sqrt(self.n_dims) * ( torch.zeros(self.b_size, self.n_dims , self.n_dims) + self.mu )

        # True parameter for quadratic regression, zero mean
        # self.quad_w = 1/math.sqrt(self.n_dims) * torch.randn(self.b_size, self.n_dims , self.n_dims)
        self.ncl_opt_loss = self.NCL_optimal_loss()

    def evaluate(self, xs_b):
        quad_w = self.quad_w.to(xs_b.device)
        tmp = (xs_b @ quad_w)
        prod = tmp * xs_b  #element-wise product
        ys_b = torch.sum(prod, dim = -1 , keepdim = False )
        
        return ys_b
    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        tmp = (xs_b @ w_b)
        prod = tmp * xs_b  #element-wise product
        ys_b = torch.sum(prod, dim = -1 , keepdim = False )
        return ys_b
    
    def sample_wb(self, b_size):
        wb = 1/math.sqrt(self.n_dims) * ( torch.zeros(b_size, self.n_dims , self.n_dims) + self.mu )
        return wb

    def NCL_optimal_loss(self):
        # NCL optimum function is x^T torch.ones x.
        return self.n_dims + 2

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class Relu2nnRegression(Task):
    ncl_opt_loss = None
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        # hidden layer size = 100 for default. May be varied.
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, n_points)
        self.scale = kwargs['scale']
        self.hidden_layer_size = kwargs.get('hidden_layer_size', 100)
        self.mu1 = kwargs['mu1']
        self.mu2 = kwargs['mu2']

        # non-zero mean gaussian
        self.W1 = torch.randn(self.b_size, self.n_dims, self.hidden_layer_size) + self.mu1
        self.W2 = torch.randn(self.b_size, self.hidden_layer_size, 1) + 0.01 * self.mu2

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn

        return ys_b_nn
    
    def evaluate_by_given_wb(self, xs_b, w_b:tuple):
        W1, W2 = w_b
        W1 = W1.to(xs_b.device)
        W2 = W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn

        return ys_b_nn
    
    def NCL_optimal_loss(self, individual_ncl_loss):
        # Initialization
        if Relu2nnRegression.ncl_opt_loss is None:
            Relu2nnRegression.ncl_opt_loss = individual_ncl_loss
    
    def sample_wb(self, b_size):
        # non-zero mean gaussian
        W1 = torch.randn(b_size, self.n_dims, self.hidden_layer_size) + self.mu1
        W2 = torch.randn(b_size, self.hidden_layer_size, 1) + 0.01 * self.mu2
        return (W1, W2)

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(Task):
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        # Default value of sparsity: is 3. May be varied
        super(SparseLinearRegression, self).__init__(n_dims, batch_size, n_points)
        self.scale = kwargs['scale']
        self.mu = kwargs['mu']
        self.n_dims = n_dims
        self.sparsity = kwargs['sparsity']
        self.valid_coords = kwargs['valid_coords']
        assert self.valid_coords <= n_dims
        
         # Non-zero mean gaussian
        self.w_b = torch.randn(self.b_size , self.n_dims , 1) + self.mu
        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            perm = torch.randperm(self.valid_coords)
            mask[perm[:self.sparsity]] = False
            w[mask] = 0
        
        self.ncl_opt_loss = self.NCL_optimal_loss()

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    def evaluate_by_given_wb(self, xs_b, w_b):
        w_b = w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    def sample_wb(self, b_size):
        w_b = torch.randn(b_size , self.n_dims , 1) + self.mu

        for i, w in enumerate(w_b):
            mask = torch.ones(self.n_dims).bool()
            perm = torch.randperm(self.valid_coords)
            mask[perm[:self.sparsity]] = False
            w[mask] = 0
            
        return w_b

    def NCL_optimal_loss(self):
        # The NCL optimum function is (sparsity / n_dims) X  <\mu,x>
        return self.sparsity * (self.n_dims + (self.n_dims - self.sparsity) * (self.mu ** 2) ) / self.n_dims

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        # Default value of depth is 4. May be varied.

        super(DecisionTree, self).__init__(n_dims, batch_size, n_points)
        self.depth = kwargs['depth']  ## depth = 4
        self.ncl_opt_loss = self.NCL_optimal_loss()
        self.n_dims = n_dims

        # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
        # dt_tensor stores the coordinate used at each node of the decision tree.
        # Only indices corresponding to non-leaf nodes are relevant
        self.dt_tensor = torch.randint( low=0, high=n_dims, size=(batch_size, 2 ** (self.depth + 1) - 1))

        # Target value at the leaf nodes.
        # Only indices corresponding to leaf nodes are relevant.
        # non-zero mean gaussian
        self.target_tensor = torch.randn(self.dt_tensor.shape) + torch.ones(self.dt_tensor.shape)


    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    def evaluate_by_given_wb(self, xs_b, w_b):
        dt_tensor, target_tensor = w_b
        xs_b = xs_b.expand(dt_tensor.shape[0], -1, -1)
        dt_tensor = dt_tensor.to(xs_b.device)
        target_tensor = target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b
    
    def sample_wb(self, b_size):
        dt_tensor = torch.randint( low=0, high=self.n_dims, size=(b_size, 2 ** (self.depth + 1) - 1))
        target_tensor = torch.randn(dt_tensor.shape) + torch.ones(dt_tensor.shape)
        return (dt_tensor, target_tensor)

    def NCL_optimal_loss(self):
        # The NCL optimum function is \mu (constant function)
        return 1

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

 # Construct a key - value pairs for retreival training

current_time = int(time.time())

def generate_kv(num_keys, n_dim, num_values=4, seed=current_time):
    original_state = torch.get_rng_state()
    torch.manual_seed(seed)
    key = torch.randn(num_keys, n_dim)
    
    value_list = []
    for _ in range(num_values):
        value = 3 * torch.randn(num_keys) 
        value_list.append(value)

    torch.set_rng_state(original_state)  
    return key, value_list

class GaussianRetrieval(Task):
    def __init__(self, n_dims, batch_size, n_points, **kwargs):
        super(GaussianRetrieval, self).__init__(n_dims, batch_size, n_points)
        # self.bias = kwargs['bias']
        # self.scale = kwargs['scale']
        self.n_dims = n_dims
        self.num_keys = kwargs['num_keys']
        self.num_values = kwargs['num_values']

        keys, value_list = generate_kv(self.num_keys, self.n_dims, self.num_values)
        self.keys = keys
        self.value_list = value_list
        
        self.ncl_opt_loss = self.NCL_optimal_loss()
        
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
    
    def NCL_optimal_loss(self):
        # The NCL optimum function is (sparsity / n_dims) X  <\mu,x>
        return 9

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error