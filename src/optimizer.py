import torch

def build_optimizer(model, args):
    if args.precision == 'half':
        if args.optimizer == 'adam':
            return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), eps=1e-4, lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        else:
            raise ValueError('Please choose sgd or adam')
    else:
        if args.weight_decay:
            return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        else:
            return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
