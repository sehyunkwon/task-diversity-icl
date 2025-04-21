import torch
from src.models import build_model

def setup_device_and_model(args):
    if isinstance(args.gpu, list):
        device = torch.device(f"cuda:{args.gpu[0]}")
        available_gpus = ','.join(map(str, args.gpu))
        torch.cuda.set_device(device)
        print(f"Using GPUs: {available_gpus}")
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)

    model = build_model(args)

    if isinstance(args.gpu, list) and len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)

    model.to(device)
    model.device = device
    model.train()

    print("device:", device)
    return device, model
