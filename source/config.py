import torch


def get_system_device():
    if torch.has_mps:
        print(f"Using mps device")
        return 'mps'
    elif torch.cuda.is_available():
        print(f"Using cuda device")
        return 'cuda'
    else:
        print(f"Using cpu device")
        return 'cpu'
