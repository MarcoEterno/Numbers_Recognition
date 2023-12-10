import torch


def get_system_device(print_info=False):
    if torch.has_mps:
        if print_info:
            print(f"Using mps device")
        return 'mps'
    elif torch.cuda.is_available():
        if print_info:
            print(f"Using cuda device")
        return 'cuda'
    else:
        if print_info:
            print(f"Using cpu device")
        return 'cpu'
