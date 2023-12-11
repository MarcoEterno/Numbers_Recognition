import os.path

import torch


checkpoints_path = os.path.join(os.path.dirname(os.getcwd()), "checkpoints")
logs_path = os.path.join(os.path.dirname(os.getcwd()), "logs", "fit")
data_path = os.path.join(os.path.dirname(os.getcwd()), "data")
#TODO: implement
training_parallelism = True  # if set to True, the training loop will be parallelized using torch.nn.DataParallel
fast_training = False if(torch.has_mps or torch.cuda.is_available()) else True  # if set to True, the neural network will lose a layer of depth, but will train faster

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

if __name__ == '__main__':
    print(checkpoints_path)