from torch.optim.lr_scheduler import _LRScheduler
import argparse
from torchvision import transforms as tfs
import os
from src.config import Config
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def load_config(mode=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1

    # test mode
    elif mode == 2:
        config.MODE = 2

    return config

def train_tf(x):
    config = load_config()
    x=x.resize((config.RESIZE,config.RESIZE))
    x=x.convert('RGB')
    im_aug = tfs.Compose([
        tfs.RandomHorizontalFlip(),  # default 0.5
        tfs.RandomCrop(config.CROP),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

def test_tf(x):
    config = load_config()
    x=x.resize((config.RESIZE,config.RESIZE))
    x=x.convert('RGB')
    im_aug = tfs.Compose([
        tfs.CenterCrop(config.CROP),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)           
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
