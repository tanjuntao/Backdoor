import os

import torch


class TorchModelIO:
    def __init__(self):
        pass

    @staticmethod
    def save(model, path, name, epoch=None, optimizer=None):
        if not os.path.exists(path):
            os.mkdir(path)

        if epoch is not None and optimizer is not None:
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
        else:
            checkpoint = {
                'model': model.state_dict()
            }

        torch.save(checkpoint, os.path.join(path, name))

    @staticmethod
    def load(path, name, model_arch=None):
        if not os.path.exists(path):
            raise Exception(f"{path} not found.")
        if model_arch is None:
            raise ValueError('model architecture should be provided')

        checkpoint = torch.load(os.path.join(path, name))
        model = model_arch.load_state_dict(checkpoint['model'])
        if 'epoch' not in checkpoint and 'optimizer' not in checkpoint:
            epoch = None
            optimizer = None
        else:
            epoch = checkpoint['epoch']
            optimizer = checkpoint['optimizer']

        return model, epoch, optimizer
