import os
import pathlib

import torch


class TorchModelIO:
    def __init__(self):
        pass

    @staticmethod
    def save(model, model_dir, model_name, epoch=None, optimizer=None):
        if not os.path.exists(model_dir):
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        checkpoint = {"model": model, "epoch": epoch, "optimizer": optimizer}

        with open(os.path.join(model_dir, model_name), "wb") as f:
            torch.save(checkpoint, f)

    @staticmethod
    def load(model_dir, model_name):
        if not os.path.exists(model_dir):
            raise Exception(f"{model_dir} not found.")

        with open(os.path.join(model_dir, model_name), "rb") as f:
            checkpoint = torch.load(f)

        return checkpoint

    """
    @staticmethod
    def save(model, path, name, epoch=None, optimizer=None):
        if not os.path.exists(path):
            os.mkdir(path)

        if epoch is not None and optimizer is not None:
            if type(model) == list:  # model is a list composed of many submodels
                checkpoint = {
                    "model": [sub_model.state_dict() for sub_model in model],
                    "epoch": epoch,
                    "optimizer": [sub_optim.state_dict() for sub_optim in optimizer],
                }
            else:  # single model
                checkpoint = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                }
        else:
            if type(model) == list:  # model is a list composed of many submodels
                checkpoint = {"model": [sub_model.state_dict() for sub_model in model]}
            else:  # single model
                checkpoint = {"model": model.state_dict()}

        torch.save(checkpoint, os.path.join(path, name))

    @staticmethod
    def load(model_arch, path, name, optimizer_arch=None):
        if not os.path.exists(path):
            raise Exception(f"{path} not found.")

        checkpoint = torch.load(os.path.join(path, name))

        # One detail that should be noted when loading a PyTorch model is that:
        # you should directly do net.load_state_dict(torch.load(PATH)),
        # rather than net = net.load_state_dict(torch.load(PATH))
        # ref: https://stackoverflow.com/a/59442773/8418540
        if "epoch" in checkpoint and "optimizer" in checkpoint:
            epoch = checkpoint["epoch"]
            if type(checkpoint["model"]) == list:
                if optimizer_arch is None:
                    for sub_model, sub_model_arch in zip(
                        checkpoint["model"], model_arch
                    ):
                        sub_model_arch.load_state_dict(sub_model)
                else:
                    for sub_model, sub_model_arch, sub_optim, sub_optim_arch in zip(
                        checkpoint["model"],
                        model_arch,
                        checkpoint["optimizer"],
                        optimizer_arch,
                    ):
                        sub_model_arch.load_state_dict(sub_model)
                        sub_optim_arch.load_state_dict(sub_optim)
            else:
                model_arch.load_state_dict(checkpoint["model"])
                if optimizer_arch is not None:
                    optimizer_arch.load_state_dict(checkpoint["optimizer"])
        else:
            epoch = None
            if type(checkpoint["model"]) == list:
                optimizer_arch = [None] * len(checkpoint["model"])
                for sub_model, sub_model_arch in zip(checkpoint["model"], model_arch):
                    sub_model_arch.load_state_dict(sub_model)
            else:
                optimizer_arch = None
                model_arch.load_state_dict(checkpoint["model"])

        return model_arch, epoch, optimizer_arch
    """
