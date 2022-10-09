import numpy as np
import torch


def cal_sensitivity(lr, clip):
    return 2 * lr * clip


def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)


def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)


def add_dp(model, lr, dp_clip, dp_mechanism, dp_epsilon, dp_delta, device):
    sensitivity = cal_sensitivity(lr, dp_clip)
    if dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in model.named_parameters():
                noise = Laplace(epsilon=dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise
    elif dp_mechanism == 'Gaussian':
        with torch.no_grad():
            for k, v in model.named_parameters():
                noise = Gaussian_Simple(epsilon=dp_epsilon, delta=dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise
    return model


def gradient_clip(model, dp_mechanism, dp_clip):
    if dp_mechanism == 'Laplace':
        for k, v in model.named_parameters():
            v.grad /= max(1, v.grad.norm(1) / dp_clip)
    elif dp_mechanism == 'Gaussian':
        for k, v in model.named_parameters():
            v.grad /= max(1, v.grad.norm(2) / dp_clip)
    return model

