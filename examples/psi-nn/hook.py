import time
from functools import partial

time_dict = {}


def _take_time_pre(layer_name, module, inputs):
    time_dict[layer_name] = time.time()


def _take_time(layer_name, module, inputs, ouputs):
    time_dict[layer_name] = time.time() - time_dict[layer_name]


def register_hook(model):
    for layer in model.children():
        layer.register_forward_pre_hook(partial(_take_time_pre, layer))
        layer.register_forward_hook(partial(_take_time, layer))

    return time_dict
