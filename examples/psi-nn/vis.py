import pickle

import matplotlib.pyplot as plt
from args_parser import get_args

args = get_args()


def loading(name):
    if name == "resnet18":
        path = "/storage/1002tjt/MC-Attack/cifar10_resnet18/importance_records.pkl"
        # path = "/home/1002tjt/models/cifar10_resnet18/importance_records.pkl"
    elif name == "vgg13":
        path = "/storage/1002tjt/MC-Attack/cifar100_vgg13/importance_records.pkl"
        # path = "/home/1002tjt/models/cifar100_vgg13/importance_records.pkl"
    else:
        raise ValueError()
    with open(path, "rb") as f:
        importance_records = pickle.load(f)
    return importance_records


def vis_resnet18(importance_records, cummulated=True, linewidth=2):
    fig = plt.figure()
    # fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1, 1, 1)
    x_epochs = list(range(50))

    if cummulated:
        from copy import deepcopy

        cummulated_importance = {name: 0 for name in importance_records[0]}
        cummulated_records = []
        for record in importance_records:
            for name, value in record.items():
                cummulated_importance[name] += value
            cummulated_records.append(deepcopy(cummulated_importance))
        importance_records = cummulated_records

    conv1 = [importance["conv1.weight"] for importance in importance_records]
    layer1_0_conv1 = [
        importance["layer1.0.conv1.weight"] for importance in importance_records
    ]
    layer1_0_conv2 = [
        importance["layer1.0.conv2.weight"] for importance in importance_records
    ]
    layer1_1_conv1 = [
        importance["layer1.1.conv1.weight"] for importance in importance_records
    ]
    layer1_1_conv2 = [
        importance["layer1.1.conv2.weight"] for importance in importance_records
    ]
    layer2_0_conv1 = [
        importance["layer2.0.conv1.weight"] for importance in importance_records
    ]
    layer2_0_conv2 = [
        importance["layer2.0.conv2.weight"] for importance in importance_records
    ]
    layer2_1_conv1 = [
        importance["layer2.1.conv1.weight"] for importance in importance_records
    ]
    layer2_1_conv2 = [
        importance["layer2.1.conv2.weight"] for importance in importance_records
    ]
    layer3_0_conv1 = [
        importance["layer3.0.conv1.weight"] for importance in importance_records
    ]

    ax.plot(x_epochs, conv1, label="layer1", linewidth=linewidth)
    ax.plot(x_epochs, layer1_0_conv1, label="layer2", linewidth=linewidth)
    ax.plot(x_epochs, layer1_0_conv2, label="layer3", linewidth=linewidth)
    ax.plot(x_epochs, layer1_1_conv1, label="layer4", linewidth=linewidth)
    ax.plot(x_epochs, layer1_1_conv2, label="layer5", linewidth=linewidth)
    ax.plot(x_epochs, layer2_0_conv1, label="layer6", linewidth=linewidth)
    ax.plot(x_epochs, layer2_0_conv2, label="layer7", linewidth=linewidth)
    ax.plot(x_epochs, layer2_1_conv1, label="layer8", linewidth=linewidth)
    ax.plot(x_epochs, layer2_1_conv2, label="layer9", linewidth=linewidth)
    ax.plot(x_epochs, layer3_0_conv1, label="layer10", linewidth=linewidth)

    ax.grid(True, linestyle="--")
    # ax.set_title("Layer gradient")
    ax.set_ylabel("Accumu. Grad. Norm", labelpad=5, loc="center", fontsize=26)
    ax.set_xlabel("Epoch", labelpad=5, loc="center", fontsize=26)
    plt.yticks(fontsize=26)
    plt.xticks(fontsize=26)
    plt.ylim(0, 20)
    plt.legend(loc="best", ncol=2, fontsize=16)
    plt.savefig("figures/grad_norm_resnet18.pdf", bbox_inches="tight")
    plt.close()


def vis_vgg13(importance_records, cummulated=True, linewidth=2):
    fig = plt.figure()
    # fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1, 1, 1)
    x_epochs = list(range(50))
    if cummulated:
        from copy import deepcopy

        cummulated_importance = {name: 0 for name in importance_records[0]}
        cummulated_records = []
        for record in importance_records:
            for name, value in record.items():
                cummulated_importance[name] += value
            cummulated_records.append(deepcopy(cummulated_importance))
        importance_records = cummulated_records

    layer1 = [
        importance["feature_extractor.0.weight"] for importance in importance_records
    ]
    layer2 = [
        importance["feature_extractor.3.weight"] for importance in importance_records
    ]
    layer3 = [
        importance["feature_extractor.7.weight"] for importance in importance_records
    ]
    layer4 = [
        importance["feature_extractor.10.weight"] for importance in importance_records
    ]
    layer5 = [
        importance["feature_extractor.14.weight"] for importance in importance_records
    ]
    layer6 = [
        importance["feature_extractor.17.weight"] for importance in importance_records
    ]
    layer7 = [
        importance["feature_extractor.21.weight"] for importance in importance_records
    ]
    layer8 = [
        importance["feature_extractor.24.weight"] for importance in importance_records
    ]
    layer9 = [
        importance["feature_extractor.28.weight"] for importance in importance_records
    ]
    layer10 = [
        importance["feature_extractor.31.weight"] for importance in importance_records
    ]

    ax.plot(x_epochs, layer1, label="layer1", linewidth=linewidth)
    ax.plot(x_epochs, layer2, label="layer2", linewidth=linewidth)
    ax.plot(x_epochs, layer3, label="layer3", linewidth=linewidth)
    ax.plot(x_epochs, layer4, label="layer4", linewidth=linewidth)
    ax.plot(x_epochs, layer5, label="layer5", linewidth=linewidth)
    ax.plot(x_epochs, layer6, label="layer6", linewidth=linewidth)
    ax.plot(x_epochs, layer7, label="layer7", linewidth=linewidth)
    ax.plot(x_epochs, layer8, label="layer8", linewidth=linewidth)
    ax.plot(x_epochs, layer9, label="layer9", linewidth=linewidth)
    ax.plot(x_epochs, layer10, label="layer10", linewidth=linewidth)

    ax.grid(True, linestyle="--")
    # ax.set_title("Layer gradient")
    ax.set_ylabel("Accumu. Grad. Norm", labelpad=5, loc="center", fontsize=26)
    ax.set_xlabel("Epoch", labelpad=5, loc="center", fontsize=26)
    plt.yticks(fontsize=26)
    plt.xticks(fontsize=26)
    # plt.ylim(0, 40)
    plt.ylim(0, 35)
    plt.legend(loc="best", ncol=2, fontsize=16)
    plt.savefig("figures/grad_norm_vgg13.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    vis_resnet18(loading(name="resnet18"), cummulated=True, linewidth=3)
    vis_vgg13(loading(name="vgg13"), cummulated=True, linewidth=3)
