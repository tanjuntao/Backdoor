import torch.nn as nn


def layer_masking(model_type, bottom_model, mask_layers, device, dataset):
    if model_type == "resnet18":
        if "conv1.weight" in mask_layers:
            bottom_model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            ).to(device)
        if "layer1.0.conv1.weight" in mask_layers:
            bottom_model.layer1[0].conv1 = nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=False
            ).to(device)
        if "layer1.0.conv2.weight" in mask_layers:
            bottom_model.layer1[0].conv2 = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer1.1.conv1.weight" in mask_layers:
            bottom_model.layer1[1].conv1 = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer1.1.conv2.weight" in mask_layers:
            bottom_model.layer1[1].co1nv2 = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer2.0.conv1.weight" in mask_layers:
            bottom_model.layer2[0].conv1 = nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ).to(device)
        if "layer2.0.conv2.weight" in mask_layers:
            bottom_model.layer2[0].conv2 = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer2.1.conv1.weight" in mask_layers:
            bottom_model.layer2[1].conv1 = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer2.1.conv2.weight" in mask_layers:
            bottom_model.layer2[1].conv2 = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer3.0.conv1.weight" in mask_layers:
            bottom_model.layer3[0].conv1 = nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ).to(device)
        if "layer3.0.conv2.weight" in mask_layers:
            bottom_model.layer3[0].conv2 = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer3.1.conv1.weight" in mask_layers:
            bottom_model.layer3[1].conv1 = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer3.1.conv2.weight" in mask_layers:
            bottom_model.layer3[1].conv2 = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer4.0.conv1.weight" in mask_layers:
            bottom_model.layer4[0].conv1 = nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ).to(device)
        if "layer4.0.conv2.weight" in mask_layers:
            bottom_model.layer4[0].conv2 = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer4.1.conv1.weight" in mask_layers:
            bottom_model.layer4[1].conv1 = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if "layer4.1.conv2.weight" in mask_layers:
            bottom_model.layer4[1].conv2 = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 18 in mask_layers:
            if dataset == "cifar100":
                out_features = 100
            else:
                out_features = 10
            bottom_model.linear = nn.Linear(
                in_features=512, out_features=out_features, bias=True
            ).to(device)

    elif model_type == "vgg13":
        if "feature_extractor.0.weight" in mask_layers:
            bottom_model.feature_extractor[0] = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.3.weight" in mask_layers:
            bottom_model.feature_extractor[3] = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.7.weight" in mask_layers:
            bottom_model.feature_extractor[7] = nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.10.weight" in mask_layers:
            bottom_model.feature_extractor[10] = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.14.weight" in mask_layers:
            bottom_model.feature_extractor[14] = nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.17.weight" in mask_layers:
            bottom_model.feature_extractor[17] = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.21.weight" in mask_layers:
            bottom_model.feature_extractor[21] = nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.24.weight" in mask_layers:
            bottom_model.feature_extractor[24] = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.28.weight" in mask_layers:
            bottom_model.feature_extractor[28] = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if "feature_extractor.31.weight" in mask_layers:
            bottom_model.feature_extractor[31] = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 11 in mask_layers:
            if dataset == "cifar100":
                out_features = 100
            else:
                out_features = 10
            bottom_model.classifier = nn.Linear(
                in_features=512, out_features=out_features, bias=True
            ).to(device)

    elif model_type == "lenet":
        if "conv1.weight" in mask_layers:
            if dataset == "svhn":
                in_channel = 3
            else:
                in_channel = 1
            bottom_model.conv1 = nn.Conv2d(
                in_channel, 6, kernel_size=(5, 5), stride=(1, 1)
            ).to(device)
        if "conv2.weight" in mask_layers:
            bottom_model.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)).to(
                device
            )
        if "fc1.weight" in mask_layers:
            bottom_model.fc1 = nn.Linear(
                in_features=80, out_features=120, bias=True
            ).to(device)
        if "fc2.weight" in mask_layers:
            bottom_model.fc2 = nn.Linear(
                in_features=120, out_features=84, bias=True
            ).to(device)
        if "fc3.weight" in mask_layers:
            bottom_model.fc3 = nn.Linear(in_features=84, out_features=10, bias=True).to(
                device
            )

    elif model_type == "mlp":
        if dataset == "criteo":
            if "sequential.0.weight" in mask_layers:
                bottom_model.sequential[0] = nn.Linear(
                    in_features=18, out_features=15, bias=True
                ).to(device)
            if "sequential.3.weight" in mask_layers:
                bottom_model.sequential[3] = nn.Linear(
                    in_features=15, out_features=10, bias=True
                ).to(device)
            if "sequential.6.weight" in mask_layers:
                bottom_model.sequential[6] = nn.Linear(
                    in_features=10, out_features=10, bias=True
                ).to(device)
        else:
            if "sequential.0.weight" in mask_layers:
                bottom_model.sequential[0] = nn.Linear(
                    in_features=392, out_features=256, bias=True
                ).to(device)
            if "sequential.3.weight" in mask_layers:
                bottom_model.sequential[3] = nn.Linear(
                    in_features=256, out_features=128, bias=True
                ).to(device)
            if "sequential.6.weight" in mask_layers:
                bottom_model.sequential[6] = nn.Linear(
                    in_features=128, out_features=128, bias=True
                ).to(device)
    else:
        raise ValueError(f"{model_type} is not valid model type.")

    return bottom_model
