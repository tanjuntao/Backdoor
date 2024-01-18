import torch.nn as nn


def layer_masking(model_type, bottom_model, mask_layers, device):
    if model_type == "resnet18":
        if 1 in mask_layers:
            bottom_model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            ).to(device)
        if 2 in mask_layers:
            bottom_model.layer1[0].conv1 = nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=False
            ).to(device)
        if 3 in mask_layers:
            bottom_model.layer1[0].conv2 = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 4 in mask_layers:
            bottom_model.layer1[1].conv1 = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 5 in mask_layers:
            bottom_model.layer1[1].conv2 = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 6 in mask_layers:
            bottom_model.layer2[0].conv1 = nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ).to(device)
        if 7 in mask_layers:
            bottom_model.layer2[0].conv2 = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 8 in mask_layers:
            bottom_model.layer2[1].conv1 = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 9 in mask_layers:
            bottom_model.layer2[1].conv2 = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 10 in mask_layers:
            bottom_model.layer3[0].conv1 = nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ).to(device)
        if 11 in mask_layers:
            bottom_model.layer3[0].conv2 = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 12 in mask_layers:
            bottom_model.layer3[1].conv1 = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 13 in mask_layers:
            bottom_model.layer3[1].conv2 = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 14 in mask_layers:
            bottom_model.layer4[0].conv1 = nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ).to(device)
        if 15 in mask_layers:
            bottom_model.layer4[0].conv2 = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 16 in mask_layers:
            bottom_model.layer4[1].conv1 = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 17 in mask_layers:
            bottom_model.layer4[1].conv2 = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ).to(device)
        if 18 in mask_layers:
            bottom_model.linear = nn.Linear(
                in_features=512, out_features=10, bias=True
            ).to(device)

    elif model_type == "vgg13":
        if 1 in mask_layers:
            bottom_model.feature_extractor[0] = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 2 in mask_layers:
            bottom_model.feature_extractor[3] = nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 3 in mask_layers:
            bottom_model.feature_extractor[7] = nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 4 in mask_layers:
            bottom_model.feature_extractor[10] = nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 5 in mask_layers:
            bottom_model.feature_extractor[14] = nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 6 in mask_layers:
            bottom_model.feature_extractor[17] = nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 7 in mask_layers:
            bottom_model.feature_extractor[21] = nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 8 in mask_layers:
            bottom_model.feature_extractor[24] = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 9 in mask_layers:
            bottom_model.feature_extractor[28] = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 10 in mask_layers:
            bottom_model.feature_extractor[31] = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ).to(device)
        if 11 in mask_layers:
            bottom_model.classifier = nn.Linear(
                in_features=512, out_features=10, bias=True
            ).to(device)

    elif model_type == "lenet":
        if 1 in mask_layers:
            bottom_model.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)).to(
                device
            )
        if 2 in mask_layers:
            bottom_model.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)).to(
                device
            )
        if 3 in mask_layers:
            bottom_model.fc1 = nn.Linear(
                in_features=80, out_features=120, bias=True
            ).to(device)
        if 4 in mask_layers:
            bottom_model.fc2 = nn.Linear(
                in_features=120, out_features=84, bias=True
            ).to(device)
        if 5 in mask_layers:
            bottom_model.fc3 = nn.Linear(in_features=84, out_features=10, bias=True).to(
                device
            )

    elif model_type == "mlp":
        raise NotImplementedError()

    else:
        raise ValueError(f"{model_type} is not valid model type.")


# fmt: off
# bottom_model.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
# bottom_model.layer1[0].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
# bottom_model.layer1[0].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
# bottom_model.layer1[1].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
# bottom_model.layer1[1].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)

# bottom_model.layer1[0].conv1.weight.data[:bottom_model.layer1[0].conv1.weight.shape[0]//2] = torch.rand(bottom_model.layer1[0].conv1.weight.shape[0]//2, bottom_model.layer1[0].conv1.weight.shape[1], 3, 3) * 2 - 1
# bottom_model.layer1[0].conv2.weight.data[:bottom_model.layer1[0].conv2.weight.shape[0]//2] = torch.rand(bottom_model.layer1[0].conv2.weight.shape[0]//2, bottom_model.layer1[0].conv2.weight.shape[1], 3, 3) * 2 - 1
# bottom_model.layer1[1].conv1.weight.data[:bottom_model.layer1[1].conv1.weight.shape[0]//2] = torch.rand(bottom_model.layer1[1].conv1.weight.shape[0]//2, bottom_model.layer1[1].conv1.weight.shape[1], 3, 3) * 2 - 1
# bottom_model.layer1[1].conv2.weight.data[:bottom_model.layer1[1].conv2.weight.shape[0]//2] = torch.rand(bottom_model.layer1[1].conv2.weight.shape[0]//2, bottom_model.layer1[1].conv2.weight.shape[1], 3, 3) * 2 - 1

# bottom_model.conv1.apply(init_uniform)
# bottom_model.layer1[0].conv1.apply(init_uniform)
# bottom_model.layer1[0].conv2.apply(init_uniform)
# bottom_model.layer1[1].conv1.apply(init_uniform)
# bottom_model.layer1[1].conv2.apply(init_uniform)
# bottom_model.layer2[0].conv1.apply(init_uniform)
# bottom_model.layer2[0].conv2.apply(init_uniform)
# bottom_model.layer2[1].conv1.apply(init_uniform)
# bottom_model.layer2[1].conv2.apply(init_uniform)
# bottom_model.layer3[0].conv1.apply(init_uniform)
# bottom_model.layer3[0].conv2.apply(init_uniform)
# bottom_model.layer3[1].conv1.apply(init_uniform)
# bottom_model.layer3[1].conv2.apply(init_uniform)
# bottom_model.layer4[0].conv1.apply(init_uniform)
# bottom_model.layer4[0].conv2.apply(init_uniform)
# bottom_model.layer4[1].conv1.apply(init_uniform)
# bottom_model.layer4[1].conv2.apply(init_uniform)
# bottom_model.linear.apply(init_uniform)
# fmt: on
