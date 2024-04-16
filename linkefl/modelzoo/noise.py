from torch.distributions.normal import Normal


def layer_noising(model_type, bottom_model, noisy_layers, device, sigma=0.01):
    gassian_noise = Normal(0.0, sigma)
    if model_type == "resnet18":
        for layer in noisy_layers:
            if layer == "conv1.weight":
                weight = bottom_model.conv1.weight.data
            elif layer == "layer1.0.conv1.weight":
                weight = bottom_model.layer1[0].conv1.weight.data
            elif layer == "layer1.0.conv2.weight":
                weight = bottom_model.layer1[0].conv2.weight.data
            elif layer == "layer1.1.conv1.weight":
                weight = bottom_model.layer1[1].conv1.weight.data
            elif layer == "layer1.1.conv2.weight":
                weight = bottom_model.layer1[1].conv2.weight.data
            elif layer == "layer2.0.conv1.weight":
                weight = bottom_model.layer2[0].conv1.weight.data
            elif layer == "layer2.0.conv2.weight":
                weight = bottom_model.layer2[0].conv2.weight.data
            elif layer == "layer2.1.conv1.weight":
                weight = bottom_model.layer2[1].conv1.weight.data
            elif layer == "layer2.1.conv2.weight":
                weight = bottom_model.layer2[1].conv2.weight.data
            elif layer == "layer3.0.conv1.weight":
                weight = bottom_model.layer3[0].conv1.weight.data
            elif layer == "layer3.0.conv2.weight":
                weight = bottom_model.layer3[0].conv2.weight.data
            elif layer == "layer3.1.conv1.weight":
                weight = bottom_model.layer3[1].conv1.weight.data
            elif layer == "layer3.1.conv2.weight":
                weight = bottom_model.layer3[1].conv2.weight.data
            elif layer == "layer4.0.conv1.weight":
                weight = bottom_model.layer4[0].conv1.weight.data
            elif layer == "layer4.0.conv2.weight":
                weight = bottom_model.layer4[0].conv2.weight.data
            elif layer == "layer4.1.conv1.weight":
                weight = bottom_model.layer4[1].conv1.weight.data
            elif layer == "layer4.1.conv2.weight":
                weight = bottom_model.layer4[1].conv2.weight.data
            elif layer == 18:
                weight = bottom_model.linear.weight.data
            else:
                raise ValueError(f"wrong layer index: {layer} for model {model_type}")
            weight.add_(gassian_noise.sample(weight.shape).to(device))

    elif model_type == "vgg13":
        for layer in noisy_layers:
            if layer == "feature_extractor.0.weight":
                weight = bottom_model.feature_extractor[0].weight.data
            elif layer == "feature_extractor.3.weight":
                weight = bottom_model.feature_extractor[3].weight.data
            elif layer == "feature_extractor.7.weight":
                weight = bottom_model.feature_extractor[7].weight.data
            elif layer == "feature_extractor.10.weight":
                weight = bottom_model.feature_extractor[10].weight.data
            elif layer == "feature_extractor.14.weight":
                weight = bottom_model.feature_extractor[14].weight.data
            elif layer == "feature_extractor.17.weight":
                weight = bottom_model.feature_extractor[17].weight.data
            elif layer == "feature_extractor.21.weight":
                weight = bottom_model.feature_extractor[21].weight.data
            elif layer == "feature_extractor.24.weight":
                weight = bottom_model.feature_extractor[24].weight.data
            elif layer == "feature_extractor.28.weight":
                weight = bottom_model.feature_extractor[28].weight.data
            elif layer == "feature_extractor.31.weight":
                weight = bottom_model.feature_extractor[31].weight.data
            elif layer == 11:
                weight = bottom_model.classifier.weight.data
            else:
                raise ValueError(f"wrong layer index: {layer} for model {model_type}")
            weight.add_(gassian_noise.sample(weight.shape).to(device))

    elif model_type == "lenet5":
        for layer in noisy_layers:
            if layer == "conv1.weight":
                weight = bottom_model.conv1.weight.data
            elif layer == "conv2.weight":
                weight = bottom_model.conv2.weight.data
            elif layer == "fc1.weight":
                weight = bottom_model.fc1.weight.data
            elif layer == "fc2.weight":
                weight = bottom_model.fc2.weight.data
            elif layer == "fc3.weight":
                weight = bottom_model.fc3.weight.data
            else:
                raise ValueError(f"wrong layer index: {layer} for model {model_type}")
            weight.add_(gassian_noise.sample(weight.shape).to(device))

    elif model_type == "mlp":
        for layer in noisy_layers:
            if layer == "sequential.0.weight":
                weight = bottom_model.sequential[0].weight.data
            elif layer == "sequential.3.weight":
                weight = bottom_model.sequential[3].weight.data
            elif layer == "sequential.6.weight":
                weight = bottom_model.sequential[6].weight.data
            else:
                raise ValueError(f"wrong layer index: {layer} for model {model_type}")
            weight.add_(gassian_noise.sample(weight.shape).to(device))
    else:
        raise ValueError(f"invalid model type: {model_type}")

    return bottom_model
