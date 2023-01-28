import numpy as np
import shap
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from xgboost import XGBClassifier

# MNIST dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# trainset = datasets.MNIST(root='data',
#                           train=True,
#                           download=True,
#                           transform=transform)
# testset = datasets.MNIST(root='data',
#                           train=False,
#                           download=True,
#                           transform=transform)

# FashionMNINST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)
testset = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)
n_samples = len(trainset)
n_features = 28 * 28
_ids = torch.arange(n_samples)
_labels = trainset.targets


imgs = []
for i in range(n_samples):
    image, _ = trainset[i]
    img = image.view(1, -1)
    imgs.append(img)
_feats = torch.Tensor(n_samples, n_features)
torch.cat(imgs, out=_feats)

x_train = _feats.numpy()
y_train = _labels.numpy()
model = XGBClassifier()
print("start training...")
model.fit(x_train, y_train)
print("model fitted...")

# xgboost
# importances = model.feature_importances_
# ranking = np.argsort(importances)[::-1]
# print(importances.shape)
# print(ranking.shape)
# print(ranking[:10])

# shap
explainer = shap.Explainer(model)
shap_values = explainer(_feats)
print("shape values computed...")
print("shap values shape: {}".format(shap_values.shape))
importances = np.mean(np.mean(np.abs(shap_values.values), axis=2), axis=0)
ranking = np.argsort(importances)[::-1]
print(ranking.tolist())
