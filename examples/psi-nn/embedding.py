import matplotlib.pyplot as plt
import numpy as np
import torch
from args_parser import get_args, get_mask_layers, get_model_dir
from mask import layer_masking
from sklearn.manifold import TSNE
from termcolor import colored
from torch import nn

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.modelio import TorchModelIO
from linkefl.vfl.nn import ActiveNeuralNetwork

args = get_args()


data_prefix = "."
_epochs = 50
_learning_rate = 0.01
_loss_fn = nn.CrossEntropyLoss()
_random_state = None
_device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
_logger = logger_factory(role=Const.ACTIVE_NAME)
if args.dataset in ("cifar10", "cinic10"):
    topk = 1
    if args.dataset == "cifar10":
        _dataset_dir = f"{data_prefix}/data"
    else:
        _dataset_dir = f"{data_prefix}/data/CINIC10"
elif args.dataset == "cifar100":
    topk = 5
    _dataset_dir = f"{data_prefix}/data"
elif args.dataset in ("mnist", "fashion_mnist", "svhn"):
    topk = 1
    _dataset_dir = f"{data_prefix}/data"
elif args.dataset in ("tab_mnist", "tab_fashion_mnist"):
    topk = 1
    _dataset_dir = f"{data_prefix}/data"
elif args.dataset == "criteo":
    topk = 1
    _dataset_dir = f"{data_prefix}/data"
else:
    raise ValueError(f"{args.dataset} is not valid dataset.")


# Load dataset
if args.model in ("resnet18", "vgg13", "lenet"):
    active_testset = MediaDataset(
        role=Const.PASSIVE_NAME,
        dataset_name=args.dataset,
        root=_dataset_dir,
        train=False,
        download=True,
    )
elif args.model == "mlp":
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    active_testset = TorchDataset.buildin_dataset(
        dataset_name=args.dataset,
        role=Const.PASSIVE_NAME,
        root=_dataset_dir,
        train=False,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
print(colored("1. Finish loading dataset.", "red"))


# Visualization
print("loading model...")
# bottom_model = TorchModelIO.load(get_model_dir(), "VFL_active.model")["model"][
#     "bottom"
# ].to(_device)
from linkefl.modelzoo import ResNet18

bottom_model = ResNet18(in_channel=3, num_classes=10).to(_device)
cut_layer = TorchModelIO.load(get_model_dir(), "VFL_active.model")["model"]["cut"].to(
    _device
)
top_model = TorchModelIO.load(get_model_dir(), "VFL_active.model")["model"]["top"].to(
    _device
)

# mask layers
bottom_model = layer_masking(
    "resnet18",
    bottom_model,
    get_mask_layers(),
    device=_device,
    dataset=args.dataset,
)
models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}
print("Done.")

active_party = ActiveNeuralNetwork(
    epochs=_epochs,
    batch_size=None,
    learning_rate=_learning_rate,
    models=models,
    optimizers=None,
    loss_fn=_loss_fn,
    messengers=None,
    logger=_logger,
    device=_device,
    num_workers=1,
    val_freq=1,
    topk=topk,
    random_state=_random_state,
    saving_model=False,
    schedulers=None,
)
active_party.models = models
_, total_embeddings = active_party.validate_alone(active_testset)

# tsne visualization
tsne = TSNE(n_components=2, random_state=0)
transformed_embeddings = tsne.fit_transform(total_embeddings.cpu().numpy())
test_targets = np.array(active_testset.buildin_dataset.targets)
for i in range(10):
    plt.scatter(
        transformed_embeddings[test_targets == i, 0],
        transformed_embeddings[test_targets == i, 1],
        label=str(i),
    )
plt.axis("off")
# plt.legend()
# plt.savefig(
#     f"figures/embed_{get_model_dir().split('/')[-1]}_layer{args.layer}.pdf",
#     bbox_inches='tight'
# )
plt.savefig(
    f"figures/embed_{get_model_dir().split('/')[-1]}_scratch.pdf", bbox_inches="tight"
)
plt.close()


# embedding vis when some layers are masked
# python3 embedding --model resnet18 --dataset cifar10 --layer 1,2,3,4
