import numpy as np
import torch
from args_parser import get_args, get_model_dir
from surrogate_fn import finetune_surrogate
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.modelio import TorchModelIO
from linkefl.vfl.nn import PassiveNeuralNetwork

args = get_args()


data_prefix = "."
_epochs = 50
_learning_rate = 0.01
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
    passive_trainset = MediaDataset(
        role=Const.PASSIVE_NAME,
        dataset_name=args.dataset,
        root=_dataset_dir,
        train=True,  # modify this
        download=True,
        valid=True,  # modify this
    )
elif args.model == "mlp":
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    passive_trainset = TorchDataset.buildin_dataset(
        dataset_name=args.dataset,
        role=Const.PASSIVE_NAME,
        root=_dataset_dir,
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
print(colored("1. Finish loading dataset.", "red"))


print("loading model...")
bottom_model = TorchModelIO.load(get_model_dir(), "VFL_passive.model")["model"][
    "bottom"
].to(_device)
cut_layer = TorchModelIO.load(get_model_dir(), "VFL_passive.model")["model"]["cut"].to(
    _device
)

models = {"bottom": bottom_model, "cut": cut_layer}
print("Done.")

passive_party = PassiveNeuralNetwork(
    epochs=_epochs,
    batch_size=256,
    learning_rate=_learning_rate,
    models=models,
    optimizers=None,
    messenger=None,
    cryptosystem=None,
    logger=_logger,
    device=_device,
    num_workers=1,
    val_freq=1,
    random_state=_random_state,
    saving_model=False,
    schedulers=None,
    args=args,
)
total_embeddings = passive_party.validate_alone(passive_trainset)

# filter topk confident embedding
topk_indices = finetune_surrogate(_epochs - 1, args)
target_embeddings = total_embeddings.cpu().numpy()[topk_indices]
print(target_embeddings.shape)

"""
# filter target embedding
labels = passive_trainset.labels.numpy()
target_embeddings = total_embeddings.cpu().numpy()[labels == args.target]
print(f"target embedding size: {target_embeddings.shape}")

# random select most relevant target embedding
np.random.seed(0)
total = len(target_embeddings)
random_idxes = np.random.permutation(total)[:int(total * 0.1)]  # random select 10%
random_embeddings = target_embeddings[random_idxes]
print(f"random embedding size: {random_embeddings.shape}")
# ref: https://stackoverflow.com/a/43073761/8418540
random_df = pd.DataFrame(np.transpose(random_embeddings))
corr_matrix = random_df.corr().abs()  # corr() is column wise
solution = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False))
most_relevant_idxes = set()
for index, value in solution.items():
    most_relevant_idxes.add(index[0])
    most_relevant_idxes.add(index[1])
    if len(most_relevant_idxes) >= int(len(random_embeddings) * 0.1):  # 10% most relevant
        break
print(f"length of final indexes: {len(most_relevant_idxes)}")
target_embeddings = random_embeddings[np.array(list(most_relevant_idxes))]
"""


# calculate mean
target_embedding_mean = args.alpha * np.mean(target_embeddings, axis=0)
print("trigger embedding: ")
print(target_embedding_mean.shape, target_embedding_mean.dtype)
print(target_embedding_mean)
import pickle

with open(f"{get_model_dir()}/target_embedding_mean_class_{args.target}.np", "wb") as f:
    pickle.dump(target_embedding_mean, f)


# usage
# python3 target_embedding_mean.py --model resnet18 --dataset cifar10 --gpu 0 --target 0 --agg add --alpha 1
