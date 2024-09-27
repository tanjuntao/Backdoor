import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="bottom model type")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--port", default=20000, type=int, help="active port")
    parser.add_argument("--gpu", default=0, type=int, help="gpu device")
    parser.add_argument(
        "--per_class", default=4, type=int, help="labeled samples per class"
    )
    parser.add_argument(
        "--topk_confident",
        default=50,
        type=int,
        help="number of topk confident samples",
    )
    parser.add_argument(
        "--poison_epochs",
        default="",
        type=str,
        help="at which epochs to poison model training",
    )

    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--attempt", default=0, type=int, help="number of trying")
    parser.add_argument("--target", default=0, type=int, help="target class")
    parser.add_argument(
        "--weight_scale", default=1, type=int, help="weight scaling factor"
    )
    parser.add_argument(
        "--world_size", default=2, type=int, help="number of total participants"
    )
    parser.add_argument(
        "--alpha", default=1.0, type=float, help="trigger vector scaling factor"
    )
    parser.add_argument("--agg", type=str, default="add", help="aggregation function")
    parser.add_argument("--rank", default=0, type=int, help="party rank")
    args = parser.parse_args()
    return args


def get_model_dir():
    args = get_args()
    model_dir = f"/home/1002tjt/models/{args.dataset}_{args.model}"
    if args.weight_scale != 1:
        model_dir += f"_scale{args.weight_scale}"
    if args.agg != "add":
        model_dir += f"_{args.agg}"
    if args.poison_epochs != "":
        model_dir += f"_poison_{args.poison_epochs}"
    if args.attempt != 0:
        model_dir += f"_attempt_{args.attempt}"
    return model_dir


def get_poison_epochs():
    args = get_args()
    poison_epochs = []
    if args.poison_epochs != "":
        poison_epochs = [int(epoch) for epoch in args.poison_epochs.split(",")]
    return poison_epochs
