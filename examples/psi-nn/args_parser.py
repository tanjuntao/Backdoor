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
    parser.add_argument("--layer", default="", type=str, help="number of mask layer")
    parser.add_argument("--attempt", default=0, type=int, help="number of trying")
    parser.add_argument("--defense", default="", type=str, help="defense method")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument(
        "--noise_scale", default=1e-5, type=float, help="ng noise scale"
    )
    parser.add_argument(
        "--preserved_ratio", default=0.1, type=float, help="cg preserved ratio"
    )
    parser.add_argument("--bins_num", default=100, type=int, help="dg bins num")
    parser.add_argument("--eps", default=1.0, type=float, help="labeldp epsilon")
    parser.add_argument(
        "--sigma2", default=1.0, type=float, help="fedpass secret key scale square"
    )
    parser.add_argument(
        "--mid_weight", default=1.0, type=float, help="weight of vib loss in mid"
    )
    parser.add_argument("--dcor_weight", default=1.0, type=float, help="dcor weight")
    args = parser.parse_args()
    return args


def get_model_dir():
    args = get_args()
    model_dir = f"/storage/1002tjt/MC-Attack/{args.dataset}_{args.model}"
    if args.defense == "ng":
        model_dir += f"_ng_{args.noise_scale}"
    elif args.defense == "cg":
        model_dir += f"_cg_{args.preserved_ratio}"
    elif args.defense == "dg":
        model_dir += f"_dg_{args.bins_num}"
    elif args.defense == "fedpass":
        model_dir += f"_fedpass_{args.sigma2}"
    elif args.defense == "labeldp":
        model_dir += f"_labeldp_{args.eps}"
    elif args.defense == "dcor":
        model_dir += f"_dcor_{args.dcor_weight}"
    elif args.defense == "mid":
        model_dir += f"_mid_{args.mid_weight}"
    elif args.defense == "":
        if args.attempt != 0:
            model_dir += f"_attempt_{args.attempt}"
    else:
        raise ValueError(f"{args.defense} is not supported.")
    return model_dir


def get_mask_layers():
    args = get_args()
    mask_layers = []
    if args.layer != "":
        mask_layers = [int(layer) for layer in args.layer.split(",")]
    return mask_layers
