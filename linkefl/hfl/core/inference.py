import torch
from torch.utils.data import DataLoader

from linkefl.modelio import TorchModelIO


def inference_hfl(
    dataset,
    model_name,
    model_dir="./models",
    loss_fn=None,
    infer_step=64,
    device=torch.device("cpu"),
    optimizer_arch=None,
):
    # 加载模型
    model = TorchModelIO.load(model_dir=model_dir, model_name=model_name)["model"]
    # 加载数据
    dataloader = DataLoader(dataset, batch_size=infer_step, shuffle=False)
    # num_batches = len(dataloader)
    # test_loss = 0
    # correct = 0
    # labels, probs = np.array([]), np.array([])  # used for computing AUC score
    preds = []

    if model_name == "HFLLinReg":
        with torch.no_grad():
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device).to(torch.long)
                log_probs = model(data)
                # test_loss += self.lossfunction(
                #     log_probs,
                #     target,
                #     reduction='sum'
                # ).item()
                # test_loss += loss_fn(log_probs, target).item()
                # y_pred = log_probs.data.max(1, keepdim=True)[1]
                # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
                preds.extend(log_probs.numpy().tolist())
        # test_loss /= num_batches
        # scores = {"loss": test_loss, "preds": preds}
        scores = {"preds": preds}
    else:
        with torch.no_grad():
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device).to(torch.long)
                log_probs = model(data)
                # test_loss += self.lossfunction(
                #     log_probs,
                #     target,
                #     reduction='sum'
                # ).item()
                # test_loss += loss_fn(log_probs, target).item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
                preds.extend(y_pred.numpy().tolist())
        # test_loss /= num_batches
        # acc = correct / len(dataloader.dataset)
        # scores = {"acc": acc, "auc": 0, "loss": test_loss, "preds": preds}
        scores = {"preds": preds}

    return scores
